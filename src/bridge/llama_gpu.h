// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// The one place that turns an n_gpu_layers request into llama.cpp params for
// the d3d12 backend (docs/gguf-gpu-decode.md, D2b). Used by run_inference_llama
// (CLI, headless bench) and the persistent LlamaSession (GUI, LAN API).
#pragma once

#include "ggml-cpu.h"
#include "gguf.h"
#include "llama.h"
#include "xllama/ggml_d3d12.h"
#include "xllama/platform.h"

#include <cstring>
#include <string>

namespace xllama {

// llama/ggml log lines worth keeping in xllama.log when layers run on d3d12:
// placement and buffer sizes, graph splits, and every warning or error.
inline void gguf_gpu_log(ggml_log_level level, const char* text, void*) {
    if (!text)
        return;
    const bool keep = level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR ||
                      std::strstr(text, "buffer size") || std::strstr(text, "offload") ||
                      std::strstr(text, "graph splits") || std::strstr(text, "D3D12");
    if (keep)
        log_output(std::string("[llama] ") + text);
}

// Working set at a load stage, for every GGUF load: the CPU and d3d12 runs log
// the same stages, so a peak difference can be attributed (#309).
inline void log_gguf_ws(const char* stage) {
    log_output(std::string("[xllama] ws ") + stage + ": " + std::to_string(working_set_mb()) +
               " MB\n");
}

// token_embd's type and whether the model has its own output.weight, read from
// the GGUF header only (no tensor data). False when the header cannot be read.
inline bool gguf_embedding_info(const std::string& path, ggml_type* embd_type, bool* has_output) {
    gguf_init_params gp = {};
    gp.no_alloc = true;
    gp.ctx = nullptr;
    gguf_context* g = gguf_init_from_file(path.c_str(), gp);
    if (!g)
        return false;
    const int64_t embd = gguf_find_tensor(g, "token_embd.weight");
    const bool ok = embd >= 0;
    if (ok) {
        *embd_type = gguf_get_tensor_type(g, embd);
        *has_output = gguf_find_tensor(g, "output.weight") >= 0;
    }
    gguf_free(g);
    return ok;
}

// Sets mparams.devices / n_gpu_layers and returns the layers actually
// offloaded: 0 when none were asked for or the device is unavailable, in
// which case the model loads exactly as before (CPU, no GPU device listed).
inline int apply_gguf_gpu_layers(int requested, llama_model_params& mparams,
                                 const std::string& model_path) {
    static ggml_backend_dev_t no_devices[] = {nullptr};
    mparams.n_gpu_layers = 0;
    mparams.devices = no_devices;
    if (requested <= 0)
        return 0;
    log_gguf_ws("before d3d12");
    if (!ggml_d3d12_register()) {
        log_output("[xllama] gguf gpu layers: d3d12 backend unavailable, using the CPU\n");
        return 0;
    }
    static ggml_backend_dev_t d3d12[] = {ggml_backend_dev_by_name("D3D12"), nullptr};
    if (!d3d12[0]) {
        log_output("[xllama] gguf gpu layers: no D3D12 device registered, using the CPU\n");
        return 0;
    }
    mparams.devices = d3d12;
    mparams.n_gpu_layers = requested;
    // D3D12_Host is also the device's host buft, so llama.cpp puts the CPU
    // backend's compute buffer there and splits share activations in place.
    // Keep it out of the CPU *weight* list: weights left on the CPU stay in the
    // CPU / repack bufts exactly as on a CPU-only load.
    mparams.no_host = true;
    // A tied token_embd also serves as the lm_head: llama.cpp would keep the
    // input copy on the CPU and duplicate it into D3D12_Weights (244 MiB on
    // Coder-3B). Placing it in D3D12_Weights, where GET_ROWS also runs, leaves
    // one copy: the duplicate finds the original in the same buffer context.
    ggml_type embd_type = GGML_TYPE_COUNT;
    bool has_output = true;
    if (gguf_embedding_info(model_path, &embd_type, &has_output) &&
        d3d12_place_tied_embedding(embd_type, has_output)) {
        static llama_model_tensor_buft_override embd_on_gpu[] = {{"^token_embd\\.weight$", nullptr},
                                                                 {nullptr, nullptr}};
        embd_on_gpu[0].buft = ggml_d3d12_weights_buft();
        mparams.tensor_buft_overrides = embd_on_gpu;
        log_output("[xllama] gguf gpu layers: tied token_embd in D3D12_Weights\n");
    }
    log_gguf_ws("after d3d12 init");
    llama_log_set(gguf_gpu_log, nullptr);
    log_output("[xllama] gguf gpu layers: " + std::to_string(requested) + " on D3D12\n");
    return requested;
}

// KV cache and attention stay on the CPU in ordinary memory: the backend only
// runs weight matmuls.
inline void apply_gguf_gpu_context(int applied_layers, llama_context_params& cparams) {
    if (applied_layers > 0)
        cparams.offload_kqv = false;
}

// Persistent CPU threadpools for a context that alternates CPU and d3d12
// splits. Without one attached, ggml-cpu builds and joins a disposable pool on
// every graph_compute (ggml-cpu.c) — once per token on a CPU-only graph, but
// ~50-70 times per token once matmuls run on the GPU: lfm25-350m decoded at
// 26.7 tok/s with 6 threads and 66.9 with 1 (D2b smoke, CI 1.6.0.1130). The
// CPU-only path keeps llama's default so its baseline does not move. Must
// outlive every context it is attached to (declare it before the context).
class GgufCpuThreadpools {
  public:
    GgufCpuThreadpools() = default;
    GgufCpuThreadpools(const GgufCpuThreadpools&) = delete;
    GgufCpuThreadpools& operator=(const GgufCpuThreadpools&) = delete;
    ~GgufCpuThreadpools() {
        reset();
    }

    void attach(int applied_layers, llama_context* ctx, int n_threads, int n_threads_batch) {
        if (applied_layers <= 0 || !ctx || n_threads <= 0)
            return;
        if (!tp_) {
            ggml_threadpool_params p = ggml_threadpool_params_default(n_threads);
            tp_ = ggml_threadpool_new(&p);
            if (n_threads_batch > 0 && n_threads_batch != n_threads) {
                ggml_threadpool_params pb = ggml_threadpool_params_default(n_threads_batch);
                tp_batch_ = ggml_threadpool_new(&pb);
            }
        }
        if (tp_)
            llama_attach_threadpool(ctx, tp_, tp_batch_ ? tp_batch_ : tp_);
    }

    void reset() {
        if (tp_batch_)
            ggml_threadpool_free(tp_batch_);
        if (tp_)
            ggml_threadpool_free(tp_);
        tp_ = tp_batch_ = nullptr;
    }

  private:
    ggml_threadpool_t tp_ = nullptr;
    ggml_threadpool_t tp_batch_ = nullptr;
};

} // namespace xllama
