// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// The one place that turns an n_gpu_layers request into llama.cpp params for
// the d3d12 backend (docs/gguf-gpu-decode.md, D2b). Used by run_inference_llama
// (CLI, headless bench) and the persistent LlamaSession (GUI, LAN API).
#pragma once

#include "ggml-cpu.h"
#include "llama.h"
#include "xllama/ggml_d3d12.h"
#include "xllama/platform.h"

#include <string>

namespace xllama {

// Sets mparams.devices / n_gpu_layers and returns the layers actually
// offloaded: 0 when none were asked for or the device is unavailable, in
// which case the model loads exactly as before (CPU, no GPU device listed).
inline int apply_gguf_gpu_layers(int requested, llama_model_params& mparams) {
    static ggml_backend_dev_t no_devices[] = {nullptr};
    mparams.n_gpu_layers = 0;
    mparams.devices = no_devices;
    if (requested <= 0)
        return 0;
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
