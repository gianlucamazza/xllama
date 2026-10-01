// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// The one place that turns an n_gpu_layers request into llama.cpp params for
// the d3d12 backend (docs/gguf-gpu-decode.md, D2b). Used by run_inference_llama
// (CLI, headless bench) and the persistent LlamaSession (GUI, LAN API).
#pragma once

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

} // namespace xllama
