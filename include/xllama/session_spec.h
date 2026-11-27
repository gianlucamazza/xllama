// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
// One resolver from catalogue data to session load parameters (ADR 0007, audit
// F03). GUI chat, LAN chat, embeddings and pull all build their session here, so
// a model loads the same way whichever surface asks first, and the hub can tell
// whether the resident session is the one being asked for. WinRT-free: the UWP
// side only copies its wide-string manifest entry into CatalogueLoad.
#pragma once

#include "xllama/routing_policy.h" // catalogue_n_ctx, resolve_gguf_gpu_layers
#include "xllama/session.h"

#include <filesystem>
#include <string>

namespace xllama {

// The catalogue fields that decide how a model is loaded.
struct CatalogueLoad {
    bool found = false; // false: not in the catalogue, load with the defaults
    bool gguf = false;  // kind "gguf" (llama.cpp); else ORT GenAI
    int n_ctx = 0;      // raw catalogue value, bounded by catalogue_n_ctx
    int gpu_layers = 0; // catalogue default; the operator knob may override it
    std::string lora;   // adapter file relative to the model directory
    float lora_scale = 1.0f;
};

// Operator override for llama.cpp thread count on interactive GGUF sessions
// (GUI and LAN API). Absent, 0, any non-positive value, or a value above
// kGgufNThreadsMax keeps SessionParams::n_threads at 0 (auto). Any other count
// is used as-is: the UWP auto cap of 6 (platform.cpp) is deliberately not
// applied, so an operator can measure past it.
inline constexpr int kGgufNThreadsKnobAbsent = -1;
// GGML_MAX_N_THREADS at the pinned llama.cpp; ggml-cpu sizes fixed arrays by
// it. session.cpp static_asserts the two agree.
inline constexpr int kGgufNThreadsMax = 512;

inline constexpr int resolve_gguf_n_threads(int knob) {
    return (knob < 1 || knob > kGgufNThreadsMax) ? 0 : knob;
}

// `model_dir` is where the model's files live (resolve_model_path on the bare
// name); the adapter is resolved against it. `gpu_layers_knob` is the operator
// override read by the caller (-1 = none, see resolve_gguf_gpu_layers).
// `threads_knob` is the same for llama.cpp thread count (0 = auto when
// absent, zero, or invalid).
inline SessionParams resolve_session_params(const std::string& model, const CatalogueLoad& c,
                                            const std::string& model_dir, int gpu_layers_knob,
                                            int threads_knob = kGgufNThreadsKnobAbsent) {
    SessionParams sp;
    sp.model_path = model;
    sp.n_ctx = kDefaultNCtx;
    sp.n_threads = resolve_gguf_n_threads(threads_knob);
    if (!c.found)
        return sp;
    sp.n_ctx = catalogue_n_ctx(c.n_ctx, c.gguf);
    if (!c.gguf)
        return sp;
    sp.backend = Backend::LlamaCpp;
    sp.n_gpu_layers = resolve_gguf_gpu_layers(c.gpu_layers, gpu_layers_knob);
    if (!c.lora.empty()) {
        sp.lora_path = model_dir;
        constexpr char sep = static_cast<char>(std::filesystem::path::preferred_separator);
        if (!sp.lora_path.empty() && sp.lora_path.back() != sep && sp.lora_path.back() != '/')
            sp.lora_path.push_back(sep);
        sp.lora_path += c.lora;
        sp.lora_scale = c.lora_scale;
    }
    return sp;
}

// Two parameter sets load the same thing into RAM. Sampling is per generate and
// threads/batch are tuning knobs, so neither is part of the load identity.
inline bool same_load(const SessionParams& a, const SessionParams& b) {
    return a.model_path == b.model_path && a.backend == b.backend && a.n_ctx == b.n_ctx &&
           a.n_gpu_layers == b.n_gpu_layers && a.lora_path == b.lora_path &&
           a.lora_scale == b.lora_scale && a.kv_q8 == b.kv_q8 &&
           a.prefix_snapshots == b.prefix_snapshots;
}

} // namespace xllama
