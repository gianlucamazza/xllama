// Copyright (c) 2024 Venere Labs
// SPDX-License-Identifier: MIT
// Per-workload EP routing policy (ORT GenAI only). GGUF models disable routing
// and KV reuse at the UI layer; this header encodes the token threshold only.
#pragma once

#include <string>

namespace xllama {

enum class RoutingMode {
    CpuOnly = 0,
    GpuOnly = 1,
    Auto = 2,
};

struct RoutingSettings {
    RoutingMode mode = RoutingMode::CpuOnly;
    int token_threshold = 600; // measured crossover between ~285 and ~1050 tok
    std::string cpu_model;
    std::string gpu_model;
};

struct RoutingDecision {
    std::string active_model;
    bool use_gpu = false;
    int token_count = 0;
};

// Decide which model directory to load for the first turn of a conversation.
// |gpu_available| must reflect IsModelProvisioned(gpu_model) — callers gate UX.
inline RoutingDecision decide_routing(const RoutingSettings& s, int n_tok, bool base_is_gguf,
                                      bool gpu_available) {
    RoutingDecision d;
    d.token_count = n_tok;
    if (base_is_gguf || s.mode == RoutingMode::CpuOnly) {
        d.active_model = s.cpu_model;
        d.use_gpu = false;
        return d;
    }
    if (s.mode == RoutingMode::GpuOnly) {
        d.active_model = gpu_available ? s.gpu_model : s.cpu_model;
        d.use_gpu = gpu_available;
        return d;
    }
    // Auto
    const bool use_gpu = gpu_available && n_tok > s.token_threshold;
    d.use_gpu = use_gpu;
    d.active_model = use_gpu ? s.gpu_model : s.cpu_model;
    return d;
}

// Feature gates by catalogue kind (mirrors MainPage capability matrix).
inline bool routing_allowed_for_kind(const std::wstring& kind) {
    return kind != L"gguf";
}

inline bool kv_reuse_allowed_for_kind(const std::wstring& kind) {
    // GGUF (llama.cpp) now supports KV reuse via a persistent llama_context
    // (LlamaSession), so it is allowed alongside ORT-GenAI. (Routing stays
    // ORT-only — the llama.cpp UWP build is CPU-only, no GPU model to route to.)
    (void)kind;
    return true;
}

// ORT GenAI continuous decoding (KV reuse) is CPU-only today; DirectML rejects
// AppendTokenSequences on a persistent generator ("Continuous decoding is not
// supported on the selected device type (DirectML)").
inline bool kv_reuse_supported_for_model(const std::string& active_model) {
    return active_model.find("dml") == std::string::npos;
}

} // namespace xllama