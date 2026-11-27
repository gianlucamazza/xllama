// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// What the LAN API tells a client before it sends a request: GET /health
// (build version, contract version, the routes this build serves) and the
// per-model fields of GET /v1/models (context, kind, role, capabilities).
// WinRT-free so the shapes are host-tested; uwp/api-server.cpp fills the
// values from the package identity and the catalogue.
#pragma once

#include "xllama/model_sampling.h"

#include <cstdint>
#include <string>
#include <vector>

namespace xllama {

// Bumped when a documented response shape or status mapping changes in a way
// a client must branch on (docs/api-endpoint.md, Discovery).
inline constexpr int kApiContractVersion = 1;

// Routes and response features this build serves.
struct ApiCapabilities {
    bool tools = false;             // tool proposals on /v1/chat/completions
    bool tools_stream = false;      // the same, with "stream": true
    bool embeddings = false;        // /api/embed, /api/embeddings, /v1/embeddings
    bool images = false;            // /v1/images/generations
    bool pull = false;              // /api/pull
    bool preferences = false;       // /v1/preferences
    bool training_status = false;   // /v1/training/status
    bool reasoning_content = false; // message.reasoning_content on chat replies
    bool reasoning_tokens = false;  // usage.completion_tokens_details.reasoning_tokens
    bool cached_tokens = false;     // usage.prompt_tokens_details.cached_tokens
    bool model_sampling = false;    // /v1/models "sampling"; omitted fields take it
    bool ollama_chat = false;       // native /api/chat and /api/generate (api_ollama.h)
};

// The capabilities of the compiled build: GGUF paths need XLLAMA_USE_LLAMA,
// images need the ORT diffusion pipeline (XLLAMA_USE_ORT).
ApiCapabilities build_api_capabilities();

// The llama.cpp commit this build compiles (common/build-info.h); empty
// without XLLAMA_USE_LLAMA. Only the commit: the build number counts the
// submodule's history, and a shallow checkout (CI) has one commit, so
// llama_build_info() reads "b1-<commit>" whatever the pin.
std::string build_llama_cpp();

// {"status":"ok","service":"xllama","version":…,"api":…,"capabilities":{…},
//  "sampling":[…],"runtime":{"llama_cpp":…,"patches":[…]}}. The first two
// fields are the original body and stay first. "sampling" lists the chat
// fields of api_sampling.h (#609). "runtime" (#611) names the llama.cpp pin
// and the patches applied to it (llama_patches.h); it is left out when
// |llama_cpp| is empty (a build without llama.cpp).
std::string health_json(const std::string& version, const ApiCapabilities& caps,
                        const std::string& llama_cpp);

struct ModelInfo {
    std::string id;
    std::string display; // catalogue display name; empty when not catalogued
    std::string kind;    // "gguf", "ort-genai" or "diffusion"; empty when unknown
    std::string role;    // catalogue role ("", "coding", "embedding")
    int n_ctx = 0;       // context the session uses (resident one, else catalogue)
    bool active = false; // the resident session's model
    bool tools = false;  // its chat template renders a tool schema
    bool reasoning = false;
    bool embedding = false;
    bool ollama_chat = false; // /api/chat and /api/generate serve it (ollama_chat_model)
    // The sampling a request that sets none of the fields gets, and where each
    // value came from (model_sampling.h). Written for chat models only.
    bool has_sampling = false;
    ResolvedSampling sampling;
};

// OpenAI {"object":"list","data":[…]}: id/object/created/owned_by first,
// then xllama's non-standard fields (clients ignore unknown keys). A chat
// model's "sampling" is {"temperature":…,"top_p":…,"top_k":…,
// "repetition_penalty":…,"source":{<field>:"default"|"model"}}.
std::string models_list_json(const std::vector<ModelInfo>& models, std::int64_t created);

} // namespace xllama
