// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <vector>

namespace xllama {

struct PullModelDescriptor {
    std::string name;
    std::string kind;
    std::string role;
    std::string source_url;
    bool trusted_catalogue = false;
    std::vector<std::string> sha256_pins;
};

inline std::string resolve_pull_model_name(const std::string& name) {
    if (name == "bge-m3")
        return "embed-bge-m3";
    if (name == "nomic-embed-text-v2-moe")
        return "embed-nomic-v2-moe";
    if (name == "qwen3-embedding:4b")
        return "embed-qwen3-4b";
    return name;
}

inline bool valid_pull_sha256(const std::string& pin) {
    if (pin.size() != 64)
        return false;
    for (const char c : pin) {
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')))
            return false;
    }
    return true;
}

inline bool api_pull_model_allowed(const PullModelDescriptor& model) {
    if (!model.trusted_catalogue || model.name.empty() ||
        model.source_url.rfind("https://", 0) != 0 || model.sha256_pins.empty())
        return false;
    if (model.kind != "gguf" && model.kind != "ort-genai")
        return false;
    if (model.role == "diffusion" || model.role == "image")
        return false;
    for (const auto& pin : model.sha256_pins) {
        if (!valid_pull_sha256(pin))
            return false;
    }
    return true;
}

} // namespace xllama
