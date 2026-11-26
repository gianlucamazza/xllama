// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
#include "xllama/api_discovery.h"

#include "xllama/json_utils.h"

namespace xllama {

namespace {

std::string quoted(const std::string& s) {
    return "\"" + json_escape(s) + "\"";
}

const char* boolean(bool b) {
    return b ? "true" : "false";
}

} // namespace

ApiCapabilities build_api_capabilities() {
    ApiCapabilities caps;
#ifdef XLLAMA_USE_LLAMA
    caps.tools = true;
    caps.tools_stream = true;
    caps.embeddings = true;
#endif
#ifdef XLLAMA_USE_ORT
    caps.images = true;
#endif
    caps.pull = true;
    caps.preferences = true;
    caps.training_status = true;
    caps.reasoning_content = true; // #564: JSON, SSE and tool replies
    caps.reasoning_tokens = true;  // #585: the same replies' usage
    caps.cached_tokens = true;     // #595: prompt tokens the KV prefix reuse skipped
    return caps;
}

std::string health_json(const std::string& version, const ApiCapabilities& caps) {
    std::string out = "{\"status\":\"ok\",\"service\":\"xllama\",\"version\":" + quoted(version) +
                      ",\"api\":" + std::to_string(kApiContractVersion) + ",\"capabilities\":{";
    out += "\"tools\":" + std::string(boolean(caps.tools));
    out += ",\"tools_stream\":" + std::string(boolean(caps.tools_stream));
    out += ",\"embeddings\":" + std::string(boolean(caps.embeddings));
    out += ",\"images\":" + std::string(boolean(caps.images));
    out += ",\"pull\":" + std::string(boolean(caps.pull));
    out += ",\"preferences\":" + std::string(boolean(caps.preferences));
    out += ",\"training_status\":" + std::string(boolean(caps.training_status));
    out += ",\"reasoning_content\":" + std::string(boolean(caps.reasoning_content));
    out += ",\"reasoning_tokens\":" + std::string(boolean(caps.reasoning_tokens));
    out += ",\"cached_tokens\":" + std::string(boolean(caps.cached_tokens));
    out += "}}";
    return out;
}

std::string models_list_json(const std::vector<ModelInfo>& models, std::int64_t created) {
    std::string out = "{\"object\":\"list\",\"data\":[";
    for (std::size_t i = 0; i < models.size(); ++i) {
        const ModelInfo& m = models[i];
        if (i)
            out += ',';
        out += "{\"id\":" + quoted(m.id) +
               ",\"object\":\"model\",\"created\":" + std::to_string(created) +
               ",\"owned_by\":\"xllama\"";
        if (m.active)
            out += ",\"active\":true";
        if (!m.display.empty())
            out += ",\"display\":" + quoted(m.display);
        if (!m.kind.empty())
            out += ",\"kind\":" + quoted(m.kind);
        out += ",\"role\":" + quoted(m.role);
        if (m.n_ctx > 0)
            out += ",\"n_ctx\":" + std::to_string(m.n_ctx);
        out += ",\"capabilities\":{\"tools\":" + std::string(boolean(m.tools)) +
               ",\"reasoning\":" + boolean(m.reasoning) + ",\"embedding\":" + boolean(m.embedding) +
               "}}";
    }
    out += "]}";
    return out;
}

} // namespace xllama
