// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
#include "xllama/api_discovery.h"

#include "xllama/api_sampling.h"
#include "xllama/json_utils.h"
#include "xllama/llama_patches.h"
#include "xllama/withdrawn_models.h"

#include <charconv>

#ifdef XLLAMA_USE_LLAMA
    #include "build-info.h"
#endif

namespace xllama {

namespace {

std::string quoted(const std::string& s) {
    return "\"" + json_escape(s) + "\"";
}

// The shortest text that reads back as |f|: 0.9f is written 0.9.
std::string number(float f) {
    char buf[32];
    const auto res = std::to_chars(buf, buf + sizeof(buf), f);
    return std::string(buf, res.ptr);
}

std::string sampling_json(const ResolvedSampling& s) {
    auto source = [](SamplingSource from) { return quoted(sampling_source_name(from)); };
    return "{\"temperature\":" + number(s.temperature) + ",\"top_p\":" + number(s.top_p) +
           ",\"top_k\":" + std::to_string(s.top_k) +
           ",\"repetition_penalty\":" + number(s.repetition_penalty) +
           ",\"source\":{\"temperature\":" + source(s.temperature_source) +
           ",\"top_p\":" + source(s.top_p_source) + ",\"top_k\":" + source(s.top_k_source) +
           ",\"repetition_penalty\":" + source(s.repetition_penalty_source) + "}}";
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
    caps.ollama_chat = true;       // /api/chat and /api/generate serve GGUF models
    caps.ollama_keep_alive = true; // session_residency.h: the console app wins
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
    caps.model_sampling = true;    // catalogue "sampling" applies to omitted fields
    caps.chat_context = true;      // options.num_ctx reduces a chat load
    caps.reasoning_control = true; // #648: per model, capabilities.reasoning_control
    caps.catalog = true;           // #658: the trusted catalogue, installed or not
    return caps;
}

std::string build_llama_cpp() {
#ifdef XLLAMA_USE_LLAMA
    return llama_commit();
#else
    return {};
#endif
}

std::string health_json(const std::string& version, const ApiCapabilities& caps,
                        const std::string& llama_cpp) {
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
    out += ",\"model_sampling\":" + std::string(boolean(caps.model_sampling));
    out += ",\"ollama_chat\":" + std::string(boolean(caps.ollama_chat));
    out += ",\"ollama_keep_alive\":" + std::string(boolean(caps.ollama_keep_alive));
    out += ",\"chat_context\":" + std::string(boolean(caps.chat_context));
    out += ",\"reasoning_control\":" + std::string(boolean(caps.reasoning_control));
    out += ",\"catalog\":" + std::string(boolean(caps.catalog));
    // The sampling fields a chat request may set (api_sampling.h, #609); a
    // client sends only these instead of assuming them per version.
    out += "},\"sampling\":[";
    for (std::size_t i = 0; i < kApiSamplingFields.size(); ++i)
        out += std::string(i ? "," : "") + quoted(kApiSamplingFields[i].key);
    out += "]";
    out += ",\"response_formats\":[" + quoted(kTextResponseFormat) + "]";
    if (!llama_cpp.empty()) {
        out += ",\"runtime\":{\"llama_cpp\":" + quoted(llama_cpp) + ",\"patches\":[";
        for (std::size_t i = 0; i < kLlamaPatches.size(); ++i)
            out += std::string(i ? "," : "") + quoted(kLlamaPatches[i]);
        out += "]}";
    }
    out += "}";
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
        if (m.has_sampling)
            out += ",\"sampling\":" + sampling_json(m.sampling);

        out += ",\"capabilities\":{\"tools\":" + std::string(boolean(m.tools)) +
               ",\"reasoning\":" + boolean(m.reasoning) +
               ",\"reasoning_control\":" + boolean(m.reasoning_control) +
               ",\"reasoning_budget\":false,\"embedding\":" + boolean(m.embedding) +
               ",\"ollama_chat\":" + boolean(m.ollama_chat) + "}}";
    }
    out += "]}";
    return out;
}

bool catalog_lists_entry(const std::string& id, const std::string& kind) {
    return !id.empty() && kind != "diffusion" && !model_id_is_withdrawn(id);
}

std::string catalog_json(const std::vector<CatalogEntryInfo>& entries) {
    std::string out = "{\"object\":\"list\",\"data\":[";
    for (std::size_t i = 0; i < entries.size(); ++i) {
        const CatalogEntryInfo& e = entries[i];
        if (i)
            out += ',';
        out += "{\"id\":" + quoted(e.id) + ",\"display\":" + quoted(e.display) +
               ",\"kind\":" + quoted(e.kind) + ",\"role\":" + quoted(e.role) +
               ",\"n_ctx\":" + std::to_string(e.n_ctx) +
               ",\"size_bytes\":" + std::to_string(e.size_bytes) +
               ",\"installed\":" + boolean(e.installed) + ",\"pullable\":" + boolean(e.pullable);
        if (e.has_sampling)
            out += ",\"sampling\":" + sampling_json(e.sampling);
        out += ",\"capabilities\":{\"reasoning\":" + std::string(boolean(e.reasoning)) +
               ",\"embedding\":" + boolean(e.embedding) + "}}";
    }
    out += "]}";
    return out;
}

} // namespace xllama
