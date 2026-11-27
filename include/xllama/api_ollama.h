// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// Native Ollama text generation (POST /api/chat, POST /api/generate) as a
// translation into the OpenAI chat body the LAN handler already serves. The
// request is checked and normalized here, before compute admission, so the
// GGUF template, budget, sampling and stops stay on one path for both
// protocols. uwp/api-server.cpp keeps the NDJSON framing.
#pragma once

#include <string>
#include <string_view>

namespace xllama {

enum class OllamaRoute { Chat, Generate };

// Load: an empty `messages` (chat) or omitted/empty `prompt` (generate) loads
// the model and returns {"done_reason":"load"} without generating.
enum class OllamaAction { Generate, Load };

struct OllamaRequest {
    // The canonical OpenAI-shaped chat body (model, messages, tools, sampling,
    // stop, max_completion_tokens) the shared chat handler consumes.
    std::string body;
    std::string error; // non-empty: 400 with this text
    OllamaAction action = OllamaAction::Generate;
    bool stream = true; // Ollama streams NDJSON unless "stream": false
    // A transcript that replays tool calls and results without offering tools
    // again: it still renders through the GGUF template (parse_tool_request).
    bool replay_without_tools = false;
};

// A model id the native routes accept: catalogue-style ids only, no path.
bool ollama_model_id(std::string_view id);

// The per-model `capabilities.ollama_chat` of /v1/models and the check
// /api/chat and /api/generate make: a servable GGUF text model (not an
// embedding model) whose id the native routes accept. ONNX models stay on
// /v1/chat/completions.
bool ollama_chat_model(std::string_view id, bool gguf, bool embedding);

// `think` is accepted when absent, null, or equal to the model's own behaviour
// (model_is_thinking in chat_prompt.h). A value that would change it is
// refused with this message until the override exists.
inline constexpr const char* kOllamaThinkUnsupported =
    "think override is not supported for this model yet";

OllamaRequest parse_ollama_request(std::string_view body, OllamaRoute route);

} // namespace xllama
