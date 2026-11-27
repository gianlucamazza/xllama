// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// Native Ollama text generation (POST /api/chat, POST /api/generate) as a
// translation into the OpenAI chat body the LAN handler already serves. The
// request is checked and normalized here, before compute admission, so the
// GGUF template, budget, sampling and stops stay on one path for both
// protocols. uwp/api-server.cpp keeps the NDJSON framing.
#pragma once

#include <chrono>
#include <optional>
#include <string>
#include <string_view>

namespace xllama {

enum class OllamaRoute { Chat, Generate };

// Load: an empty `messages` (chat) or omitted/empty `prompt` (generate) loads
// the model and returns {"done_reason":"load"} without generating. Unload: the
// same empty request with keep_alive 0 ({"done_reason":"unload"}).
enum class OllamaAction { Generate, Load, Unload };

struct OllamaRequest {
    // The canonical OpenAI-shaped chat body (model, messages, tools, sampling,
    // stop, max_completion_tokens, options.num_ctx) the shared chat handler
    // consumes.
    std::string body;
    std::string error; // non-empty: 400 with this text
    OllamaAction action = OllamaAction::Generate;
    // nullopt when omitted or null: no expiry (session_residency.h). Negative
    // never expires; 0 unloads at the end of the request.
    std::optional<std::chrono::milliseconds> keep_alive;
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

// keep_alive is a number of seconds or a Go-style duration string ("5m",
// "1h30m", "750ms", "-1s"); anything else is 400. Any negative value becomes
// -1 ms. options.num_ctx passes through for the handler to check against the
// catalogue context (validate_context_override, api_policy.h).
OllamaRequest parse_ollama_request(std::string_view body, OllamaRoute route);

} // namespace xllama
