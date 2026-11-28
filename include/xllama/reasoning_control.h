// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// Per-request reasoning control (#648): whether a request may turn a model's
// reasoning on or off, and the one effective decision that plain chat
// (chat_format_for) and a tool proposal (the template's enable_thinking)
// both render. The model's own behaviour stays model_is_thinking
// (chat_prompt.h); a request overrides it only where the template has both
// paths. The LAN API reads OpenAI `reasoning_effort` and
// `chat_template_kwargs.enable_thinking`, and Ollama `think` maps onto them
// (api_ollama.h). WinRT-free: the GUI can call the same decision.
#pragma once

#include <string>
#include <string_view>

namespace xllama {

// True when a request can switch this model's reasoning both ways: a model
// that does not reason by default (model_is_thinking false) whose template
// has a thinking path and a no-think path that chat_format_for renders —
// Qwen3 and Qwen3.5 hybrids (not Qwen3 Coder or Instruct ids, which have no
// thinking path), MiniCPM5, SmolLM3 and the catalogue Gemma 4 E2B IT checkpoint.
// False for a thinking-only model
// (LFM2.5 Thinking, lfm25-8b-a1b: no off switch), for a model without
// reasoning, and for unqualified Gemma checkpoints.
// /v1/models reports it as capabilities.reasoning_control.
bool model_supports_reasoning_control(const std::string& model_id);

// One requested switch, as the JSON parser saw it.
enum class ReasoningSwitch {
    Absent,  // not given, or null
    Off,     // reasoning_effort "none", enable_thinking false
    On,      // reasoning_effort "low" / "medium" / "high", enable_thinking true
    Invalid, // wrong type or unknown value
};

// reasoning_effort: "none" is Off; "low", "medium" and "high" are On (the
// level itself is not modelled: an On model reasons as its template does).
// Anything else is Invalid.
ReasoningSwitch reasoning_effort_switch(std::string_view value);

// The reasoning fields of a chat request body.
struct ReasoningFields {
    ReasoningSwitch effort = ReasoningSwitch::Absent;          // reasoning_effort
    ReasoningSwitch enable_thinking = ReasoningSwitch::Absent; // chat_template_kwargs.…
    bool kwargs_not_object = false; // chat_template_kwargs present, not null, not an object
    std::string kwargs_unknown_key; // first chat_template_kwargs key other than enable_thinking
};

// Reads the fields from an OpenAI-shaped chat body (the canonical body for
// the Ollama routes too). A body that is not a JSON object yields no fields:
// the chat handler reports it.
ReasoningFields read_reasoning_fields(std::string_view body);

inline constexpr const char* kReasoningUnsupportedCode = "reasoning_control_unsupported";

struct ReasoningDecision {
    bool thinking = false; // render the thinking path (valid when error is empty)
    std::string error;     // non-empty: 400 with this message
    const char* code = ""; // OpenAI error code
    const char* param = "";
};

// Effective thinking for |model_id|: the request's switch when it is
// honoured, else model_is_thinking(model_id). A switch that restates the
// model's own behaviour is always honoured; one that changes it needs
// model_supports_reasoning_control, otherwise 400 with
// kReasoningUnsupportedCode. reasoning_effort and enable_thinking given
// together must agree. Checked before compute admission.
ReasoningDecision decide_reasoning(const std::string& model_id, const ReasoningFields& fields);

} // namespace xllama
