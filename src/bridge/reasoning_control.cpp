// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include "xllama/reasoning_control.h"

#include "xllama/chat_prompt.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <utility>

namespace xllama {
namespace {

using Json = nlohmann::json;

constexpr const char* kEffortParam = "reasoning_effort";
constexpr const char* kKwargsParam = "chat_template_kwargs";
constexpr const char* kEnableThinkingParam = "chat_template_kwargs.enable_thinking";

std::string lower_basename(const std::string& model_id) {
    const size_t sep = model_id.find_last_of("/\\");
    std::string b = sep == std::string::npos ? model_id : model_id.substr(sep + 1);
    std::transform(b.begin(), b.end(), b.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return b;
}

ReasoningDecision refused(std::string message, const char* code, const char* param) {
    ReasoningDecision d;
    d.error = std::move(message);
    d.code = code;
    d.param = param;
    return d;
}

} // namespace

bool model_supports_reasoning_control(const std::string& model_id) {
    if (model_is_thinking(model_id))
        return false; // thinking-only: no template here has an off switch
    if (model_is_minicpm5(model_id) || model_is_smollm3(model_id))
        return true;
    if (!model_is_qwen3(model_id))
        return false;
    // Qwen3 Coder and the 2507 Instruct releases are no-think only.
    const std::string b = lower_basename(model_id);
    return b.find("coder") == std::string::npos && b.find("instruct") == std::string::npos;
}

ReasoningSwitch reasoning_effort_switch(std::string_view value) {
    if (value == "none")
        return ReasoningSwitch::Off;
    if (value == "low" || value == "medium" || value == "high")
        return ReasoningSwitch::On;
    return ReasoningSwitch::Invalid;
}

ReasoningFields read_reasoning_fields(std::string_view body) {
    ReasoningFields fields;
    const Json root = Json::parse(body, nullptr, /*allow_exceptions=*/false);
    if (!root.is_object())
        return fields;
    if (const auto it = root.find(kEffortParam); it != root.end() && !it->is_null()) {
        fields.effort = it->is_string() ? reasoning_effort_switch(it->get_ref<const std::string&>())
                                        : ReasoningSwitch::Invalid;
    }
    const auto kwargs = root.find(kKwargsParam);
    if (kwargs == root.end() || kwargs->is_null())
        return fields;
    if (!kwargs->is_object()) {
        fields.kwargs_not_object = true;
        return fields;
    }
    for (const auto& [key, value] : kwargs->items()) {
        if (key != "enable_thinking") {
            if (fields.kwargs_unknown_key.empty())
                fields.kwargs_unknown_key = key;
            continue;
        }
        if (value.is_null())
            continue;
        fields.enable_thinking = !value.is_boolean() ? ReasoningSwitch::Invalid
                                 : value.get<bool>() ? ReasoningSwitch::On
                                                     : ReasoningSwitch::Off;
    }
    return fields;
}

ReasoningDecision decide_reasoning(const std::string& model_id, const ReasoningFields& fields) {
    using Switch = ReasoningSwitch;
    if (fields.effort == Switch::Invalid)
        return refused("reasoning_effort must be \"none\", \"low\", \"medium\" or \"high\"",
                       "invalid_value", kEffortParam);
    if (fields.kwargs_not_object)
        return refused("chat_template_kwargs must be an object", "invalid_value", kKwargsParam);
    if (!fields.kwargs_unknown_key.empty())
        return refused("chat_template_kwargs supports only enable_thinking (got '" +
                           fields.kwargs_unknown_key + "')",
                       "invalid_value", kKwargsParam);
    if (fields.enable_thinking == Switch::Invalid)
        return refused("chat_template_kwargs.enable_thinking must be a boolean", "invalid_value",
                       kEnableThinkingParam);
    if (fields.effort != Switch::Absent && fields.enable_thinking != Switch::Absent &&
        fields.effort != fields.enable_thinking)
        return refused("reasoning_effort and chat_template_kwargs.enable_thinking disagree",
                       "invalid_value", kEffortParam);

    ReasoningDecision d;
    d.thinking = model_is_thinking(model_id);
    const Switch requested =
        fields.effort != Switch::Absent ? fields.effort : fields.enable_thinking;
    if (requested == Switch::Absent)
        return d;
    const bool on = requested == Switch::On;
    if (on == d.thinking)
        return d; // restates what the model does
    const char* param = fields.effort != Switch::Absent ? kEffortParam : kEnableThinkingParam;
    if (!model_supports_reasoning_control(model_id)) {
        return refused(d.thinking ? "model '" + model_id +
                                        "' always reasons: its template cannot turn reasoning off "
                                        "(capabilities.reasoning_control in /v1/models)"
                                  : "model '" + model_id +
                                        "' does not reason: its template cannot turn reasoning on "
                                        "(capabilities.reasoning_control in /v1/models)",
                       kReasoningUnsupportedCode, param);
    }
    d.thinking = on;
    return d;
}

} // namespace xllama
