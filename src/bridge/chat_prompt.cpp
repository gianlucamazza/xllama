// Copyright (c) 2024 Venere Labs
// SPDX-License-Identifier: MIT

#include "xllama/chat_prompt.h"

#include <algorithm>
#include <cctype>
#include <utility>

namespace xllama {
namespace {

constexpr char kThinkOpen[] = "<think>";
constexpr char kThinkClose[] = "</think>";

std::string empty_think_block() {
    return std::string(kThinkOpen) + "\n\n" + kThinkClose;
}

std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

void ltrim_inplace(std::string& s) {
    size_t i = 0;
    while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i])))
        ++i;
    if (i > 0)
        s.erase(0, i);
}

bool starts_with(const std::string& s, const std::string& prefix) {
    return s.size() >= prefix.size() && s.compare(0, prefix.size(), prefix) == 0;
}

} // namespace

bool model_is_qwen(const std::string& model_id) {
    return to_lower(model_id).find("qwen") != std::string::npos;
}

bool model_is_gemma(const std::string& model_id) {
    return to_lower(model_id).find("gemma") != std::string::npos;
}

std::string qwen_no_think_gen_suffix(const std::string& model_id) {
    if (!model_is_qwen(model_id))
        return {};
    return empty_think_block() + "\n\n";
}

std::string strip_empty_thinking_tags(std::string text) {
    const std::string kEmpty = empty_think_block();
    ltrim_inplace(text);
    while (starts_with(text, kEmpty)) {
        text.erase(0, kEmpty.size());
        ltrim_inplace(text);
    }
    return text;
}

bool apply_stop_sequences(std::string& output, const std::vector<std::string>& stops) {
    for (const std::string& stop : stops) {
        if (!stop.empty() && output.size() >= stop.size() &&
            output.compare(output.size() - stop.size(), stop.size(), stop) == 0) {
            output.erase(output.size() - stop.size());
            return true;
        }
    }
    return false;
}

ChatFormat chat_format_for(const std::string& model_id) {
    ChatFormat f;
    // Gemma is checked before the ChatML default; the Qwen no-think suffix is
    // resolved inside the ChatML branch. "gemma" does not collide with
    // smollm2/lfm2/qwen ids.
    if (model_is_gemma(model_id)) {
        f.kind = ChatFormatKind::Gemma;
        f.turn_open = "<start_of_turn>";
        f.turn_close = "<end_of_turn>\n";
        f.user_tag = "user";
        f.assistant_tag = "model";
        f.system_tag = "";
        f.system_sep = "\n\n";
        f.system_style = SystemStyle::MergeIntoFirstUser;
        f.stop_sequences = {"<end_of_turn>"};
        f.gen_suffix = {}; // Gemma has no think prefill; <bos> is added by add_bos.
    } else {
        f.kind = ChatFormatKind::ChatML;
        f.turn_open = "<|im_start|>";
        f.turn_close = "<|im_end|>\n";
        f.user_tag = "user";
        f.assistant_tag = "assistant";
        f.system_tag = "system";
        f.system_sep = "";
        f.system_style = SystemStyle::DedicatedTurn;
        f.stop_sequences = {"<|im_end|>"};
        f.gen_suffix = qwen_no_think_gen_suffix(model_id);
    }
    return f;
}

std::string ChatFormat::render_prompt(const std::string& system,
                                      const std::vector<ChatTurn>& history,
                                      const std::string& final_user) const {
    std::string p;

    // System: dedicated turn (ChatML, emitted unconditionally) or merged into
    // the first user turn (Gemma).
    if (system_style == SystemStyle::DedicatedTurn)
        p += turn_open + system_tag + "\n" + system + turn_close;

    bool sys_pending = (system_style == SystemStyle::MergeIntoFirstUser) && !system.empty();
    auto user_content = [&](const std::string& u) -> std::string {
        if (sys_pending) {
            sys_pending = false;
            return system + system_sep + u;
        }
        return u;
    };

    for (const ChatTurn& t : history) {
        p += turn_open + user_tag + "\n" + user_content(t.user) + turn_close;
        // History assistant headers never carry gen_suffix (suffix only on the
        // trailing header below), matching the legacy hand-built prompt.
        p += turn_open + assistant_tag + "\n" + t.assistant + turn_close;
    }

    p += turn_open + user_tag + "\n" + user_content(final_user) + turn_close;
    p += turn_open + assistant_tag + "\n" + gen_suffix;
    return p;
}

std::string ChatFormat::render_delta(const std::string& user, bool prev_ended_with_stop) const {
    // The reused KV already holds the previous assistant tokens. If generation
    // stopped on the stop token the turn is already closed (only a newline is
    // needed); otherwise close it with turn_close.
    std::string d = prev_ended_with_stop ? std::string("\n") : turn_close;
    d += turn_open + user_tag + "\n" + user + turn_close;
    d += turn_open + assistant_tag + "\n" + gen_suffix;
    return d;
}

std::string ChatFormat::postprocess_output(std::string text) const {
    return strip_empty_thinking_tags(std::move(text));
}

} // namespace xllama