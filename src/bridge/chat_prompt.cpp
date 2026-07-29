// Copyright (c) 2024 Gianluca Mazza
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

// Callers pass catalogue ids ("lfm25-350m"), filenames, or FULL paths (CLI -m).
// Substring-match only the last path component: directory names must not select
// a template — a repo/cache dir literally named "xllama" contains "llama" and
// silently forced the Llama-3 template onto every model under it (LFM2.5 then
// echoes <|eot_id|> as text instead of answering).
std::string model_basename(const std::string& model_id) {
    const size_t sep = model_id.find_last_of("/\\");
    return sep == std::string::npos ? model_id : model_id.substr(sep + 1);
}

} // namespace

bool model_is_qwen(const std::string& model_id) {
    return to_lower(model_basename(model_id)).find("qwen") != std::string::npos;
}

bool model_is_qwen3(const std::string& model_id) {
    // Qwen3 / Qwen3.5 thinking models. Must NOT match Qwen2.5 / Qwen2.5-Coder
    // (basename "qwen25-coder-1.5b", "Qwen2.5-Coder-…").
    const std::string b = to_lower(model_basename(model_id));
    return b.find("qwen3") != std::string::npos || b.find("qwen-3") != std::string::npos;
}

bool model_is_gemma(const std::string& model_id) {
    return to_lower(model_basename(model_id)).find("gemma") != std::string::npos;
}

bool model_is_llama(const std::string& model_id) {
    return to_lower(model_basename(model_id)).find("llama") != std::string::npos;
}

bool model_is_phi(const std::string& model_id) {
    return to_lower(model_basename(model_id)).find("phi") != std::string::npos;
}

bool model_is_thinking(const std::string& model_id) {
    // Catalogue / file ids like lfm25-1.2b-thinking, LFM2.5-1.2B-Thinking-….
    return to_lower(model_basename(model_id)).find("thinking") != std::string::npos;
}

std::string qwen_no_think_gen_suffix(const std::string& model_id) {
    // Only Qwen3.x uses the enable_thinking=false empty-<think> prefill.
    // Applying it to Qwen2.5-Coder injects alien special tokens into ChatML.
    // Thinking models must NOT get this suffix either (they produce real CoT).
    if (!model_is_qwen3(model_id) || model_is_thinking(model_id))
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

std::string strip_thinking_blocks(std::string text) {
    constexpr char kOpen[] = "<think>";
    constexpr char kClose[] = "</think>";
    // Unbalanced closer first: a model whose template opens the reasoning for it
    // (or a stop sequence that ate the opener) streams "reasoning…</think>answer"
    // with no <think> at all. Without this the whole chain of thought AND the raw
    // tag reach the screen and the saved history.
    {
        const size_t first_close = text.find(kClose);
        if (first_close != std::string::npos) {
            const size_t first_open = text.find(kOpen);
            if (first_open == std::string::npos || first_open > first_close)
                text.erase(0, first_close + sizeof(kClose) - 1);
        }
    }
    // Complete blocks (may appear more than once).
    for (;;) {
        const size_t o = text.find(kOpen);
        if (o == std::string::npos)
            break;
        const size_t c = text.find(kClose, o + sizeof(kOpen) - 1);
        if (c == std::string::npos) {
            // Truncated mid-thought: drop the open and everything after.
            text.erase(o);
            break;
        }
        text.erase(o, (c + sizeof(kClose) - 1) - o);
    }
    ltrim_inplace(text);
    // Trailing whitespace after the answer.
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back())))
        text.pop_back();
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
    // Gemma / Llama-3 / Phi-3 before the ChatML default. Substrings do not
    // collide with smollm2 / lfm2 / qwen ids. BOS is added by the tokenizer
    // (add_bos) for Gemma and Llama-3 GGUF — not emitted in the template string.
    if (model_is_gemma(model_id)) {
        f.kind = ChatFormatKind::Gemma;
        f.turn_open = "<start_of_turn>";
        f.turn_close = "<end_of_turn>\n";
        f.role_sep = "\n";
        f.user_tag = "user";
        f.assistant_tag = "model";
        f.system_tag = "";
        f.system_sep = "\n\n";
        f.system_style = SystemStyle::MergeIntoFirstUser;
        f.stop_sequences = {"<end_of_turn>"};
        f.gen_suffix = {};
    } else if (model_is_llama(model_id)) {
        // Meta Llama-3 / 3.1 / 3.2 instruct template.
        f.kind = ChatFormatKind::Llama3;
        f.turn_open = "<|start_header_id|>";
        f.turn_close = "<|eot_id|>";
        f.role_sep = "<|end_header_id|>\n\n";
        f.user_tag = "user";
        f.assistant_tag = "assistant";
        f.system_tag = "system";
        f.system_sep = "";
        f.system_style = SystemStyle::DedicatedTurn;
        f.stop_sequences = {"<|eot_id|>"};
        f.gen_suffix = {};
    } else if (model_is_phi(model_id)) {
        // Microsoft Phi-3 / 3.5 instruct: <|role|>\n content <|end|>\n
        f.kind = ChatFormatKind::Phi3;
        f.turn_open = "<|";
        f.turn_close = "<|end|>\n";
        f.role_sep = "|>\n";
        f.user_tag = "user";
        f.assistant_tag = "assistant";
        f.system_tag = "system";
        f.system_sep = "";
        f.system_style = SystemStyle::DedicatedTurn;
        f.stop_sequences = {"<|end|>"};
        f.gen_suffix = {};
    } else {
        f.kind = ChatFormatKind::ChatML;
        f.turn_open = "<|im_start|>";
        f.turn_close = "<|im_end|>\n";
        f.role_sep = "\n";
        f.user_tag = "user";
        f.assistant_tag = "assistant";
        f.system_tag = "system";
        f.system_sep = "";
        f.system_style = SystemStyle::DedicatedTurn;
        f.stop_sequences = {"<|im_end|>"};
        f.gen_suffix = qwen_no_think_gen_suffix(model_id);
        f.strip_thinking_content = model_is_thinking(model_id);
    }
    return f;
}

std::string ChatFormat::render_system_prefix(const std::string& system) const {
    // Must stay byte-identical to what render_prompt emits before the first
    // user turn — a context shift pins exactly this many tokens (#169).
    if (system_style == SystemStyle::DedicatedTurn)
        return turn_open + system_tag + role_sep + system + turn_close;
    return {};
}

std::string ChatFormat::render_prompt(const std::string& system,
                                      const std::vector<ChatTurn>& history,
                                      const std::string& final_user) const {
    // System: dedicated turn (ChatML / Llama-3, emitted unconditionally) or
    // merged into the first user turn (Gemma).
    std::string p = render_system_prefix(system);

    bool sys_pending = (system_style == SystemStyle::MergeIntoFirstUser) && !system.empty();
    auto user_content = [&](const std::string& u) -> std::string {
        if (sys_pending) {
            sys_pending = false;
            return system + system_sep + u;
        }
        return u;
    };

    for (const ChatTurn& t : history) {
        p += turn_open + user_tag + role_sep + user_content(t.user) + turn_close;
        // History assistant headers never carry gen_suffix (suffix only on the
        // trailing header below), matching the legacy hand-built prompt.
        p += turn_open + assistant_tag + role_sep + t.assistant + turn_close;
    }

    p += turn_open + user_tag + role_sep + user_content(final_user) + turn_close;
    p += turn_open + assistant_tag + role_sep + gen_suffix;
    return p;
}

std::string ChatFormat::render_delta(const std::string& user, bool prev_ended_with_stop) const {
    // The reused KV already holds the previous assistant tokens. If generation
    // stopped on the stop token, that token is in the KV — emit only any trailing
    // glue of turn_close beyond the stop (ChatML/Gemma: "\n"; Llama-3: empty).
    // Otherwise close the turn with the full turn_close.
    std::string d;
    if (prev_ended_with_stop) {
        if (!stop_sequences.empty() && !stop_sequences.front().empty() &&
            turn_close.compare(0, stop_sequences.front().size(), stop_sequences.front()) == 0) {
            d = turn_close.substr(stop_sequences.front().size());
        } else {
            d = "\n"; // defensive fallback for misconfigured formats
        }
    } else {
        d = turn_close;
    }
    d += turn_open + user_tag + role_sep + user + turn_close;
    d += turn_open + assistant_tag + role_sep + gen_suffix;
    return d;
}

std::string ChatFormat::postprocess_output(std::string text) const {
    if (strip_thinking_content)
        text = strip_thinking_blocks(std::move(text));
    return strip_empty_thinking_tags(std::move(text));
}

} // namespace xllama