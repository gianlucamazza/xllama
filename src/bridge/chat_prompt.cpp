// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include "xllama/chat_prompt.h"

#include "xllama/thinking_split.h"

#include <algorithm>
#include <cctype>
#include <utility>

namespace xllama {
namespace {

constexpr char kSmolLM3FrozenDate[] = "15 June 2025";
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

bool model_is_qwen35(const std::string& model_id) {
    const std::string b = to_lower(model_basename(model_id));
    for (const char* tag : {"qwen35", "qwen3.5", "qwen3_5", "qwen-3.5"})
        if (b.find(tag) != std::string::npos)
            return true;
    return false;
}

bool model_is_gemma(const std::string& model_id) {
    return to_lower(model_basename(model_id)).find("gemma") != std::string::npos;
}

bool model_is_gemma4(const std::string& model_id) {
    const std::string b = to_lower(model_basename(model_id));
    return b.find("gemma4") != std::string::npos || b.find("gemma-4") != std::string::npos ||
           b.find("gemma_4") != std::string::npos;
}

bool model_is_gemma4_e2b_it(const std::string& model_id) {
    const std::string b = to_lower(model_basename(model_id));
    return b == "gemma4-e2b" || b == "gemma-4-e2b-it-q3_k_s.gguf";
}

bool model_is_llama(const std::string& model_id) {
    return to_lower(model_basename(model_id)).find("llama") != std::string::npos;
}

bool model_is_phi(const std::string& model_id) {
    return to_lower(model_basename(model_id)).find("phi") != std::string::npos;
}

bool model_is_granite(const std::string& model_id) {
    return to_lower(model_basename(model_id)).find("granite") != std::string::npos;
}

bool model_is_olmoe(const std::string& model_id) {
    const std::string b = to_lower(model_basename(model_id));
    return b.find("olmoe") != std::string::npos || b.find("olmo-e") != std::string::npos ||
           b.find("olmo_e") != std::string::npos;
}

bool model_is_smollm3(const std::string& model_id) {
    const std::string b = to_lower(model_basename(model_id));
    return b.find("smollm3") != std::string::npos || b.find("smol-lm3") != std::string::npos ||
           b.find("smol_lm3") != std::string::npos;
}

bool model_is_thinking(const std::string& model_id) {
    const std::string id = to_lower(model_basename(model_id));
    // Catalogue / file ids like lfm25-1.2b-thinking, LFM2.5-1.2B-Thinking-….
    if (id.find("thinking") != std::string::npos)
        return true;
    // LFM2.5-8B-A1B reasons unconditionally without saying so in its name: the
    // shipped template emits <think> on every turn, and its `preserve_thinking`
    // flag governs only whether past turns' CoT is replayed into the history,
    // not whether the model produces it. Missing this would leak raw CoT into
    // the UI and — worse — leave KV snapshots enabled on a model that must be
    // excluded from them, because the saved history is stripped while the KV
    // holds the full CoT, so the prefix diff can never match (MainPage.cpp).
    return id.find("a1b") != std::string::npos;
}

bool model_is_minicpm5(const std::string& model_id) {
    // Deliberately NOT folded into qwen_no_think_gen_suffix: that function is
    // named for the family whose template it reproduces, and MiniCPM5 needs the
    // same prefill for its own reason. general.architecture is `llama` here, so
    // the id must never be spelled with "llama" — that would force the Llama-3
    // renderer and echo <|eot_id|> as text.
    return to_lower(model_basename(model_id)).find("minicpm5") != std::string::npos;
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
    return split_thinking(text).content;
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
    return chat_format_for(model_id, model_is_thinking(model_id));
}

ChatFormat chat_format_for(const std::string& model_id, bool thinking) {
    ChatFormat f;
    // Gemma / Llama-3 / Phi-3 / Granite before the ChatML default. Substrings do
    // not collide with smollm2 / lfm2 / qwen ids. Granite file names often carry
    // a runtime tag ("-ollama", "-llamacpp"), so "llama"/"phi" yield to it. BOS is added by the
    // tokenizer (add_bos) for Gemma and Llama-3 GGUF — not emitted in the template string.
    if (model_is_gemma4(model_id)) {
        // Gemma 4 canonical template: new turn tokens and a dedicated system
        // turn. The E2B template signals thinking in the system turn.
        f.kind = ChatFormatKind::Gemma;
        f.turn_open = "<|turn>";
        f.turn_close = "<turn|>\n";
        f.role_sep = "\n";
        f.user_tag = "user";
        f.assistant_tag = "model";
        f.system_tag = "system";
        f.system_sep = "";
        f.system_style = SystemStyle::DedicatedTurn;
        f.stop_sequences = {"<turn|>"};
        if (model_is_gemma4_e2b_it(model_id)) {
            f.thinking_markup = ThinkingMarkup::GemmaChannel;
            f.strip_thinking_content = thinking;
        }
        f.gen_suffix = {};
    } else if (model_is_gemma(model_id)) {
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
    } else if (model_is_llama(model_id) && !model_is_granite(model_id) &&
               !model_is_olmoe(model_id)) {
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
    } else if (model_is_phi(model_id) && !model_is_granite(model_id) && !model_is_olmoe(model_id)) {
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
    } else if (model_is_granite(model_id)) {
        // IBM Granite 4.x instruct (chat_template.jinja): dedicated system
        // turn, no extra space after <|end_of_role|>, generation header has
        // no closer. Same tokens as Granite 3.x instruct. Built once so #521
        // (dense 4.2-3B) reuses this branch.
        f.kind = ChatFormatKind::Granite;
        f.turn_open = "<|start_of_role|>";
        f.turn_close = "<|end_of_text|>\n";
        f.role_sep = "<|end_of_role|>";
        f.user_tag = "user";
        f.assistant_tag = "assistant";
        f.system_tag = "system";
        f.system_sep = "";
        f.system_style = SystemStyle::DedicatedTurn;
        f.stop_sequences = {"<|end_of_text|>"};
        f.gen_suffix = {};
    } else if (model_is_olmoe(model_id)) {
        f.kind = ChatFormatKind::Olmoe;
        f.turn_open = "<|";
        f.turn_close = "\n";
        f.assistant_turn_close = "<|endoftext|>\n";
        f.role_sep = "|>\n";
        f.user_tag = "user";
        f.assistant_tag = "assistant";
        f.system_tag = "system";
        f.system_sep = "";
        f.system_style = SystemStyle::DedicatedTurn;
        f.bos = "<|endoftext|>";
        f.stop_sequences = {"<|endoftext|>"};
        f.gen_suffix = {};
    } else if (model_is_smollm3(model_id)) {
        f.kind = ChatFormatKind::SmolLM3;
        f.turn_open = "<|im_start|>";
        f.turn_close = "<|im_end|>\n";
        f.role_sep = "\n";
        f.user_tag = "user";
        f.assistant_tag = "assistant";
        f.system_tag = "system";
        f.system_sep = "";
        f.system_style = SystemStyle::DedicatedTurn;
        f.stop_sequences = {"<|im_end|>"};
        // The template's /no_think mode prefills an empty think block in the
        // generation prompt and in every past assistant turn; /think has
        // neither (render_system_prefix names the mode).
        f.gen_suffix = thinking ? "" : "<think>\n\n</think>\n";
        f.gen_suffix_in_history = !thinking;
        f.strip_thinking_content = thinking;
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
        // |thinking| is the decision the tool path renders too (#655, #648):
        // the no-think prefill only when it is off. With it on, the Qwen3.5
        // and MiniCPM5 templates open <think> in the prompt; Qwen3 leaves it
        // to the model.
        const bool minicpm5 = model_is_minicpm5(model_id);
        if (!thinking && (minicpm5 || model_is_qwen3(model_id))) {
            f.gen_suffix = empty_think_block() + "\n\n";
        } else if (thinking && (minicpm5 || model_is_qwen35(model_id))) {
            f.gen_suffix = std::string(kThinkOpen) + "\n";
            f.reasoning_opened = true;
        }
        f.bos = minicpm5 ? "<s>" : "";
        f.strip_thinking_content = thinking;
    }
    return f;
}

std::string ChatFormat::render_system_prefix(const std::string& system) const {
    // Must stay byte-identical to what render_prompt emits before the first
    // user turn — a context shift pins exactly this many tokens (#169).
    // bos belongs here rather than in render_prompt: this is the one function
    // both the prompt and the #169 pinned head go through, so a template BOS
    // cannot be counted by one and missed by the other.
    if (kind == ChatFormatKind::SmolLM3) {
        std::string instructions = system;
        for (const std::string flag : {"/no_think", "/think", "/system_override"}) {
            size_t pos = 0;
            while ((pos = instructions.find(flag, pos)) != std::string::npos)
                instructions.erase(pos, flag.size());
        }
        while (!instructions.empty() &&
               std::isspace(static_cast<unsigned char>(instructions.back())))
            instructions.pop_back();
        if (instructions.empty())
            instructions = "You are a helpful AI assistant named SmolLM, trained by Hugging Face.";
        return turn_open + system_tag + role_sep +
               "## Metadata\n\n"
               "Knowledge Cutoff Date: June 2025\nToday Date: " +
               kSmolLM3FrozenDate +
               "\nReasoning Mode: " + (strip_thinking_content ? "/think" : "/no_think") +
               "\n\n## Custom Instructions\n\n" + instructions + "\n\n" + turn_close;
    }
    if (thinking_markup == ThinkingMarkup::GemmaChannel && strip_thinking_content)
        return bos + turn_open + system_tag + role_sep + "<|think|>\n" + system + turn_close;
    if (system_style == SystemStyle::DedicatedTurn)
        return bos + turn_open + system_tag + role_sep + system + turn_close;
    return bos;
}

std::string ChatFormat::render_prompt(const std::string& system,
                                      const std::vector<ChatTurn>& history,
                                      const std::string& final_user) const {
    // System: dedicated turn (ChatML / Llama-3 / Gemma 4 / Granite, emitted
    // unconditionally — an empty `system` still produces an empty system turn;
    // IBM's Granite jinja would insert a default message instead. CLI, GUI, and
    // the LAN API always pass kDefaultSystemPrompt or the coding default today)
    // or merged into the first user turn (Gemma 3).
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
        // History assistant turns use assistant_close() (OLMoE ends those
        // turns with the end-of-text token). They also carry gen_suffix when
        // gen_suffix_in_history is set (SmolLM3 puts its empty-think block
        // there, and the KV cache already holds it).
        p += turn_open + assistant_tag + role_sep + (gen_suffix_in_history ? gen_suffix : "") +
             t.assistant + assistant_close();
    }

    p += turn_open + user_tag + role_sep + user_content(final_user) + turn_close;
    p += turn_open + assistant_tag + role_sep + gen_suffix;
    return p;
}

std::string ChatFormat::render_delta(const std::string& user, bool prev_ended_with_stop) const {
    // The reused KV already holds the previous assistant tokens. If generation
    // stopped on the stop token, that token is in the KV — emit only any trailing
    // glue of assistant_close() beyond the stop (ChatML/Gemma: "\n"; Llama-3: empty).
    // Otherwise close the turn with the full assistant_close().
    std::string d;
    if (prev_ended_with_stop) {
        if (!stop_sequences.empty() && !stop_sequences.front().empty() &&
            assistant_close().compare(0, stop_sequences.front().size(), stop_sequences.front()) ==
                0) {
            d = assistant_close().substr(stop_sequences.front().size());
        } else {
            d = "\n"; // defensive fallback for misconfigured formats
        }
    } else {
        d = assistant_close();
    }
    d += turn_open + user_tag + role_sep + user + turn_close;
    d += turn_open + assistant_tag + role_sep + gen_suffix;
    return d;
}

std::string ChatFormat::postprocess_output(std::string text) const {
    return split_output(std::move(text)).content;
}

ThinkingSplit ChatFormat::split_output(std::string text) const {
    ThinkingSplit out;
    if (strip_thinking_content) {
        // After an opened block the reply starts inside it, and one cut off
        // mid-thought has no closer either: split it as the model saw it.
        text.insert(0, reasoning_prefill());
        out = split_thinking(text, thinking_markup);
        text = std::move(out.content);
    }
    out.content = strip_empty_thinking_tags(std::move(text));
    return out;
}

} // namespace xllama