// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
// Chat prompt / output helpers shared by UWP UI and unit tests.
#pragma once

#include <string>
#include <vector>

namespace xllama {

// Shared system prompts (UI default, LAN API empty-system fill, CLI --chat).
inline constexpr const char kDefaultSystemPrompt[] = "You are a helpful AI assistant.";
// Applied when catalogue role is "coding" and the user has not set a custom system.
inline constexpr const char kCodingSystemPrompt[] =
    "You are a concise coding assistant. Prefer correct, complete code and brief explanations.";

// True for catalogue ids or filenames that refer to a Qwen GGUF chat model.
bool model_is_qwen(const std::string& model_id);

// True for Qwen3.x (thinking-capable) ids — not Qwen2.5 / Qwen2.5-Coder.
// Substring "qwen3" / "qwen-3" on the basename (catalogue id qwen35-0.8b matches).
bool model_is_qwen3(const std::string& model_id);

// True for catalogue ids or filenames that refer to a Gemma chat model
// (substring "gemma", case-insensitive).
bool model_is_gemma(const std::string& model_id);

// True for Llama / Llama-3.x catalogue ids or filenames (substring "llama",
// case-insensitive). Used to select the Llama-3 instruct header template.
bool model_is_llama(const std::string& model_id);

// True for Phi / Phi-3.x catalogue ids or filenames (substring "phi",
// case-insensitive). Selects the Phi-3 instruct template (<|user|>…<|end|>).
bool model_is_phi(const std::string& model_id);

// True for models that emit chain-of-thought inside <think>…</think> before the
// user-visible answer (basename contains "thinking", e.g. lfm25-1.2b-thinking).
// These keep plain ChatML with no Qwen3 no-think prefill; postprocess strips
// think blocks for display/persist.
bool model_is_thinking(const std::string& model_id);

// Qwen3.x no-think generation prefill (matches Qwen3.5 Jinja with enable_thinking=false).
// Empty when the model is not Qwen3 (Qwen2.5-Coder must not receive <think> markers).
std::string qwen_no_think_gen_suffix(const std::string& model_id);

// Remove leading empty </think> blocks (whitespace-only inside tags).
std::string strip_empty_thinking_tags(std::string text);

// Remove complete <think>…</think> blocks (any content). If an unclosed
// <think> remains (n_predict cut mid-thought), drop from that open to EOF so
// the UI does not keep raw reasoning. Leading/trailing whitespace cleaned.
std::string strip_thinking_blocks(std::string text);

// Stop-sequence check for streamed generation. Call after each token is appended
// to `output`: if `output` now ENDS WITH any (non-empty) stop sequence, trim that
// trailing match off `output` and return true. Suffix-match (not substring) is
// the correct streaming semantics and handles multi-piece stop tokens (e.g.
// Gemma's <end_of_turn>, split across llama_token_to_piece calls). Shared by the
// llama and ORT decode loops so their stop behaviour can't diverge.
bool apply_stop_sequences(std::string& output, const std::vector<std::string>& stops);

// ---------------------------------------------------------------------------
// Per-architecture chat template
//
// Data-driven, backend-agnostic. Supported formats differ in delimiters, the
// assistant role label, how the system prompt is placed, the stop sequence and
// a per-model generation suffix. Built via chat_format_for().
// ---------------------------------------------------------------------------

enum class ChatFormatKind { ChatML, Gemma, Llama3, Phi3 };

// How the system prompt is emitted.
// - DedicatedTurn: its own turn (ChatML / Llama-3 system role).
// - MergeIntoFirstUser: prepended to the first user turn (Gemma has no system role).
enum class SystemStyle { DedicatedTurn, MergeIntoFirstUser };

// One completed exchange used when rendering a full multi-turn prompt.
struct ChatTurn {
    std::string user;
    std::string assistant;
};

struct ChatFormat {
    ChatFormatKind kind = ChatFormatKind::ChatML;

    // Delimiters / role labels.
    // Layout: turn_open + role_tag + role_sep + content + turn_close
    std::string turn_open;     // "<|im_start|>" | "<start_of_turn>" | "<|start_header_id|>"
    std::string turn_close;    // "<|im_end|>\n" | "<end_of_turn>\n" | "<|eot_id|>"
    std::string role_sep;      // "\n" | "\n" | "<|end_header_id|>\n\n"
    std::string user_tag;      // "user"
    std::string assistant_tag; // "assistant" | "model"
    std::string system_tag;    // "system"    | "" (Gemma: unused)
    std::string system_sep;    // ""          | "\n\n" (merge separator)
    SystemStyle system_style = SystemStyle::DedicatedTurn;

    std::vector<std::string> stop_sequences; // stop token(s); may be a prefix of turn_close
    std::string gen_suffix;                  // Qwen no-think prefill; else ""

    // When true, postprocess_output strips full <think>…</think> reasoning
    // (thinking models). Empty-think stripping for Qwen3 no-think always runs.
    bool strip_thinking_content = false;

    // Full multi-turn prompt ending with the assistant generation header +
    // gen_suffix. gen_suffix is appended ONLY to the final (trailing) assistant
    // header, never to completed history turns. History turns are complete
    // (user + assistant) exchanges.
    std::string render_prompt(const std::string& system, const std::vector<ChatTurn>& history,
                              const std::string& final_user) const;

    // Delta appended onto a reused KV cache. prev_ended_with_stop == the previous
    // generation stopped on the stop token (turn already closed) vs cut short by
    // the n_predict cap.
    std::string render_delta(const std::string& user, bool prev_ended_with_stop) const;

    // #169: the exact prefix render_prompt emits before the first user turn —
    // what a context shift must pin (GenerateParams::n_keep is its token
    // count). Empty for MergeIntoFirstUser formats (Gemma), whose system text
    // lives inside the first user turn and cannot be pinned separately;
    // count_tokens("") still pins the BOS.
    std::string render_system_prefix(const std::string& system) const;

    // Output post-processing before display/persist: empty Qwen3 no-think tags;
    // full think blocks when strip_thinking_content (thinking catalogue models).
    std::string postprocess_output(std::string text) const;
};

// Select the chat format for a catalogue id / filename (as model_is_qwen does today).
ChatFormat chat_format_for(const std::string& model_id);

} // namespace xllama
