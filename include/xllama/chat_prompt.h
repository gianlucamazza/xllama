// Copyright (c) 2024 Venere Labs
// SPDX-License-Identifier: MIT
// Chat prompt / output helpers shared by UWP UI and unit tests.
#pragma once

#include <string>
#include <vector>

namespace xllama {

// True for catalogue ids or filenames that refer to a Qwen GGUF chat model.
bool model_is_qwen(const std::string& model_id);

// True for catalogue ids or filenames that refer to a Gemma chat model
// (substring "gemma", case-insensitive).
bool model_is_gemma(const std::string& model_id);

// True for Llama / Llama-3.x catalogue ids or filenames (substring "llama",
// case-insensitive). Used to select the Llama-3 instruct header template.
bool model_is_llama(const std::string& model_id);

// True for Phi / Phi-3.x catalogue ids or filenames (substring "phi",
// case-insensitive). Selects the Phi-3 instruct template (<|user|>…<|end|>).
bool model_is_phi(const std::string& model_id);

// Qwen3.x no-think generation prefill (matches Qwen3.5 Jinja with enable_thinking=false).
// Empty when the model is not Qwen.
std::string qwen_no_think_gen_suffix(const std::string& model_id);

// Remove leading empty </think> blocks (whitespace-only inside tags).
std::string strip_empty_thinking_tags(std::string text);

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

    // Output post-processing before display/persist (strips empty <think> blocks;
    // no-op for Gemma).
    std::string postprocess_output(std::string text) const;
};

// Select the chat format for a catalogue id / filename (as model_is_qwen does today).
ChatFormat chat_format_for(const std::string& model_id);

} // namespace xllama
