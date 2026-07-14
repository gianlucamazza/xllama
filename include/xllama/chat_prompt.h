// Copyright (c) 2024 Venere Labs
// SPDX-License-Identifier: MIT
// Chat prompt / output helpers shared by UWP UI and unit tests.
#pragma once

#include <string>

namespace xllama {

// True for catalogue ids or filenames that refer to a Qwen GGUF chat model.
bool model_is_qwen(const std::string& model_id);

// Qwen3.x no-think generation prefill (matches Qwen3.5 Jinja with enable_thinking=false).
// Empty when the model is not Qwen.
std::string qwen_no_think_gen_suffix(const std::string& model_id);

// Remove leading empty </think> blocks (whitespace-only inside tags).
std::string strip_empty_thinking_tags(std::string text);

} // namespace xllama