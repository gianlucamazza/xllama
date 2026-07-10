// Copyright (c) 2024 Venere Labs
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

namespace xllama {

// ---------------------------------------------------------------------------
// Path resolution
//
// Linux: returns the input unchanged (assumed absolute).
// UWP: resolves relative to ApplicationData::Current().LocalFolder().
// ---------------------------------------------------------------------------

// Resolve a model filename: UWP -> LocalFolder\models\<filename>
std::string resolve_model_path(const std::string& filename);

// Resolve a generic filename: UWP -> LocalFolder\<filename>
std::string resolve_local_path(const std::string& filename);

// Returns true if the model identifier (bare catalogue name or path) refers to
// a GGUF model for the llama.cpp backend.
// - Fast path: input ends with ".gguf".
// - Otherwise resolves the path and checks for a .gguf file or a directory
//   containing at least one *.gguf. This makes bare-name catalogue entries
//   (e.g. "qwen35-0.8b") work correctly with Backend::Auto in unified builds.
bool model_uses_llama_backend(const std::string& model_id);

// llama_model_load_from_file needs a FILE path, but catalogue GGUF entries
// resolve to a directory (LocalState\models\<name>\<file>.gguf). If path is a
// directory, return the first *.gguf inside it; empty string if it contains
// none. If path is not a directory, return it unchanged.
std::string first_gguf_in_dir(const std::string& path);

} // namespace xllama
