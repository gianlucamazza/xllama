// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
// Preference capture — LocalState training samples for host retrain (hybrid loop).
// Pure helpers (no WinRT) so host unit tests cover formatting.
#pragma once

#include <string>
#include <vector>

namespace xllama {

// like | dislike | correction | implicit
bool preference_label_valid(const std::string& label);

// One JSONL line (no trailing newline). messages are role/content pairs already UTF-8.
// Returns empty string on invalid label or empty messages.
std::string format_preference_sample_jsonl(
    const std::string& label, const std::vector<std::pair<std::string, std::string>>& messages,
    const std::string& preferred_assistant = {}, const std::string& ts_iso = {});

// Append one line + newline to path (create parent dirs not handled here).
// Returns false on I/O error.
bool append_preference_sample_file(const std::string& path, const std::string& jsonl_line);

// Default relative path under LocalState.
inline const char* kPreferenceSamplesRelPath = "training/samples.jsonl";

} // namespace xllama
