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

} // namespace xllama
