// Copyright (c) 2024 Venere Labs
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

namespace xllama {

// ---------------------------------------------------------------------------
// UTF-8 <-> wide string conversions (Windows/UWP)
// ---------------------------------------------------------------------------

#ifdef _WIN32
std::wstring utf8_to_wstring(const std::string& s);
std::string wstring_to_utf8(const std::wstring& w);
#endif

} // namespace xllama
