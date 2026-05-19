// Copyright (c) 2024 Venere Labs
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <string>

namespace xllama {

// ---------------------------------------------------------------------------
// Platform helpers
// ---------------------------------------------------------------------------

// Number of hardware threads; falls back to 4 if detection fails.
int detect_threads() noexcept;

// Emit a log line. On UWP: OutputDebugStringA; otherwise: stderr.
void log_output(const char* msg) noexcept;
void log_output(const std::string& msg) noexcept;

// Peak working-set size in MB. Returns 0 on platforms where it is unavailable.
std::size_t peak_working_set_mb() noexcept;

} // namespace xllama
