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

// Default thread count for the llama.cpp backend. On UWP this is capped at 6:
// ggml's spin-wait threadpool livelocks at t7/t8 on the console (~6 cores left
// to the app in Dev Mode, no thread affinity in AppContainer) — measured
// 2026-07-08 (phase35-llamacpp-scaling, t6 optimal) and re-hit 2026-07-10 on
// the first unified bench. An explicit n_threads always wins over this.
int detect_threads_llama() noexcept;

// Emit a log line. On UWP: OutputDebugStringA; otherwise: stderr.
void log_output(const char* msg) noexcept;
void log_output(const std::string& msg) noexcept;

// Peak working-set size in MB. Returns 0 on platforms where it is unavailable.
std::size_t peak_working_set_mb() noexcept;

// Per-process video memory (DXGI LOCAL segment). On Xbox Series S the OS
// grants an App-mode UWP a Budget of roughly 768 MB; CurrentUsage climbing
// toward the model size is direct evidence the DML EP resides on the GPU.
struct GpuMemInfo {
    std::size_t current_mb = 0; // IDXGIAdapter3::QueryVideoMemoryInfo CurrentUsage
    std::size_t budget_mb = 0;  // OS-granted budget for this process
    bool available = false;     // false on non-UWP builds or DXGI failure
};
GpuMemInfo gpu_mem_info() noexcept;

// Pin the process CWD to ApplicationData LocalFolder (UWP only, no-op elsewhere).
// Relative paths — e.g. the ORT enable_profiling prefix — then land in LocalState
// instead of the read-only install root.
void set_cwd_to_local_folder() noexcept;

} // namespace xllama
