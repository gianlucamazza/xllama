// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// Heap ceiling probe — measures the RAM a GGUF may actually spend on console.
//
// The GGUF path reads weights into the heap; mmap is unavailable in the sandbox
// and enabling it measured zero benefit (uwp-constraints.md §1). So a model's
// admissibility is decided by how much heap this process can commit under
// GameOS, and that number has never been measured: `avail_phys` 5.0 GB is a
// single incidental log line, and the 3.5 GB peak gate is an H4 acceptance
// policy, not a platform limit. Promoting either to "the ceiling" is exactly
// the estimate-as-decision mistake architecture.md warns about.
//
// This commits heap in fixed steps, faulting every page in (Windows does not
// back an untouched commit, so an unwritten allocation would measure nothing),
// and records the platform's own counters after each step. It stops on the
// first failed allocation or when available physical memory drops under a
// floor — it deliberately does not run to OOM, which the PLM would resolve by
// killing the process before it could report anything.
#pragma once

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

namespace xllama {

// One committed step. Counters are read after the step's pages are faulted in.
struct RamCeilStep {
    std::size_t committed_mb = 0;   // total heap this probe has committed so far
    std::size_t working_set_mb = 0; // peak_working_set_mb() — process-wide
    std::size_t avail_phys_mb = 0;  // avail_phys_mb() — what is left to claim
    std::size_t gpu_budget_mb = 0;  // QueryVideoMemoryInfo Budget (0 off-console)
    std::size_t gpu_current_mb = 0; // QueryVideoMemoryInfo CurrentUsage
    bool ok = false;                // false: this step is the one that failed
};

struct RamCeilResult {
    std::size_t step_mb = 0;
    std::size_t limit_mb = 0;
    std::size_t floor_avail_mb = 0;
    std::size_t max_committed_mb = 0;    // the answer: last successful commit
    std::size_t avail_phys_start_mb = 0; // before the first allocation
    std::string stop_reason;             // "alloc_failed" | "avail_floor" | "limit"
    std::vector<RamCeilStep> steps;
};

// Called after each step is recorded, successful or not. Use it to append the
// row to disk immediately: if the OS kills the process mid-probe, the evidence
// up to the last surviving step is what tells us where the ceiling was.
using RamCeilSink = std::function<void(const RamCeilStep&)>;

// Commit `step_mb` at a time up to `limit_mb`, stopping early if an allocation
// fails or available physical memory falls below `floor_avail_mb`. All memory
// is released before returning. `sink` may be empty.
RamCeilResult probe_ram_ceiling(std::size_t step_mb = 256, std::size_t limit_mb = 6144,
                                std::size_t floor_avail_mb = 256, const RamCeilSink& sink = {});

// CSV serialization (self-contained schema; not the model-bench schema).
const char* ramceil_csv_header();
std::string format_ramceil_row(const RamCeilStep& s, const char* host_label);

} // namespace xllama
