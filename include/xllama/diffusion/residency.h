// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// Whether SD-Turbo can run next to the resident chat model (#548).
//
// Image generation used to run with the chat session always kept resident.
// Measured on a Series S (bench/results/2026-11-25-diffusion-coexistence/):
// the pipeline adds 2.5-3.2 GB to the process working set over whatever is
// resident, inside an app limit of 5120 MB. Five catalogue models from 0.35B to
// 3B generate fine; with lfm25-8b-a1b resident (3574 MB after context) the
// UNet load fails with "bad allocation" and the request returns 500. Always
// releasing the session would cost every small model a reload after each
// image; never releasing fails the large ones. So the decision is the memory:
// release the resident session first only when the measured headroom does not
// fit, the same way device training frees it.
#pragma once

#include "xllama/session_hub.h"

#include <cstddef>
#include <mutex>

namespace xllama::diffusion {

// Largest working-set increase measured over a resident session (3210 MB, the
// lfm25-350m cell of the 2026-11-25 receipt), rounded up.
inline constexpr std::size_t kDiffusionHeadroomMb = 3250;

// True when |working_set_mb| plus the pipeline's headroom would exceed
// |app_limit_mb|. An unknown limit (0: a host build) never releases.
inline constexpr bool diffusion_needs_session_release(std::size_t working_set_mb,
                                                      std::size_t app_limit_mb) {
    return app_limit_mb != 0 && working_set_mb + kDiffusionHeadroomMb > app_limit_mb;
}

// Apply the decision before running the pipeline. The caller holds hub.mtx
// (lock order: compute permit, then hub.mtx). Returns true when the resident
// session was released; the next chat turn reloads it, and hub.generation tells
// the GUI its KV-reuse state is gone.
inline bool make_room_for_diffusion_locked(SessionHub& hub, std::size_t working_set_mb,
                                           std::size_t app_limit_mb) {
    if (!hub.session || !diffusion_needs_session_release(working_set_mb, app_limit_mb))
        return false;
    hub.reset_locked();
    return true;
}

// make_room_for_diffusion_locked for callers that do not hold hub.mtx yet
// (they must already hold the compute permit, which comes first).
inline bool make_room_for_diffusion(SessionHub& hub, std::size_t working_set_mb,
                                    std::size_t app_limit_mb) {
    std::lock_guard<std::mutex> lk(hub.mtx);
    return make_room_for_diffusion_locked(hub, working_set_mb, app_limit_mb);
}

} // namespace xllama::diffusion
