// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
// Which job a cancel request targets.
//
// The UI has one Cancel affordance and one back-request handler, but three jobs
// that can be running behind them — text inference, image generation and
// on-device training — and each stops by a different mechanism: the text loop
// polls an abort flag, diffusion watches diffuse-cancel.flag on disk, training
// checks its own flag between epochs. Picking the wrong one is silent: the
// caller disables the Cancel button, reports "cancelling", and the job keeps
// running with no way left to stop it.
//
// That failure shipped. MainPageController::SetRunning() is called by all three
// jobs, so m_is_running means "a job is running", not "text is running"; a
// back-request handler that read it and set the text abort flag looked correct
// and cancelled nothing when an image was on screen.
//
// The decision is pure policy over three booleans, so it lives here and is
// exhaustively tested on the host, where the UI it serves cannot be built.
#pragma once

namespace xllama {

enum class CancelTarget {
    None,     // nothing is running — a cancel request must be a no-op
    Image,    // diffuse-cancel.flag
    Training, // the training abort flag, honoured between epochs
    Text,     // the inference abort flag, polled per token
};

// Precedence is image > training > text, and it is not arbitrary: the generic
// "a job is running" flag is set by all three, so the specific flags are the
// only ones that identify the job. Testing them first is what makes the generic
// flag safe to pass in as `text_running`.
inline constexpr CancelTarget cancel_target(bool image_running, bool training_running,
                                            bool text_running) {
    if (image_running)
        return CancelTarget::Image;
    if (training_running)
        return CancelTarget::Training;
    if (text_running)
        return CancelTarget::Text;
    return CancelTarget::None;
}

} // namespace xllama
