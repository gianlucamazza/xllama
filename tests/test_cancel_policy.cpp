// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include <doctest/doctest.h>

#include "xllama/cancel_policy.h"

using namespace xllama;

TEST_CASE("cancel: every combination of the three running flags") {
    // Exhaustive rather than representative — eight cases is small enough to
    // enumerate, and the bug this replaces lived in a combination nobody
    // thought to try (image running, generic flag also set).
    CHECK(cancel_target(false, false, false) == CancelTarget::None);
    CHECK(cancel_target(false, false, true) == CancelTarget::Text);
    CHECK(cancel_target(false, true, false) == CancelTarget::Training);
    CHECK(cancel_target(false, true, true) == CancelTarget::Training);
    CHECK(cancel_target(true, false, false) == CancelTarget::Image);
    CHECK(cancel_target(true, false, true) == CancelTarget::Image);
    CHECK(cancel_target(true, true, false) == CancelTarget::Image);
    CHECK(cancel_target(true, true, true) == CancelTarget::Image);
}

TEST_CASE("cancel: the generic running flag never wins over a specific one") {
    // This is the regression. SetRunning() is called by all three jobs, so the
    // caller passes that flag as `text_running`; if precedence were the other
    // way round, an image or an epoch would be answered with the text abort.
    CHECK(cancel_target(true, false, /*text_running=*/true) == CancelTarget::Image);
    CHECK(cancel_target(false, true, /*text_running=*/true) == CancelTarget::Training);
}

TEST_CASE("cancel: idle is a no-op, not a text cancel") {
    // A back request on an idle chat reaches the same policy. Answering it with
    // Text would set an abort flag with no job to abort.
    CHECK(cancel_target(false, false, false) == CancelTarget::None);
}
