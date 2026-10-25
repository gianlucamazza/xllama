// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include <doctest/doctest.h>

#include "xllama/stability_attribution.h"

#include <string>

using namespace xllama;

TEST_CASE("stability attribution: exact §IV.a credit text") {
    CHECK(std::string(kStabilityAttribution) == "Powered by Stability AI");
}

TEST_CASE("stability attribution: empty licence falls back to sd-turbo family") {
    CHECK(show_stability_attribution("sd-turbo-fp16"));
    CHECK(show_stability_attribution("sd-turbo"));
    CHECK(show_stability_attribution("sd-turbo-fp32"));
    CHECK(show_stability_attribution("SD-Turbo-fp16"));
}

TEST_CASE("stability attribution: empty licence hides non-family ids") {
    CHECK_FALSE(show_stability_attribution(""));
    CHECK_FALSE(show_stability_attribution("lfm25-350m"));
    CHECK_FALSE(show_stability_attribution("embed-nomic-v2-moe"));
    CHECK_FALSE(show_stability_attribution("smollm2-360m-cpu-int4"));
    // Prefix only — "sd-turbocharged" is not a catalogue family suffix.
    CHECK_FALSE(show_stability_attribution("sd-turbocharged"));
    CHECK_FALSE(show_stability_attribution("my-sd-turbo-fp16"));
}

TEST_CASE("stability attribution: licence field wins over model id") {
    CHECK(show_stability_attribution("future-image", "Stability AI Community License"));
    CHECK(show_stability_attribution("future-image", "stability ai community licence"));
    CHECK_FALSE(show_stability_attribution("sd-turbo-fp16", "MIT"));
    CHECK_FALSE(show_stability_attribution("sd-turbo-fp16", "Apache-2.0"));
    CHECK_FALSE(show_stability_attribution("future-image", "BSD-3-Clause"));
}
