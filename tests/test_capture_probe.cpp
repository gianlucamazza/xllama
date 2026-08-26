#include "xllama/capture_probe.h"

#include <doctest/doctest.h>

TEST_CASE("graphics capture probe is conservative on the host") {
    const auto metadata = xllama::probe_graphics_capture();

    CHECK_FALSE(metadata.capture_path_viable);
    CHECK_FALSE(metadata.encoding_seam_exists);
    CHECK(metadata.diagnostic.find("[gcprobe]") != std::string::npos);
}

TEST_CASE("graphics capture readiness requires a real implementation") {
    CHECK_FALSE(xllama::is_graphics_capture_ready());
}
