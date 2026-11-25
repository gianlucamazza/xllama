// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// doctest entry point for xllama-tests.
#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>

#include <cstdlib>

int main(int argc, char** argv) {
    // Model-backed opt-in cases (XLLAMA_TEST_MODEL, XLLAMA_GRANITE_H_TINY, …) run
    // real prefill, and in a Debug or ASan build one logical batch can outlast
    // the 180 s product watchdog: it aborts a healthy run as "stalled" and drops
    // inference-stall.txt in the working directory. Tests assert results, not
    // speed, so the watchdog is off unless the caller exports a value; the
    // watchdog itself is covered by test_inference_progress.cpp.
    setenv("XLLAMA_INFER_STALL_TIMEOUT_S", "0", /*overwrite=*/0);
    doctest::Context context(argc, argv);
    return context.run();
}
