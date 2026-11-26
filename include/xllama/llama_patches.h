// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
// The patches every build applies to the llama.cpp pin
// (scripts/apply-llama-patches.sh; reasons in patches/README.md), by file
// name without ".patch". /health reports them with the pin (#611), so a client
// can tell whether a given upstream fix is in the build it talks to.
// tests/test_llama_patches.cpp fails when this list and patches/ disagree.
#pragma once

#include <array>

namespace xllama {

inline constexpr std::array<const char*, 4> kLlamaPatches = {{
    "0001-uwp-appcontainer-guards",
    "0003-kv-cache-k-shift-unallocated-input",
    "0004-recurrent-state-fill-zero",
    "0005-required-tool-call-no-content-prefix",
}};

} // namespace xllama
