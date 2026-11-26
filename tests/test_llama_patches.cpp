// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include <doctest/doctest.h>

#include "xllama/llama_patches.h"

#include <algorithm>
#include <filesystem>
#include <string>
#include <vector>

// /health reports kLlamaPatches as what the build applied (#611): a patch
// added to or dropped from patches/ without this list would report a build
// that does not exist.
TEST_CASE("kLlamaPatches lists exactly the patches/0*.patch files, in order") {
    std::vector<std::string> on_disk;
    for (const auto& entry : std::filesystem::directory_iterator(LLAMA_PATCHES_DIR)) {
        const std::string name = entry.path().filename().string();
        const std::string suffix = ".patch";
        if (name.size() > suffix.size() && name[0] == '0' &&
            name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0)
            on_disk.push_back(name.substr(0, name.size() - suffix.size()));
    }
    std::sort(on_disk.begin(), on_disk.end());
    const std::vector<std::string> listed(xllama::kLlamaPatches.begin(),
                                          xllama::kLlamaPatches.end());
    CHECK(listed == on_disk);
}
