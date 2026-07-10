// Copyright (c) 2024 Venere Labs
// SPDX-License-Identifier: MIT

#include <doctest/doctest.h>

#include "xllama/path_utils.h"

#include <filesystem>
#include <fstream>

#include <unistd.h>

TEST_CASE("Path utils: Linux returns input unchanged") {
    std::string path = "/home/user/model.gguf";
    CHECK(xllama::resolve_model_path(path) == path);
    CHECK(xllama::resolve_local_path("output.txt") == "output.txt");
}

TEST_CASE("first_gguf_in_dir: descends catalogue dirs, passes files through") {
    namespace fs = std::filesystem;
    // PID-suffixed so concurrent test invocations don't clobber each other.
    const fs::path dir =
        fs::temp_directory_path() / ("xllama-test-gguf-dir-" + std::to_string(::getpid()));
    fs::remove_all(dir);
    fs::create_directories(dir);

    SUBCASE("non-directory input is returned unchanged") {
        CHECK(xllama::first_gguf_in_dir("/no/such/model.gguf") == "/no/such/model.gguf");
    }
    SUBCASE("directory without gguf yields empty") {
        CHECK(xllama::first_gguf_in_dir(dir.string()).empty());
    }
    SUBCASE("directory with a gguf yields the file path") {
        const fs::path gguf = dir / "model-Q4_K_M.gguf";
        std::ofstream(gguf) << "stub";
        std::ofstream(dir / "LICENSE.txt") << "license"; // ignored sibling
        CHECK(xllama::first_gguf_in_dir(dir.string()) == gguf.string());
    }

    fs::remove_all(dir);
}
