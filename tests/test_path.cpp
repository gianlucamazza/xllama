// Copyright (c) 2024 Venere Labs
// SPDX-License-Identifier: MIT

#include <doctest/doctest.h>

#include "xllama/path_utils.h"

TEST_CASE("Path utils: Linux returns input unchanged") {
    std::string path = "/home/user/model.gguf";
    CHECK(xllama::resolve_model_path(path) == path);
    CHECK(xllama::resolve_local_path("output.txt") == "output.txt");
}
