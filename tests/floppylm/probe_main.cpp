// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
#include "floppylm_internal.h"
#include "xllama/floppylm.h"
#include <iostream>
using namespace xllama::floppy;
int main(int argc, char** argv) {
    try {
        if (argc != 3)
            return 2;
        auto bytes = read_bytes(argv[1]);
        auto str = xllama::floppylm_probe(std::string(bytes.begin(), bytes.end()));
        write_bytes(argv[2], Bytes(str.begin(), str.end()));
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
}
