// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
#pragma once

#include <string>

namespace xllama {

// An operator declaration is not hardware detection. Unknown stays explicit.
inline std::string declared_benchmark_host(const std::string& hardware) {
    if (hardware == "xbox-series-s" || hardware == "xbox-series-x")
        return hardware + "-declared";
    return "xbox-unknown";
}

} // namespace xllama
