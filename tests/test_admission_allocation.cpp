// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
// Separate process: allocation failure injection must not affect the main host suite.
#include "xllama/api_connection.h"

#include <cstdio>
#include <cstdlib>
#include <new>

static int allocations_before_failure = -1;

void* operator new(std::size_t bytes) {
    if (allocations_before_failure >= 0 && allocations_before_failure-- == 0)
        throw std::bad_alloc();
    if (void* ptr = std::malloc(bytes == 0 ? 1 : bytes))
        return ptr;
    throw std::bad_alloc();
}

void operator delete(void* ptr) noexcept {
    std::free(ptr);
}

void operator delete(void* ptr, std::size_t) noexcept {
    std::free(ptr);
}

// Fail each allocation in turn until acquisition succeeds. This also accommodates
// future implementations that combine the permit and control block.
template <typename Gate, typename CheckOwners>
static bool check_acquisition(Gate& gate, CheckOwners check_owners) {
    for (int i = 0;; ++i) {
        allocations_before_failure = i;
        try {
            auto acquired = gate.try_acquire();
            allocations_before_failure = -1;
            return static_cast<bool>(acquired);
        } catch (const std::bad_alloc&) {
            allocations_before_failure = -1;
            if (!check_owners()) {
                std::fprintf(stderr, "Admission state corrupted after allocation failure %d\n",
                             i + 1);
                return false;
            }
        }
    }
}

int main() {
    xllama::ApiConnectionGate connections;
    auto existing = connections.try_acquire();
    if (!check_acquisition(connections, [&] { return connections.in_flight() == 1; }))
        return 1;

    xllama::ComputeJobAdmission compute;
    if (!check_acquisition(compute, [&] {
            auto next = compute.try_acquire();
            return next && !compute.try_acquire();
        }))
        return 1;
    return 0;
}
