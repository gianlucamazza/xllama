// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
// Independent review harness. Compile against the contributor's extracted,
// unmodified expert_cache.h, not an upstream implementation or a model runtime.
#include "expert_cache.h"

#include <functional>
#include <iostream>
#include <string>
#include <vector>

namespace {
void check(bool ok, const char* message) {
    if (!ok)
        throw std::runtime_error(message);
}
void refuses(const std::function<void()>& call) {
    bool caught = false;
    try {
        call();
    } catch (const std::exception&) {
        caught = true;
    }
    check(caught, "expected refusal");
}
struct Reader final : ua50::Reader {
    enum class Mode { full, short_read, throwing, over_report } mode = Mode::full;
    std::vector<uint64_t> reads;
    size_t read(uint64_t offset, unsigned char* destination, size_t bytes) override {
        reads.push_back(offset);
        if (mode == Mode::throwing)
            throw std::runtime_error("injected read failure");
        for (size_t i = 0; i < bytes; ++i)
            destination[i] = static_cast<unsigned char>((offset + i) % 251);
        if (mode == Mode::short_read)
            return bytes - 1;
        if (mode == Mode::over_report)
            return bytes + 1;
        return bytes;
    }
};
const ua50::Tensor tensor{0, ua50::Quant::f32, 4, 2, 4, 0, 16, 32};
void bytes_match(const ua50::Cache::Lease& lease, uint64_t offset) {
    for (size_t i = 0; i < lease.size(); ++i)
        check(lease.data()[i] == (offset + i) % 251, "payload mismatch");
}
void lru() {
    Reader r;
    ua50::Cache c(r, 128, &tensor, 1, 2, 32);
    c.acquire(0, 0).reset();
    c.acquire(0, 1).reset();
    auto hit = c.acquire(0, 0);
    bytes_match(hit, 0);
    hit.reset();
    c.acquire(0, 2).reset();
    c.acquire(0, 0).reset();
    c.acquire(0, 1).reset();
    check(r.reads == std::vector<uint64_t>({0, 32, 64, 32}), "wrong LRU reads");
    check(c.counters().hits == 2 && c.counters().evictions == 2, "wrong LRU counts");
}
void pinned_and_move() {
    Reader r;
    ua50::Cache c(r, 128, &tensor, 1, 2, 32);
    auto held = c.acquire(0, 0);
    auto moved = std::move(held);
    refuses([&] { held.data(); });
    c.acquire(0, 1).reset();
    c.acquire(0, 2).reset();
    bytes_match(moved, 0);
    check(c.active_leases() == 1, "move changed lease count");
    moved.reset();
    moved.reset();
    check(c.active_leases() == 0, "reset did not drain");
}
void all_pinned() {
    Reader r;
    ua50::Cache c(r, 128, &tensor, 1, 2, 32);
    auto first = c.acquire(0, 0);
    auto second = c.acquire(0, 1);
    refuses([&] { c.acquire(0, 2); });
    check(r.reads.size() == 2, "pinned refusal performed I/O");
    bytes_match(first, 0);
    bytes_match(second, 32);
    first.reset();
    auto next = c.acquire(0, 2);
    bytes_match(next, 64);
    bytes_match(second, 32);
}
void failed_read(Reader::Mode mode) {
    Reader r;
    ua50::Cache c(r, 128, &tensor, 1, 1, 32);
    c.acquire(0, 0).reset();
    r.mode = mode;
    refuses([&] { c.acquire(0, 1); });
    check(c.active_leases() == 0, "failed read published a lease");
    r.mode = Reader::Mode::full;
    auto recovered = c.acquire(0, 1);
    bytes_match(recovered, 32);
    check(r.reads == std::vector<uint64_t>({0, 32, 32}), "failed entry was cached");
}
void budgets_and_extents() {
    Reader r;
    auto required = ua50::Cache::required_owned_bytes(2, 32);
    ua50::Cache exact(r, 128, &tensor, 1, 2, 32, required);
    check(exact.counters().arena_bytes + exact.counters().metadata_bytes == required,
          "owned budget disagreement");
    refuses([&] { ua50::Cache small(r, 128, &tensor, 1, 2, 32, required - 1); });
    refuses([&] { ua50::Cache truncated(r, 127, &tensor, 1, 2, 32); });
    refuses([&] { ua50::Cache::required_owned_bytes(2, SIZE_MAX); });
    refuses([&] { exact.acquire(0, 4); });
    refuses([&] { exact.acquire(1, 0); });
    check(r.reads.empty(), "invalid request performed I/O");
}
void mixed_quant() {
    Reader r;
    const auto q4 = ua50::block(ua50::Quant::q4_k).bytes;
    const auto q6 = ua50::block(ua50::Quant::q6_k).bytes;
    ua50::Tensor tensors[] = {{0, ua50::Quant::q4_k, 256, 1, 2, 0, q4, q4},
                              {1, ua50::Quant::q6_k, 256, 1, 2, 2 * q4, q6, q6}};
    ua50::Cache c(r, 2 * q4 + 2 * q6, tensors, 2, 2, static_cast<size_t>(q6));
    auto a = c.acquire(0, 1);
    auto b = c.acquire(1, 1);
    check(a.size() == q4 && b.size() == q6, "mixed quant slice size");
    bytes_match(a, q4);
    bytes_match(b, 2 * q4 + q6);
    a.reset();
    b.reset();
    c.acquire(0, 1).reset();
    check(c.counters().requested_bytes == 2 * q4 + q6, "request bytes wrong");
    check(c.counters().hit_bytes == q4, "hit bytes wrong");
    check(c.counters().read_bytes == q4 + q6, "logical bytes wrong");
}
} // namespace

int main() {
    const std::vector<std::pair<std::string, std::function<void()>>> cases = {
        {"lru", lru},
        {"pinned_and_move", pinned_and_move},
        {"all_pinned", all_pinned},
        {"short_read", [] { failed_read(Reader::Mode::short_read); }},
        {"throwing_reader", [] { failed_read(Reader::Mode::throwing); }},
        {"over_report", [] { failed_read(Reader::Mode::over_report); }},
        {"budgets_and_extents", budgets_and_extents},
        {"mixed_quant", mixed_quant},
    };
    size_t passed = 0;
    for (const auto& test : cases) {
        try {
            test.second();
            ++passed;
            std::cout << "PASS " << test.first << '\n';
        } catch (const std::exception& e) {
            std::cerr << "FAIL " << test.first << ": " << e.what() << '\n';
            return 1;
        }
    }
    std::cout << passed << '/' << cases.size() << " cache contract groups passed\n";
}
