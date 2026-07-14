// Copyright (c) 2024 Venere Labs
// SPDX-License-Identifier: MIT

#include "xllama/membw.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <thread>
#include <vector>

namespace xllama {

namespace {

using clock_t_ = std::chrono::steady_clock;

// Run `fn(begin,end)` across `threads` workers partitioning [0,n), return seconds.
template <typename Fn> double timed_parallel(std::size_t n, int threads, Fn&& fn) {
    const auto t0 = clock_t_::now();
    if (threads <= 1) {
        fn(std::size_t{0}, n);
    } else {
        std::vector<std::thread> pool;
        pool.reserve(static_cast<std::size_t>(threads));
        const std::size_t chunk =
            (n + static_cast<std::size_t>(threads) - 1) / static_cast<std::size_t>(threads);
        for (int t = 0; t < threads; ++t) {
            const std::size_t b = std::min(n, static_cast<std::size_t>(t) * chunk);
            const std::size_t e = std::min(n, b + chunk);
            if (b >= e)
                break;
            pool.emplace_back([&fn, b, e] { fn(b, e); });
        }
        for (auto& th : pool)
            th.join();
    }
    return std::chrono::duration<double>(clock_t_::now() - t0).count();
}

// Volatile sink so the read reduction is not optimized away.
volatile double g_membw_sink = 0.0;

} // namespace

MembwResult measure_membw(std::size_t buffer_bytes, int iterations, int threads) {
    MembwResult r;
    if (threads <= 0) {
        unsigned hc = std::thread::hardware_concurrency();
        threads = hc > 0 ? static_cast<int>(hc) : 1;
    }
    if (iterations < 1)
        iterations = 1;

    // Work in doubles; round the element count so all three arrays are equal.
    const std::size_t elems = std::max<std::size_t>(buffer_bytes / sizeof(double), 1);
    const std::size_t bytes = elems * sizeof(double);
    r.buffer_bytes = bytes;
    r.iterations = iterations;
    r.threads = threads;

    std::vector<double> a(elems), b(elems), c(elems);
    // Fault every page in before timing (avoid first-touch cost polluting run 1).
    for (std::size_t i = 0; i < elems; ++i) {
        a[i] = 1.0;
        b[i] = 2.0;
        c[i] = 3.0;
    }

    const double GB = 1e9;
    double best_read = 0, best_copy = 0, best_triad = 0;
    const double scalar = 3.0;

    for (int it = 0; it < iterations; ++it) {
        // READ: sum reduction over `a` (1×bytes moved). Four independent
        // accumulators break the serial add-dependency chain so the loop is
        // limited by memory bandwidth, not FP-add latency.
        double read_s = timed_parallel(elems, threads, [&](std::size_t bi, std::size_t ei) {
            double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
            std::size_t i = bi;
            for (; i + 4 <= ei; i += 4) {
                s0 += a[i];
                s1 += a[i + 1];
                s2 += a[i + 2];
                s3 += a[i + 3];
            }
            for (; i < ei; ++i)
                s0 += a[i];
            g_membw_sink += s0 + s1 + s2 + s3;
        });
        best_read = std::max(best_read, static_cast<double>(bytes) / GB / read_s);

        // COPY: a <- c (2×bytes: read c + write a).
        double copy_s = timed_parallel(elems, threads, [&](std::size_t bi, std::size_t ei) {
            std::memcpy(a.data() + bi, c.data() + bi, (ei - bi) * sizeof(double));
        });
        best_copy = std::max(best_copy, 2.0 * static_cast<double>(bytes) / GB / copy_s);

        // TRIAD: a <- b + scalar*c (3×bytes: read b,c + write a).
        double triad_s = timed_parallel(elems, threads, [&](std::size_t bi, std::size_t ei) {
            for (std::size_t i = bi; i < ei; ++i)
                a[i] = b[i] + scalar * c[i];
        });
        best_triad = std::max(best_triad, 3.0 * static_cast<double>(bytes) / GB / triad_s);
    }

    r.read_gbs = best_read;
    r.copy_gbs = best_copy;
    r.triad_gbs = best_triad;
    return r;
}

const char* membw_csv_header() {
    return "buffer_mb,threads,read_gbs,copy_gbs,triad_gbs,host,date\n";
}

std::string format_membw_row(const MembwResult& r, const char* host_label) {
    char date_buf[32];
    std::time_t now = std::time(nullptr);
    std::tm* tm_utc = std::gmtime(&now);
    std::strftime(date_buf, sizeof(date_buf), "%Y-%m-%dT%H:%M:%SZ", tm_utc);

    char buf[256];
    std::snprintf(buf, sizeof(buf), "%zu,%d,%.2f,%.2f,%.2f,%s,%s\n", r.buffer_bytes / (1024 * 1024),
                  r.threads, r.read_gbs, r.copy_gbs, r.triad_gbs,
                  host_label ? host_label : "unknown", date_buf);
    return std::string(buf);
}

} // namespace xllama
