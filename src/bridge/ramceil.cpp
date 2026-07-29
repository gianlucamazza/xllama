// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include "xllama/ramceil.h"

#include "xllama/platform.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>

namespace xllama {

namespace {

// 4 KB: the x86 page. Touching one byte per page is what turns a commit into
// resident pages — an untouched allocation costs address space, not memory,
// and would measure a ceiling that does not exist.
constexpr std::size_t kPageBytes = 4096;

// Volatile sink so the page-fault writes cannot be optimized away.
volatile std::uint8_t g_ramceil_sink = 0;

RamCeilStep sample(std::size_t committed_mb, bool ok) {
    RamCeilStep s;
    s.committed_mb = committed_mb;
    s.ok = ok;
    s.working_set_mb = peak_working_set_mb();
    s.avail_phys_mb = avail_phys_mb();
    const GpuMemInfo gpu = gpu_mem_info();
    if (gpu.available) {
        s.gpu_budget_mb = gpu.budget_mb;
        s.gpu_current_mb = gpu.current_mb;
    }
    return s;
}

} // namespace

RamCeilResult probe_ram_ceiling(std::size_t step_mb, std::size_t limit_mb,
                                std::size_t floor_avail_mb, const RamCeilSink& sink) {
    RamCeilResult r;
    if (step_mb == 0)
        step_mb = 256;
    r.step_mb = step_mb;
    r.limit_mb = limit_mb;
    r.floor_avail_mb = floor_avail_mb;
    r.avail_phys_start_mb = avail_phys_mb();

    const std::size_t step_bytes = step_mb * 1024 * 1024;
    std::vector<void*> blocks;
    std::size_t committed_mb = 0;

    while (committed_mb + step_mb <= limit_mb) {
        void* p = std::malloc(step_bytes);
        if (!p) {
            r.stop_reason = "alloc_failed";
            const RamCeilStep s = sample(committed_mb, false);
            r.steps.push_back(s);
            if (sink)
                sink(s);
            break;
        }
        // Fault the block in. Write, do not read: a read of fresh anonymous
        // memory can be served by the shared zero page on Linux and never
        // become a private resident page.
        auto* bytes = static_cast<std::uint8_t*>(p);
        for (std::size_t off = 0; off < step_bytes; off += kPageBytes)
            bytes[off] = static_cast<std::uint8_t>(off);
        g_ramceil_sink += bytes[0];

        blocks.push_back(p);
        committed_mb += step_mb;

        const RamCeilStep s = sample(committed_mb, true);
        r.steps.push_back(s);
        if (sink)
            sink(s);
        r.max_committed_mb = committed_mb;

        // Stop before the OS has to choose for us: past this floor the PLM may
        // terminate the process, and a killed probe reports nothing.
        if (s.avail_phys_mb != 0 && s.avail_phys_mb < floor_avail_mb) {
            r.stop_reason = "avail_floor";
            break;
        }
    }
    if (r.stop_reason.empty())
        r.stop_reason = "limit";

    for (void* p : blocks)
        std::free(p);

    return r;
}

const char* ramceil_csv_header() {
    return "committed_mb,working_set_mb,avail_phys_mb,gpu_budget_mb,gpu_current_mb,ok,host,date\n";
}

std::string format_ramceil_row(const RamCeilStep& s, const char* host_label) {
    char date_buf[32];
    std::time_t now = std::time(nullptr);
    std::tm* tm_utc = std::gmtime(&now);
    std::strftime(date_buf, sizeof(date_buf), "%Y-%m-%dT%H:%M:%SZ", tm_utc);

    char buf[256];
    std::snprintf(buf, sizeof(buf), "%zu,%zu,%zu,%zu,%zu,%d,%s,%s\n", s.committed_mb,
                  s.working_set_mb, s.avail_phys_mb, s.gpu_budget_mb, s.gpu_current_mb,
                  s.ok ? 1 : 0, host_label ? host_label : "unknown", date_buf);
    return std::string(buf);
}

} // namespace xllama
