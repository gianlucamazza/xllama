// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include "xllama/diskbw.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <numeric>
#include <random>
#include <thread>
#include <vector>

#ifdef _WIN32
    #include <windows.h>
#else
    #include <fcntl.h>
    #include <sys/stat.h>
    #include <unistd.h>
#endif

namespace xllama {

namespace {

using clock_t_ = std::chrono::steady_clock;

// Buffer alignment for O_DIRECT / FILE_FLAG_NO_BUFFERING (sector multiple).
constexpr std::size_t kAlign = 4096;

#ifdef _WIN32
std::wstring widen(const std::string& s) {
    if (s.empty())
        return {};
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
    std::wstring w(static_cast<std::size_t>(n > 0 ? n - 1 : 0), L'\0');
    if (n > 1)
        MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, w.data(), n);
    return w;
}

// pread-shaped file handle. CreateFile2 is the AppContainer-legal opener and
// exists on desktop Win8+ too, so one path serves both.
struct FileHandle {
    HANDLE h = INVALID_HANDLE_VALUE;
    bool open_read(const std::string& path, bool unbuffered) {
        CREATEFILE2_EXTENDED_PARAMETERS p{};
        p.dwSize = sizeof(p);
        p.dwFileAttributes = FILE_ATTRIBUTE_NORMAL;
        p.dwFileFlags = FILE_FLAG_SEQUENTIAL_SCAN | (unbuffered ? FILE_FLAG_NO_BUFFERING : 0);
        h = CreateFile2(widen(path).c_str(), GENERIC_READ, FILE_SHARE_READ, OPEN_EXISTING, &p);
        return h != INVALID_HANDLE_VALUE;
    }
    bool pread_block(void* buf, std::size_t len, std::uint64_t off) const {
        OVERLAPPED ov{};
        ov.Offset = static_cast<DWORD>(off & 0xffffffffu);
        ov.OffsetHigh = static_cast<DWORD>(off >> 32);
        DWORD got = 0;
        if (!ReadFile(h, buf, static_cast<DWORD>(len), &got, &ov))
            return false;
        return got == len;
    }
    ~FileHandle() {
        if (h != INVALID_HANDLE_VALUE)
            CloseHandle(h);
    }
};

void* aligned_alloc_(std::size_t bytes) {
    return _aligned_malloc(bytes, kAlign);
}
void aligned_free_(void* p) {
    _aligned_free(p);
}

bool file_size_of(const std::string& path, std::uint64_t* out) {
    WIN32_FILE_ATTRIBUTE_DATA fad{};
    if (!GetFileAttributesExW(widen(path).c_str(), GetFileExInfoStandard, &fad))
        return false;
    *out = (static_cast<std::uint64_t>(fad.nFileSizeHigh) << 32) | fad.nFileSizeLow;
    return true;
}

void drop_cache_hint(const std::string&) {} // no per-file equivalent without privileges
#else
struct FileHandle {
    int fd = -1;
    bool open_read(const std::string& path, bool unbuffered) {
        int flags = O_RDONLY;
    #ifdef O_DIRECT
        if (unbuffered)
            flags |= O_DIRECT;
    #else
        if (unbuffered)
            return false;
    #endif
        fd = ::open(path.c_str(), flags);
        return fd >= 0;
    }
    bool pread_block(void* buf, std::size_t len, std::uint64_t off) const {
        std::size_t done = 0;
        while (done < len) {
            const ssize_t n = ::pread(fd, static_cast<char*>(buf) + done, len - done,
                                      static_cast<off_t>(off + done));
            if (n <= 0)
                return false;
            done += static_cast<std::size_t>(n);
        }
        return true;
    }
    ~FileHandle() {
        if (fd >= 0)
            ::close(fd);
    }
};

void* aligned_alloc_(std::size_t bytes) {
    void* p = nullptr;
    if (posix_memalign(&p, kAlign, bytes) != 0)
        return nullptr;
    return p;
}
void aligned_free_(void* p) {
    free(p);
}

bool file_size_of(const std::string& path, std::uint64_t* out) {
    struct stat st{};
    if (::stat(path.c_str(), &st) != 0)
        return false;
    *out = static_cast<std::uint64_t>(st.st_size);
    return true;
}

// Buffered fallback: evict the file from page cache so every pass touches the
// device. Best-effort — a failure just means warmer numbers.
void drop_cache_hint(const std::string& path) {
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd >= 0) {
        (void)posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
        ::close(fd);
    }
}
#endif

// Volatile sink so reads are not optimized away.
volatile std::uint64_t g_diskbw_sink = 0;

} // namespace

bool ensure_diskbw_file(const std::string& path, std::size_t file_bytes, std::string* err) {
    std::uint64_t existing = 0;
    if (file_size_of(path, &existing) && existing == file_bytes)
        return true;

#ifdef _WIN32
    FILE* fp = _wfopen(widen(path).c_str(), L"wb");
#else
    FILE* fp = std::fopen(path.c_str(), "wb");
#endif
    if (!fp) {
        if (err)
            *err = "cannot create " + path;
        return false;
    }
    // Incompressible chunks (xorshift64) so transparent compression on the
    // filesystem cannot inflate the measured read rate.
    constexpr std::size_t kChunk = static_cast<std::size_t>(8) << 20;
    std::vector<std::uint64_t> chunk(kChunk / sizeof(std::uint64_t));
    std::uint64_t x = 0x9e3779b97f4a7c15ull;
    bool ok = true;
    for (std::size_t written = 0; written < file_bytes && ok; written += kChunk) {
        for (auto& v : chunk) {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            v = x;
        }
        const std::size_t n = std::min(kChunk, file_bytes - written);
        ok = std::fwrite(chunk.data(), 1, n, fp) == n;
    }
    ok = std::fclose(fp) == 0 && ok;
    if (!ok && err)
        *err = "short write creating " + path + " (disk full?)";
    return ok;
}

DiskbwResult measure_diskbw(const std::string& path, std::size_t file_bytes,
                            std::size_t block_bytes, bool random, int threads, int iterations,
                            bool try_unbuffered) {
    DiskbwResult r;
    if (threads < 1)
        threads = 1;
    if (iterations < 1)
        iterations = 1;
    block_bytes = ((block_bytes + (1 << 20) - 1) >> 20) << 20; // round up to 1 MiB
    const std::size_t n_blocks = file_bytes / block_bytes;
    if (n_blocks == 0) {
        r.error_msg = "file smaller than one block";
        return r;
    }
    const std::size_t measured_bytes = n_blocks * block_bytes;
    r.file_bytes = measured_bytes;
    r.block_bytes = block_bytes;
    r.threads = threads;
    r.iterations = iterations;
    r.random = random;

    // Probe whether the unbuffered open is accepted at all (UWP AppContainer
    // may refuse FILE_FLAG_NO_BUFFERING; non-Linux POSIX has no O_DIRECT).
    bool unbuffered = false;
    if (try_unbuffered) {
        FileHandle probe;
        unbuffered = probe.open_read(path, /*unbuffered=*/true);
    }
    r.unbuffered = unbuffered;

    // Block visit order: identity for sequential, fixed-seed shuffle for random
    // (same order every run, so results are comparable across runs/machines).
    std::vector<std::uint32_t> order(n_blocks);
    std::iota(order.begin(), order.end(), 0u);
    if (random) {
        std::mt19937 rng(0x5eed);
        std::shuffle(order.begin(), order.end(), rng);
    }

    const double GB = 1e9;
    for (int it = 0; it < iterations; ++it) {
        if (!unbuffered)
            drop_cache_hint(path);

        std::atomic<bool> failed{false};
        const auto t0 = clock_t_::now();
        std::vector<std::thread> pool;
        pool.reserve(static_cast<std::size_t>(threads));
        const std::size_t chunk =
            (n_blocks + static_cast<std::size_t>(threads) - 1) / static_cast<std::size_t>(threads);
        for (int t = 0; t < threads; ++t) {
            const std::size_t b = std::min(n_blocks, static_cast<std::size_t>(t) * chunk);
            const std::size_t e = std::min(n_blocks, b + chunk);
            if (b >= e)
                break;
            pool.emplace_back([&, b, e] {
                FileHandle fh;
                void* buf = aligned_alloc_(block_bytes);
                if (!buf || !fh.open_read(path, unbuffered)) {
                    failed = true;
                    if (buf)
                        aligned_free_(buf);
                    return;
                }
                std::uint64_t sink = 0;
                for (std::size_t i = b; i < e && !failed; ++i) {
                    const std::uint64_t off = static_cast<std::uint64_t>(order[i]) * block_bytes;
                    if (!fh.pread_block(buf, block_bytes, off)) {
                        failed = true;
                        break;
                    }
                    sink += static_cast<const std::uint64_t*>(buf)[0];
                }
                g_diskbw_sink += sink;
                aligned_free_(buf);
            });
        }
        for (auto& th : pool)
            th.join();
        const double secs = std::chrono::duration<double>(clock_t_::now() - t0).count();
        if (failed) {
            r.error_msg = "read failed (pattern=" + std::string(random ? "rnd" : "seq") + ")";
            r.read_gbs_first = r.read_gbs_best = 0.0;
            return r;
        }
        const double gbs = static_cast<double>(measured_bytes) / GB / secs;
        if (it == 0)
            r.read_gbs_first = gbs;
        r.read_gbs_best = std::max(r.read_gbs_best, gbs);
    }
    return r;
}

const char* diskbw_csv_header() {
    return "file_mb,block_kb,pattern,threads,unbuffered,read_gbs_first,read_gbs_best,host,date\n";
}

std::string format_diskbw_row(const DiskbwResult& r, const char* host_label) {
    char date_buf[32];
    std::time_t now = std::time(nullptr);
    std::tm* tm_utc = std::gmtime(&now);
    std::strftime(date_buf, sizeof(date_buf), "%Y-%m-%dT%H:%M:%SZ", tm_utc);

    char buf[320];
    std::snprintf(buf, sizeof(buf), "%zu,%zu,%s,%d,%d,%.2f,%.2f,%s,%s\n",
                  r.file_bytes / (1024 * 1024), r.block_bytes / 1024, r.random ? "rnd" : "seq",
                  r.threads, r.unbuffered ? 1 : 0, r.read_gbs_first, r.read_gbs_best,
                  host_label ? host_label : "unknown", date_buf);
    return std::string(buf);
}

} // namespace xllama
