// llama-mmap-uwp.cpp
// UWP replacement for llama.cpp/src/llama-mmap.cpp.
//
// Design:
// - llama_file: full implementation using ReadFile/WriteFile/SetFilePointerEx
//   which are available in WINAPI_PARTITION_APP.
//   FormatMessageA (desktop-only) is replaced with FormatMessageW.
// - llama_mmap: stub that throws if use_mmap=true reaches it.
//   In practice, llama-bridge.cpp sets mparams.use_mmap=false so this is
//   never instantiated. Stage 1E will implement CreateFileMappingFromApp.
// - llama_mlock: uses VirtualLock/VirtualUnlock (available in UWP).

#ifdef XLLAMA_UWP

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include "../llama.cpp/src/llama-mmap.h"
#include "../llama.cpp/ggml/src/ggml.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <algorithm>

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

static std::string win32_error_str(DWORD code) {
    wchar_t buf[512] = {};
    FormatMessageW(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                   nullptr, code, 0, buf, (DWORD)std::size(buf), nullptr);
    // convert to narrow string
    char narrow[512] = {};
    WideCharToMultiByte(CP_UTF8, 0, buf, -1, narrow, (int)std::size(narrow), nullptr, nullptr);
    // trim trailing newline
    size_t n = strlen(narrow);
    while (n > 0 && (narrow[n-1] == '\n' || narrow[n-1] == '\r')) narrow[--n] = 0;
    return narrow;
}

// ---------------------------------------------------------------------------
// llama_file
// ---------------------------------------------------------------------------

struct llama_file::impl {
    FILE   * fp      = nullptr;
    HANDLE   fp_win32;
    size_t   sz      = 0;
    bool     owns_fp = true;

    impl(const char * fname, const char * mode, bool /*use_direct_io*/ = false) {
        fp = ggml_fopen(fname, mode);
        if (!fp) throw std::runtime_error(std::string("failed to open ") + fname);
        fp_win32 = (HANDLE)_get_osfhandle(_fileno(fp));
        seek(0, SEEK_END);
        sz = tell();
        seek(0, SEEK_SET);
    }

    impl(FILE * file) : owns_fp(false) {
        fp = file;
        fp_win32 = (HANDLE)_get_osfhandle(_fileno(fp));
        seek(0, SEEK_END);
        sz = tell();
        seek(0, SEEK_SET);
    }

    ~impl() {
        if (fp && owns_fp) std::fclose(fp);
    }

    size_t tell() const {
        LARGE_INTEGER li{};
        if (!SetFilePointerEx(fp_win32, li, &li, FILE_CURRENT))
            throw std::runtime_error("tell error: " + win32_error_str(GetLastError()));
        return (size_t)li.QuadPart;
    }

    void seek(size_t offset, int whence) const {
        LARGE_INTEGER li{};
        li.QuadPart = (LONGLONG)offset;
        if (!SetFilePointerEx(fp_win32, li, nullptr, whence))
            throw std::runtime_error("seek error: " + win32_error_str(GetLastError()));
    }

    void read_raw(void * ptr, size_t len) {
        size_t done = 0;
        while (done < len) {
            DWORD chunk = (DWORD)std::min<size_t>(len - done, 64 * 1024 * 1024);
            DWORD got   = 0;
            if (!ReadFile(fp_win32, (char *)ptr + done, chunk, &got, nullptr))
                throw std::runtime_error("read error: " + win32_error_str(GetLastError()));
            if (got == 0) throw std::runtime_error("unexpected EOF");
            done += got;
        }
    }

    uint32_t read_u32() {
        uint32_t v; read_raw(&v, sizeof(v)); return v;
    }

    void write_raw(const void * ptr, size_t len) const {
        size_t done = 0;
        while (done < len) {
            DWORD chunk = (DWORD)std::min<size_t>(len - done, 64 * 1024 * 1024);
            DWORD wrote = 0;
            if (!WriteFile(fp_win32, (const char *)ptr + done, chunk, &wrote, nullptr))
                throw std::runtime_error("write error: " + win32_error_str(GetLastError()));
            if (wrote == 0) throw std::runtime_error("write returned 0 bytes");
            done += wrote;
        }
    }

    void write_u32(uint32_t val) const { write_raw(&val, sizeof(val)); }

    bool has_direct_io() const { return false; }

    size_t read_alignment() const { return 1; }
};

llama_file::llama_file(const char * fname, const char * mode, bool use_direct_io)
    : pimpl(std::make_unique<impl>(fname, mode, use_direct_io)) {}

llama_file::llama_file(FILE * file)
    : pimpl(std::make_unique<impl>(file)) {}

llama_file::~llama_file() = default;

size_t llama_file::tell()           const { return pimpl->tell(); }
size_t llama_file::size()           const { return pimpl->sz; }
size_t llama_file::read_alignment() const { return pimpl->read_alignment(); }
bool   llama_file::has_direct_io()  const { return pimpl->has_direct_io(); }

int llama_file::file_id() const { return _fileno(pimpl->fp); }

void llama_file::seek(size_t offset, int whence) const { pimpl->seek(offset, whence); }
void llama_file::read_raw(void * ptr, size_t len)       { pimpl->read_raw(ptr, len); }
void llama_file::read_raw_unsafe(void * ptr, size_t len){ pimpl->read_raw(ptr, len); }

void llama_file::read_aligned_chunk(void * dest, size_t len) {
    // No O_DIRECT on UWP; just delegate to regular read.
    pimpl->read_raw(dest, len);
}

uint32_t llama_file::read_u32()                          { return pimpl->read_u32(); }
void llama_file::write_raw(const void * ptr, size_t len) const { pimpl->write_raw(ptr, len); }
void llama_file::write_u32(uint32_t val)                 const { pimpl->write_u32(val); }

size_t llama_path_max() { return MAX_PATH; }

// ---------------------------------------------------------------------------
// llama_mmap  (stub — never instantiated with use_mmap=false)
// ---------------------------------------------------------------------------

struct llama_mmap::impl {
    impl(llama_file *, size_t, bool) {
        throw std::runtime_error(
            "mmap not supported on UWP (use_mmap must be false). "
            "Stage 1E will implement CreateFileMappingFromApp.");
    }
    size_t sz = 0;
    void * addr_ptr = nullptr;
};

const bool llama_mmap::SUPPORTED = false;

llama_mmap::llama_mmap(llama_file * file, size_t prefetch, bool numa)
    : pimpl(std::make_unique<impl>(file, prefetch, numa)) {}

llama_mmap::~llama_mmap() = default;

size_t llama_mmap::size()             const { return pimpl->sz; }
void * llama_mmap::addr()             const { return pimpl->addr_ptr; }
void   llama_mmap::unmap_fragment(size_t, size_t) {}

// ---------------------------------------------------------------------------
// llama_mlock  (VirtualLock / VirtualUnlock — available in UWP)
// ---------------------------------------------------------------------------

struct llama_mlock::impl {
    void * ptr  = nullptr;
    size_t locked = 0;

    void init(void * p) { ptr = p; }

    void grow_to(size_t target) {
        if (!ptr || target <= locked) return;
        if (VirtualLock((char *)ptr + locked, target - locked))
            locked = target;
        // silently ignore failure: VirtualLock requires SE_LOCK_MEMORY_NAME
    }

    ~impl() {
        if (ptr && locked) VirtualUnlock(ptr, locked);
    }
};

const bool llama_mlock::SUPPORTED = true;

llama_mlock::llama_mlock()  : pimpl(std::make_unique<impl>()) {}
llama_mlock::~llama_mlock() = default;

void llama_mlock::init(void * ptr)          { pimpl->init(ptr); }
void llama_mlock::grow_to(size_t sz)        { pimpl->grow_to(sz); }

#endif // XLLAMA_UWP
