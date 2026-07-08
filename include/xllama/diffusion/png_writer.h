// Copyright (c) 2024 Venere Labs
// SPDX-License-Identifier: MIT
//
// Minimal, dependency-free PNG writer for 8-bit RGB images. The console diffusion
// pipeline (uwp/diffuse.cpp) decodes the VAE output to RGB and writes a PNG to
// LocalState for Device Portal fetch; a self-contained encoder avoids pulling in
// libpng/zlib in the UWP AppContainer. Uses stored (uncompressed) DEFLATE blocks
// inside a valid zlib stream — larger files, trivial and correct. Header-only and
// pure so the CRC/adler/stream framing is unit-tested on the host.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace xllama::diffusion {

namespace detail {

inline uint32_t crc32_png(const uint8_t* data, size_t len, uint32_t crc = 0xFFFFFFFFu) {
    for (size_t i = 0; i < len; ++i) {
        crc ^= data[i];
        for (int k = 0; k < 8; ++k)
            crc = (crc >> 1) ^ (0xEDB88320u & (~(crc & 1) + 1));
    }
    return crc;
}

inline uint32_t adler32(const uint8_t* data, size_t len) {
    uint32_t a = 1, b = 0;
    for (size_t i = 0; i < len; ++i) {
        a = (a + data[i]) % 65521u;
        b = (b + a) % 65521u;
    }
    return (b << 16) | a;
}

inline void put_be32(std::vector<uint8_t>& v, uint32_t x) {
    v.push_back((uint8_t)(x >> 24));
    v.push_back((uint8_t)(x >> 16));
    v.push_back((uint8_t)(x >> 8));
    v.push_back((uint8_t)x);
}

inline void chunk(std::vector<uint8_t>& out, const char type[4], const std::vector<uint8_t>& data) {
    put_be32(out, (uint32_t)data.size());
    const size_t crc_start = out.size();
    out.insert(out.end(), type, type + 4);
    out.insert(out.end(), data.begin(), data.end());
    const uint32_t crc = crc32_png(out.data() + crc_start, out.size() - crc_start) ^ 0xFFFFFFFFu;
    put_be32(out, crc);
}

} // namespace detail

// Encode an 8-bit RGB image (rgb.size() == 3*w*h, row-major HWC) to PNG bytes.
inline std::vector<uint8_t> encode_png_rgb(int w, int h, const std::vector<uint8_t>& rgb) {
    using namespace detail;
    std::vector<uint8_t> out = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};

    std::vector<uint8_t> ihdr;
    put_be32(ihdr, (uint32_t)w);
    put_be32(ihdr, (uint32_t)h);
    ihdr.push_back(8); // bit depth
    ihdr.push_back(2); // color type 2 = truecolor RGB
    ihdr.push_back(0); // compression
    ihdr.push_back(0); // filter
    ihdr.push_back(0); // interlace
    chunk(out, "IHDR", ihdr);

    // Raw filtered scanlines: filter byte 0 (none) + RGB per row.
    std::vector<uint8_t> raw;
    raw.reserve((size_t)h * (1 + 3 * w));
    for (int y = 0; y < h; ++y) {
        raw.push_back(0);
        const uint8_t* row = rgb.data() + (size_t)y * 3 * w;
        raw.insert(raw.end(), row, row + 3 * w);
    }

    // zlib stream: header + stored DEFLATE blocks + adler32.
    std::vector<uint8_t> zlib = {0x78, 0x01};
    size_t off = 0;
    while (off < raw.size()) {
        const size_t n = std::min<size_t>(65535, raw.size() - off);
        const bool last = (off + n) >= raw.size();
        zlib.push_back(last ? 1 : 0); // BFINAL, BTYPE=00 (stored)
        zlib.push_back((uint8_t)(n & 0xFF));
        zlib.push_back((uint8_t)((n >> 8) & 0xFF));
        zlib.push_back((uint8_t)(~n & 0xFF));
        zlib.push_back((uint8_t)((~n >> 8) & 0xFF));
        zlib.insert(zlib.end(), raw.begin() + off, raw.begin() + off + n);
        off += n;
    }
    put_be32(zlib, adler32(raw.data(), raw.size()));
    chunk(out, "IDAT", zlib);

    chunk(out, "IEND", {});
    return out;
}

// Write the PNG to a file. Returns false on I/O error. (fopen path is UTF-8; the
// UWP caller passes a resolved LocalState path.)
inline bool write_png_rgb(const std::string& path, int w, int h, const std::vector<uint8_t>& rgb) {
    const std::vector<uint8_t> bytes = encode_png_rgb(w, h, rgb);
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f)
        return false;
    const size_t wrote = std::fwrite(bytes.data(), 1, bytes.size(), f);
    std::fclose(f);
    return wrote == bytes.size();
}

} // namespace xllama::diffusion
