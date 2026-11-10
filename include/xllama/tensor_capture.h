// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// Opt-in intermediate tensor capture for numerical root-cause diagnosis (#420).
//
// The llama.cpp path installs a ggml scheduler eval callback only when a
// capture directory is set. The callback asks for nodes whose graph name
// matches the spec, copies them to host memory after the node completed and
// writes one record per (graph evaluation, node). Two captures of the same
// prompt on different backends can then be diffed in graph order by
// scripts/compare-tensors.py to find the first divergent operator/layer.
//
// This header is the pure, host-testable part: spec parsing, name matching,
// the byte budget and the on-disk record format. It does not include ggml.
//
// Format, in <dir>:
//   NNNNNN.f32       raw float32 little-endian, row-major in ggml order
//                    (ne0 fastest). `last` mode keeps only the final slice of
//                    the outermost non-singleton dimension (the last token of
//                    that ubatch for an activation); see tensor_capture_values.
//   manifest.jsonl   one JSON object per record, in evaluation order:
//                    { seq, eval, name, op, type, ne:[4], mode, values, file }
//                    plus { skipped:"reason" } entries for matched nodes that
//                    could not be copied (unsupported type, budget exhausted).
//
// Capture changes how the scheduler splits the graph (each asked node forces a
// synchronisation point), so a capture-on run must be compared against a
// capture-off run of the same arm before its tensors are trusted.

#pragma once

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "xllama/json_utils.h"

namespace xllama {

struct TensorCaptureSpec {
    // Comma-separated patterns; a trailing '*' matches any suffix. Empty = none.
    std::vector<std::string> patterns;
    bool last_column_only = true; // `last` (default) or `full`
    uint64_t max_bytes = 256ull << 20;
};

inline std::string tensor_capture_trim(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && (s[a] == ' ' || s[a] == '\t' || s[a] == '\r' || s[a] == '\n'))
        ++a;
    while (b > a && (s[b - 1] == ' ' || s[b - 1] == '\t' || s[b - 1] == '\r' || s[b - 1] == '\n'))
        --b;
    return s.substr(a, b - a);
}

// Spec text: `patterns[;mode=last|full][;max_mb=N]`, e.g.
//   "l_out-*,result_norm,result_output;mode=last;max_mb=128"
// Returns false (and sets err) on an unknown option, bad value or no pattern.
inline bool parse_tensor_capture_spec(const std::string& text, TensorCaptureSpec& out,
                                      std::string& err) {
    out = TensorCaptureSpec{};
    std::vector<std::string> parts;
    size_t start = 0;
    for (;;) {
        const size_t p = text.find(';', start);
        parts.push_back(tensor_capture_trim(text.substr(start, p - start)));
        if (p == std::string::npos)
            break;
        start = p + 1;
    }
    {
        const std::string& pats = parts[0];
        size_t s = 0;
        for (;;) {
            const size_t p = pats.find(',', s);
            std::string one = tensor_capture_trim(pats.substr(s, p - s));
            if (!one.empty()) {
                if (one.find('*') != std::string::npos && one.find('*') != one.size() - 1) {
                    err = "'*' is only allowed at the end of a pattern: " + one;
                    return false;
                }
                out.patterns.push_back(one);
            }
            if (p == std::string::npos)
                break;
            s = p + 1;
        }
    }
    if (out.patterns.empty()) {
        err = "no tensor name pattern";
        return false;
    }
    for (size_t i = 1; i < parts.size(); ++i) {
        const std::string& opt = parts[i];
        if (opt.empty())
            continue;
        if (opt == "mode=last") {
            out.last_column_only = true;
        } else if (opt == "mode=full") {
            out.last_column_only = false;
        } else if (opt.rfind("max_mb=", 0) == 0) {
            const std::string v = opt.substr(7);
            if (v.empty() || v.size() > 6 ||
                v.find_first_not_of("0123456789") != std::string::npos) {
                err = "bad max_mb: " + v;
                return false;
            }
            const uint64_t mb = std::stoull(v);
            if (mb == 0 || mb > 4096) {
                err = "max_mb out of range (1..4096): " + v;
                return false;
            }
            out.max_bytes = mb << 20;
        } else {
            err = "unknown option: " + opt;
            return false;
        }
    }
    return true;
}

inline bool tensor_capture_matches(const TensorCaptureSpec& spec, const char* name) {
    if (!name || !*name)
        return false;
    const std::string n(name);
    for (const auto& p : spec.patterns) {
        if (!p.empty() && p.back() == '*') {
            if (n.compare(0, p.size() - 1, p, 0, p.size() - 1) == 0)
                return true;
        } else if (n == p) {
            return true;
        }
    }
    return false;
}

// Total element count of a tensor of shape ne[4]; 0 for an empty/invalid shape.
inline uint64_t tensor_capture_total(const int64_t ne[4]) {
    uint64_t total = 1;
    for (int i = 0; i < 4; ++i) {
        if (ne[i] <= 0)
            return 0;
        total *= static_cast<uint64_t>(ne[i]);
    }
    return total;
}

// Number of float values a record keeps for a contiguous tensor of shape ne[4].
// `full` keeps everything. `last` keeps the final slice along the outermost
// non-singleton dimension, which for llama.cpp activations is the token axis:
// [n_embd, n_tokens] keeps the last token's n_embd values, [head_dim, n_head,
// n_tokens] keeps the last token's head_dim * n_head. Tensors with no
// non-singleton dimension above ne0 (1-D, or one output row) are kept whole.
// The kept values are always the tail of the contiguous buffer.
inline uint64_t tensor_capture_values(const TensorCaptureSpec& spec, const int64_t ne[4]) {
    const uint64_t total = tensor_capture_total(ne);
    if (total == 0 || !spec.last_column_only)
        return total;
    for (int i = 3; i >= 1; --i)
        if (ne[i] > 1)
            return total / static_cast<uint64_t>(ne[i]);
    return total;
}

// Writes records and the manifest; tracks the byte budget. Not thread-safe:
// the scheduler calls the eval callback from the compute thread only.
class TensorCaptureWriter {
  public:
    TensorCaptureWriter(std::string dir, TensorCaptureSpec spec)
        : dir_(std::move(dir)), spec_(std::move(spec)) {}
    ~TensorCaptureWriter() {
        if (manifest_)
            std::fclose(manifest_);
    }
    TensorCaptureWriter(const TensorCaptureWriter&) = delete;
    TensorCaptureWriter& operator=(const TensorCaptureWriter&) = delete;

    const TensorCaptureSpec& spec() const {
        return spec_;
    }
    bool failed() const {
        return failed_;
    }
    uint64_t bytes_written() const {
        return bytes_;
    }
    int records() const {
        return seq_;
    }
    int skipped() const {
        return skipped_;
    }

    // Call once per graph evaluation (one physical ubatch), before it runs.
    void begin_eval() {
        ++eval_;
    }

    // True when `values` more floats still fit the budget.
    bool fits(uint64_t values) const {
        return bytes_ + values * sizeof(float) <= spec_.max_bytes;
    }

    bool write(const std::string& name, const std::string& op, const std::string& type,
               const int64_t ne[4], const float* data, uint64_t values) {
        if (!open_manifest())
            return false;
        char file[32];
        std::snprintf(file, sizeof(file), "%06d.f32", seq_);
        const std::string path = dir_ + "/" + file;
        std::FILE* f = std::fopen(path.c_str(), "wb");
        if (!f)
            return fail();
        const bool ok = std::fwrite(data, sizeof(float), values, f) == values;
        if (std::fclose(f) != 0 || !ok)
            return fail();
        bytes_ += values * sizeof(float);
        std::fprintf(manifest_,
                     "{\"seq\":%d,\"eval\":%d,\"name\":\"%s\",\"op\":\"%s\",\"type\":\"%s\","
                     "\"ne\":[%lld,%lld,%lld,%lld],\"mode\":\"%s\",\"values\":%llu,"
                     "\"file\":\"%s\"}\n",
                     seq_, eval_, json_escape(name).c_str(), json_escape(op).c_str(),
                     json_escape(type).c_str(), static_cast<long long>(ne[0]),
                     static_cast<long long>(ne[1]), static_cast<long long>(ne[2]),
                     static_cast<long long>(ne[3]), mode(), static_cast<unsigned long long>(values),
                     file);
        ++seq_;
        return std::fflush(manifest_) == 0 || fail();
    }

    bool skip(const std::string& name, const std::string& op, const std::string& type,
              const int64_t ne[4], const std::string& reason) {
        if (!open_manifest())
            return false;
        std::fprintf(manifest_,
                     "{\"eval\":%d,\"name\":\"%s\",\"op\":\"%s\",\"type\":\"%s\","
                     "\"ne\":[%lld,%lld,%lld,%lld],\"skipped\":\"%s\"}\n",
                     eval_, json_escape(name).c_str(), json_escape(op).c_str(),
                     json_escape(type).c_str(), static_cast<long long>(ne[0]),
                     static_cast<long long>(ne[1]), static_cast<long long>(ne[2]),
                     static_cast<long long>(ne[3]), json_escape(reason).c_str());
        ++skipped_;
        return std::fflush(manifest_) == 0 || fail();
    }

  private:
    const char* mode() const {
        return spec_.last_column_only ? "last" : "full";
    }
    bool open_manifest() {
        if (failed_)
            return false;
        if (!manifest_) {
            manifest_ = std::fopen((dir_ + "/manifest.jsonl").c_str(), "wb");
            if (!manifest_)
                return fail();
        }
        return true;
    }
    bool fail() {
        failed_ = true;
        return false;
    }

    std::string dir_;
    TensorCaptureSpec spec_;
    std::FILE* manifest_ = nullptr;
    int seq_ = 0;
    int eval_ = -1;
    int skipped_ = 0;
    uint64_t bytes_ = 0;
    bool failed_ = false;
};

} // namespace xllama
