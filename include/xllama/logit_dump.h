// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// Logit-parity harness: portable dump of a single token's logit vector plus a
// JSON metadata sidecar. Shared by both text backends (llama.cpp and ORT) so
// their dumps are byte-compatible and can be diffed by scripts/compare-logits.py.
//
// Format:
//   <path>        raw float32 little-endian, `vocab_size` values (last token)
//   <path>.json   { model, prompt, backend, vocab_size, greedy, top1_id, top1_piece }
//
// float32 raw matches numpy `np.fromfile(dtype=np.float32)`.
// json_escape is in json_utils.h — included above.

#pragma once

#include <cstdint>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

#include "xllama/json_utils.h"

namespace xllama {

// json_escape is in json_utils.h — included above.

// Write the logit vector and its metadata sidecar. `top1_piece` is the detokenized
// argmax token (backend-specific), used to catch tokenizer/vocab misalignment when
// two dumps are compared. Returns false on any I/O error.
inline bool write_logit_dump(const std::string& bin_path, const float* logits, int vocab_size,
                             const std::string& model, const std::string& prompt,
                             const std::string& backend, bool greedy,
                             const std::string& top1_piece) {
    if (!logits || vocab_size <= 0)
        return false;

    // argmax over the vocabulary — the deterministic next-token prediction.
    int top1_id = 0;
    float top1_val = logits[0];
    for (int i = 1; i < vocab_size; ++i) {
        if (logits[i] > top1_val) {
            top1_val = logits[i];
            top1_id = i;
        }
    }

    std::FILE* f = std::fopen(bin_path.c_str(), "wb");
    if (!f)
        return false;
    const size_t n = static_cast<size_t>(vocab_size);
    const bool ok = std::fwrite(logits, sizeof(float), n, f) == n;
    std::fclose(f);
    if (!ok)
        return false;

    const std::string json_path = bin_path + ".json";
    std::FILE* jf = std::fopen(json_path.c_str(), "wb");
    if (!jf)
        return false;
    std::fprintf(jf,
                 "{\n"
                 "  \"model\": \"%s\",\n"
                 "  \"prompt\": \"%s\",\n"
                 "  \"backend\": \"%s\",\n"
                 "  \"vocab_size\": %d,\n"
                 "  \"greedy\": %s,\n"
                 "  \"top1_id\": %d,\n"
                 "  \"top1_piece\": \"%s\"\n"
                 "}\n",
                 json_escape(model).c_str(), json_escape(prompt).c_str(),
                 json_escape(backend).c_str(), vocab_size, greedy ? "true" : "false", top1_id,
                 json_escape(top1_piece).c_str());
    std::fclose(jf);
    return true;
}

// Per-position dump for ADR 0010 part 2: the logits of several prompt
// positions under teacher forcing, so two arms can be compared as
// distributions over many tokens instead of on the last token only.
//
//   <bin_path>.positions.bin    n_positions rows of vocab_size float32
//   <bin_path>.positions.json   { model, prompt, backend, vocab_size, stride,
//                                 n_prompt_tokens, positions:[...] }
//
// Position p holds the logits produced after token p, i.e. the prediction of
// token p + 1. `logit_dump_keeps` selects every `stride`-th position and
// always the last one.
inline bool logit_dump_keeps(int position, int n_tokens, int stride) {
    if (stride <= 0 || position < 0 || position >= n_tokens)
        return false;
    return (position + 1) % stride == 0 || position == n_tokens - 1;
}

class LogitPositionsWriter {
  public:
    LogitPositionsWriter(std::string bin_path, int vocab_size)
        : base_(std::move(bin_path)), vocab_(vocab_size) {}
    ~LogitPositionsWriter() {
        if (f_)
            std::fclose(f_);
    }
    LogitPositionsWriter(const LogitPositionsWriter&) = delete;
    LogitPositionsWriter& operator=(const LogitPositionsWriter&) = delete;

    bool failed() const {
        return failed_;
    }
    size_t positions() const {
        return positions_.size();
    }

    bool append(int position, const float* logits) {
        if (failed_ || !logits || vocab_ <= 0)
            return fail();
        if (!f_) {
            f_ = std::fopen((base_ + ".positions.bin").c_str(), "wb");
            if (!f_)
                return fail();
        }
        const size_t n = static_cast<size_t>(vocab_);
        if (std::fwrite(logits, sizeof(float), n, f_) != n)
            return fail();
        positions_.push_back(position);
        return true;
    }

    // Closes the data file and writes the sidecar. False on any write error.
    bool finish(const std::string& model, const std::string& prompt, const std::string& backend,
                int stride, int n_prompt_tokens) {
        if (f_) {
            const bool closed = std::fclose(f_) == 0;
            f_ = nullptr;
            if (!closed)
                return fail();
        }
        if (failed_ || positions_.empty())
            return fail();
        std::FILE* jf = std::fopen((base_ + ".positions.json").c_str(), "wb");
        if (!jf)
            return fail();
        std::fprintf(jf,
                     "{\n  \"model\": \"%s\",\n  \"prompt\": \"%s\",\n"
                     "  \"backend\": \"%s\",\n  \"vocab_size\": %d,\n  \"stride\": %d,\n"
                     "  \"n_prompt_tokens\": %d,\n  \"positions\": [",
                     json_escape(model).c_str(), json_escape(prompt).c_str(),
                     json_escape(backend).c_str(), vocab_, stride, n_prompt_tokens);
        for (size_t i = 0; i < positions_.size(); ++i)
            std::fprintf(jf, "%s%d", i ? ", " : "", positions_[i]);
        std::fprintf(jf, "]\n}\n");
        return std::fclose(jf) == 0 || fail();
    }

  private:
    bool fail() {
        failed_ = true;
        return false;
    }
    std::string base_;
    int vocab_ = 0;
    std::FILE* f_ = nullptr;
    std::vector<int> positions_;
    bool failed_ = false;
};

} // namespace xllama
