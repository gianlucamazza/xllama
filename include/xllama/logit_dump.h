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

#pragma once

#include <cstdint>
#include <cstdio>
#include <string>

namespace xllama {

// Minimal JSON string escaping (quotes, backslash, control chars). Prompts and
// detokenized pieces can contain newlines/quotes, so the sidecar must escape.
inline std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) {
        case '"':
            out += "\\\"";
            break;
        case '\\':
            out += "\\\\";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\r':
            out += "\\r";
            break;
        case '\t':
            out += "\\t";
            break;
        default:
            if (c < 0x20) {
                char buf[8];
                std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                out += buf;
            } else {
                out += static_cast<char>(c);
            }
        }
    }
    return out;
}

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

} // namespace xllama
