// Copyright (c) 2024 Venere Labs
// SPDX-License-Identifier: MIT
//
// CLIPTokenizer — a header-only, byte-level BPE tokenizer matching the
// transformers 4.46.3 CLIPTokenizer used by SD-Turbo's text encoder. Reproduces:
//   lowercase + whitespace-clean -> regex pre-tokenization -> byte encoding ->
//   ranked BPE merges (+ </w> end-of-word) -> vocab ids -> [bos ... eos] padded
//   to max_length (77).
// Unit-tested on the host (tests/test_diffusion.cpp) against golden token ids
// captured from the Python reference (diffusion/gen_golden_vectors.py). This is
// the correctness gate before it ships in the console pipeline (uwp/diffuse.cpp).
//
// KNOWN LIMITATION: pre-tokenization approximates the CLIP regex's Unicode
// character classes (\p{L}, \p{N}) — ASCII letters/digits are exact; any
// codepoint >= 0x80 is treated as a letter (covers Latin-accented prompts such
// as "café", misclassifies non-letter high codepoints like emoji/symbols).
// std::regex lacks Unicode property support; a full ICU-class classifier is out
// of scope for the console demo, whose prompts are English.
#pragma once

#include <cstdint>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace xllama::diffusion {

class ClipTokenizer {
  public:
    static constexpr int kBos = 49406;
    static constexpr int kEos = 49407;
    static constexpr int kMaxLength = 77;

    // Load from a HF tokenizer directory's vocab.json + merges.txt.
    static ClipTokenizer from_files(const std::string& vocab_json, const std::string& merges_txt) {
        ClipTokenizer t;
        t.init_byte_encoder();
        // vocab.json: flat { "<byte-encoded token>": id }. Parsed with a minimal
        // scanner (no JSON lib) so this header stays dependency-free for the UWP
        // build, which does not check out the llama.cpp submodule (nlohmann).
        {
            std::ifstream vf(vocab_json, std::ios::binary);
            std::string s((std::istreambuf_iterator<char>(vf)), std::istreambuf_iterator<char>());
            parse_flat_string_int(s, t.vocab_);
        }
        // merges.txt: "#version" header then one "a b" pair per line; rank = order.
        std::ifstream mf(merges_txt);
        std::string line;
        int rank = 0;
        while (std::getline(mf, line)) {
            if (line.empty() || line[0] == '#')
                continue;
            auto sp = line.find(' ');
            if (sp == std::string::npos)
                continue;
            t.bpe_ranks_[{line.substr(0, sp), line.substr(sp + 1)}] = rank++;
        }
        return t;
    }

    // Encode to exactly kMaxLength ids: [bos] + tokens + [eos], pad with pad id.
    std::vector<int> encode(const std::string& text) const {
        std::vector<int> ids;
        ids.push_back(kBos);
        for (const std::string& word : pretokenize(clean(text))) {
            for (const std::string& tok : bpe(byte_encode(word))) {
                auto it = vocab_.find(tok);
                if (it != vocab_.end())
                    ids.push_back(it->second);
            }
        }
        // Truncate keeping room for eos, then append eos, then pad.
        if ((int)ids.size() >= kMaxLength)
            ids.resize(kMaxLength - 1);
        ids.push_back(kEos);
        const int pad_id = pad_id_();
        while ((int)ids.size() < kMaxLength)
            ids.push_back(pad_id);
        return ids;
    }

  private:
    using Pair = std::pair<std::string, std::string>;
    struct PairHash {
        size_t operator()(const Pair& p) const {
            return std::hash<std::string>()(p.first) * 1000003u ^
                   std::hash<std::string>()(p.second);
        }
    };

    std::unordered_map<std::string, int> vocab_;
    std::unordered_map<Pair, int, PairHash> bpe_ranks_;
    std::string byte_encoder_[256]; // byte -> UTF-8 of mapped codepoint

    int pad_id_() const {
        auto it = vocab_.find("!"); // pad_token = "!" (id 0) per tokenizer_config
        return it != vocab_.end() ? it->second : 0;
    }

    // Minimal parser for a flat JSON object of "string": integer pairs (CLIP
    // vocab.json). Handles the escapes that occur in byte-BPE keys (\", \\, \/,
    // \b\f\n\r\t, \uXXXX). Not a general JSON parser — only this one shape.
    static void parse_flat_string_int(const std::string& s,
                                      std::unordered_map<std::string, int>& out) {
        size_t i = 0;
        const size_t n = s.size();
        auto skip_ws = [&]() {
            while (i < n && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r'))
                ++i;
        };
        auto parse_string = [&](std::string& key) -> bool {
            if (i >= n || s[i] != '"')
                return false;
            ++i;
            while (i < n && s[i] != '"') {
                char c = s[i++];
                if (c != '\\') {
                    key.push_back(c);
                    continue;
                }
                if (i >= n)
                    return false;
                char e = s[i++];
                switch (e) {
                case '"':
                    key.push_back('"');
                    break;
                case '\\':
                    key.push_back('\\');
                    break;
                case '/':
                    key.push_back('/');
                    break;
                case 'b':
                    key.push_back('\b');
                    break;
                case 'f':
                    key.push_back('\f');
                    break;
                case 'n':
                    key.push_back('\n');
                    break;
                case 'r':
                    key.push_back('\r');
                    break;
                case 't':
                    key.push_back('\t');
                    break;
                case 'u': {
                    if (i + 4 > n)
                        return false;
                    uint32_t cp = 0;
                    for (int k = 0; k < 4; ++k) {
                        char h = s[i++];
                        cp <<= 4;
                        if (h >= '0' && h <= '9')
                            cp |= (h - '0');
                        else if (h >= 'a' && h <= 'f')
                            cp |= (h - 'a' + 10);
                        else if (h >= 'A' && h <= 'F')
                            cp |= (h - 'A' + 10);
                    }
                    append_utf8(key, cp);
                    break;
                }
                default:
                    key.push_back(e);
                    break;
                }
            }
            if (i >= n)
                return false;
            ++i; // closing quote
            return true;
        };
        skip_ws();
        if (i < n && s[i] == '{')
            ++i;
        while (i < n) {
            skip_ws();
            if (i < n && s[i] == '}')
                break;
            std::string key;
            if (!parse_string(key))
                break;
            skip_ws();
            if (i < n && s[i] == ':')
                ++i;
            skip_ws();
            bool neg = false;
            if (i < n && (s[i] == '-' || s[i] == '+'))
                neg = (s[i++] == '-');
            long val = 0;
            bool any = false;
            while (i < n && s[i] >= '0' && s[i] <= '9') {
                val = val * 10 + (s[i++] - '0');
                any = true;
            }
            if (any)
                out[key] = (int)(neg ? -val : val);
            skip_ws();
            if (i < n && s[i] == ',')
                ++i;
        }
    }

    static void append_utf8(std::string& out, uint32_t cp) {
        if (cp < 0x80) {
            out.push_back((char)cp);
        } else if (cp < 0x800) {
            out.push_back((char)(0xC0 | (cp >> 6)));
            out.push_back((char)(0x80 | (cp & 0x3F)));
        } else if (cp < 0x10000) {
            out.push_back((char)(0xE0 | (cp >> 12)));
            out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back((char)(0x80 | (cp & 0x3F)));
        } else {
            out.push_back((char)(0xF0 | (cp >> 18)));
            out.push_back((char)(0x80 | ((cp >> 12) & 0x3F)));
            out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back((char)(0x80 | (cp & 0x3F)));
        }
    }

    // GPT-2/CLIP bytes_to_unicode: printable ASCII + Latin-1 ranges map to
    // themselves; the remaining bytes map to codepoints 256+.
    void init_byte_encoder() {
        std::vector<int> mapped(256, -1);
        auto keep = [&](int lo, int hi) {
            for (int b = lo; b <= hi; ++b)
                mapped[b] = b;
        };
        keep('!', '~');   // 33..126
        keep(0xA1, 0xAC); // ¡..¬
        keep(0xAE, 0xFF); // ®..ÿ
        int n = 0;
        for (int b = 0; b < 256; ++b)
            if (mapped[b] < 0)
                mapped[b] = 256 + n++;
        for (int b = 0; b < 256; ++b) {
            std::string s;
            append_utf8(s, (uint32_t)mapped[b]);
            byte_encoder_[b] = s;
        }
    }

    // Decode UTF-8 to codepoints (for pre-tokenization classification).
    static std::vector<uint32_t> to_codepoints(const std::string& s) {
        std::vector<uint32_t> cps;
        size_t i = 0;
        while (i < s.size()) {
            uint8_t c = (uint8_t)s[i];
            uint32_t cp;
            int len;
            if (c < 0x80) {
                cp = c;
                len = 1;
            } else if ((c >> 5) == 0x6) {
                cp = c & 0x1F;
                len = 2;
            } else if ((c >> 4) == 0xE) {
                cp = c & 0x0F;
                len = 3;
            } else if ((c >> 3) == 0x1E) {
                cp = c & 0x07;
                len = 4;
            } else {
                cp = c;
                len = 1;
            }
            for (int k = 1; k < len && i + k < s.size(); ++k)
                cp = (cp << 6) | ((uint8_t)s[i + k] & 0x3F);
            cps.push_back(cp);
            i += len;
        }
        return cps;
    }

    static bool is_space(uint32_t c) {
        return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == 0x0B;
    }
    static bool is_ascii_digit(uint32_t c) {
        return c >= '0' && c <= '9';
    }
    // \p{L} approximation: ASCII letters (post-lowercase => a..z, A..Z guard) or
    // any non-ASCII codepoint. See KNOWN LIMITATION above.
    static bool is_letter(uint32_t c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c >= 0x80;
    }

    // whitespace_clean + ASCII lowercase. Collapses whitespace runs to one space
    // and strips ends (transformers CLIP does this before tokenizing).
    static std::string clean(const std::string& text) {
        auto cps = to_codepoints(text);
        std::string out;
        bool prev_space = false;
        for (uint32_t c : cps) {
            if (is_space(c)) {
                prev_space = true;
                continue;
            }
            if (prev_space && !out.empty())
                out.push_back(' ');
            prev_space = false;
            if (c >= 'A' && c <= 'Z')
                c += 32; // ASCII lowercase
            append_utf8(out, c);
        }
        return out;
    }

    // Split cleaned text into words, approximating the CLIP regex:
    //   's|'t|'re|'ve|'m|'ll|'d | letters+ | one digit | other+ (whitespace splits)
    static std::vector<std::string> pretokenize(const std::string& text) {
        auto cps = to_codepoints(text);
        std::vector<std::string> words;
        const size_t N = cps.size();
        size_t i = 0;
        auto emit = [&](size_t lo, size_t hi) {
            std::string w;
            for (size_t k = lo; k < hi; ++k)
                append_utf8(w, cps[k]);
            words.push_back(std::move(w));
        };
        while (i < N) {
            uint32_t c = cps[i];
            if (is_space(c)) {
                ++i;
                continue;
            }
            // Contractions: 's 't 're 've 'm 'll 'd
            if (c == '\'') {
                auto at = [&](size_t k) -> uint32_t { return k < N ? cps[k] : 0; };
                uint32_t a = at(i + 1), b = at(i + 2);
                if (a == 's' || a == 't' || a == 'm' || a == 'd') {
                    emit(i, i + 2);
                    i += 2;
                    continue;
                }
                if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) {
                    emit(i, i + 3);
                    i += 3;
                    continue;
                }
                emit(i, i + 1);
                ++i;
                continue; // lone apostrophe -> other
            }
            if (is_letter(c)) {
                size_t j = i + 1;
                while (j < N && is_letter(cps[j]))
                    ++j;
                emit(i, j);
                i = j;
                continue;
            }
            if (is_ascii_digit(c)) {
                emit(i, i + 1);
                ++i;
                continue;
            } // one digit
            // other: run of non-space, non-letter, non-digit, non-apostrophe
            size_t j = i + 1;
            while (j < N && !is_space(cps[j]) && !is_letter(cps[j]) && !is_ascii_digit(cps[j]) &&
                   cps[j] != '\'')
                ++j;
            emit(i, j);
            i = j;
        }
        return words;
    }

    // Byte-encode a word: each UTF-8 byte -> its mapped-codepoint string.
    std::string byte_encode(const std::string& word) const {
        std::string out;
        for (unsigned char b : word)
            out += byte_encoder_[b];
        return out;
    }

    // Split a byte-encoded string into its initial BPE symbols (one per mapped
    // codepoint), appending </w> to the last symbol.
    static std::vector<std::string> initial_symbols(const std::string& s) {
        std::vector<std::string> sym;
        size_t i = 0;
        while (i < s.size()) {
            uint8_t c = (uint8_t)s[i];
            int len = c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xE ? 3 : 4;
            sym.push_back(s.substr(i, len));
            i += len;
        }
        if (!sym.empty())
            sym.back() += "</w>";
        return sym;
    }

    // BPE merge loop over a byte-encoded word -> final subword tokens.
    std::vector<std::string> bpe(const std::string& token) const {
        std::vector<std::string> word = initial_symbols(token);
        if (word.size() <= 1)
            return word;
        while (true) {
            int best_rank = std::numeric_limits<int>::max();
            size_t best_i = 0;
            bool found = false;
            for (size_t k = 0; k + 1 < word.size(); ++k) {
                auto it = bpe_ranks_.find({word[k], word[k + 1]});
                if (it != bpe_ranks_.end() && it->second < best_rank) {
                    best_rank = it->second;
                    best_i = k;
                    found = true;
                }
            }
            if (!found)
                break;
            std::vector<std::string> merged;
            merged.reserve(word.size());
            for (size_t k = 0; k < word.size();) {
                if (k == best_i && k + 1 < word.size()) {
                    merged.push_back(word[k] + word[k + 1]);
                    k += 2;
                } else {
                    merged.push_back(word[k]);
                    k += 1;
                }
            }
            word.swap(merged);
            if (word.size() == 1)
                break;
        }
        return word;
    }
};

} // namespace xllama::diffusion
