// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
#include "floppylm_internal.h"
#include "ggml.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <set>
#include <stdexcept>
namespace xllama::floppy {
static void require(bool v, const char* s) {
    if (!v)
        throw std::runtime_error(s);
}
Json parse_json(const std::string& text) {
    std::vector<std::set<std::string>> objects;
    return Json::parse(text, [&](int, Json::parse_event_t event, Json& value) {
        if (event == Json::parse_event_t::object_start)
            objects.emplace_back();
        else if (event == Json::parse_event_t::object_end)
            objects.pop_back();
        else if (event == Json::parse_event_t::key)
            require(objects.back().insert(value.get<std::string>()).second, "duplicate JSON key");
        return true;
    });
}
Config::Config(const Json& j) {
    require(j.is_object(), "model config must be an object");
    const std::vector<std::string> keys = {"vocab", "d",      "n_layers", "n_heads", "d_ff",
                                           "ctx",   "mlp",    "core_fmt", "emb_fmt", "scale_policy",
                                           "delta", "qk_norm"};
    require(j.size() == keys.size(), "model config has missing or unknown fields");
    for (const auto& k : keys)
        require(j.contains(k), "missing model config field");
    for (const auto* key : {"vocab", "d", "n_layers", "n_heads", "d_ff", "ctx"})
        require(j.at(key).is_number_integer(), "shape must be integer");
    require(j.at("qk_norm").is_boolean() && j.at("delta").is_number(), "invalid config types");
    vocab = j.at("vocab");
    d = j.at("d");
    layers = j.at("n_layers");
    heads = j.at("n_heads");
    ff = j.at("d_ff");
    ctx = j.at("ctx");
    mlp = j.at("mlp");
    core = j.at("core_fmt");
    emb = j.at("emb_fmt");
    policy = j.at("scale_policy");
    delta = j.at("delta");
    qknorm = j.at("qk_norm");
    require(vocab == 256 && d > 0 && d <= 1024 && layers > 0 && layers <= 32 && heads > 0 &&
                d % heads == 0 && (d / heads) % 2 == 0 && ff > 0 && ff <= 16 * d && ctx > 0 &&
                ctx <= 2048,
            "invalid or unsupported model shape");
    require(mlp == "gelu" || mlp == "relu2" || mlp == "swiglu", "invalid MLP");
    require((core == "ternary" || core == "2bit" || core == "4bit") && emb == "4bit",
            "invalid scalar format");
    require(policy == "row16" || policy == "row8log" || policy == "tensor16",
            "invalid scale policy");
    require(std::isfinite(delta) && delta >= 0 && delta < 4, "invalid ternary threshold");
}
Json Config::json() const {
    return {{"vocab", vocab},     {"d", d},
            {"n_layers", layers}, {"n_heads", heads},
            {"d_ff", ff},         {"ctx", ctx},
            {"mlp", mlp},         {"core_fmt", core},
            {"emb_fmt", emb},     {"scale_policy", policy},
            {"delta", delta},     {"qk_norm", qknorm}};
}
Model::Model(const Json& b) : cfg(b.at("config")) {
    uint64_t up = uint64_t(cfg.ff) * (cfg.mlp == "swiglu" ? 2 : 1);
    uint64_t parameters = uint64_t(cfg.vocab) * cfg.d +
                          uint64_t(cfg.layers) * (2 * cfg.d + 4ull * cfg.d * cfg.d + up * cfg.d +
                                                  uint64_t(cfg.d) * cfg.ff) +
                          cfg.d;
    require(parameters <= 16000000, "model exceeds native 16M parameter limit");
    auto add = [&](std::string name, int r, int c, bool norm = false) {
        Tensor t{name, r, c, norm, {}, {}, {}};
        t.w.resize(size_t(r) * c);
        t.m.resize(t.w.size());
        t.v.resize(t.w.size());
        tensors.push_back(std::move(t));
    };
    add("emb.weight", cfg.vocab, cfg.d);
    for (int i = 0; i < cfg.layers; ++i) {
        std::string p = "blocks." + std::to_string(i) + ".";
        add(p + "norm1.weight", 1, cfg.d, true);
        add(p + "qkv.weight", 3 * cfg.d, cfg.d);
        add(p + "proj.weight", cfg.d, cfg.d);
        add(p + "norm2.weight", 1, cfg.d, true);
        add(p + "fc.weight", (cfg.mlp == "swiglu" ? 2 : 1) * cfg.ff, cfg.d);
        add(p + "fc2.weight", cfg.d, cfg.ff);
    }
    add("norm.weight", 1, cfg.d, true);
    if (b.contains("weights")) {
        const auto& ws = b.at("weights");
        require(ws.size() == tensors.size(), "wrong weight count");
        for (auto& t : tensors) {
            auto w = ws.at(t.name).get<std::vector<float>>();
            require(w.size() == t.w.size(), "wrong weight shape");
            for (float x : w)
                require(std::isfinite(x), "nonfinite initial weight");
            t.w = std::move(w);
        }
    }
}
Json Model::state() const {
    require(canonical.empty(), "FLP2 cannot become a training checkpoint");
    Json j = {{"config", cfg.json()}, {"weights", Json::object()}, {"moments", Json::object()}};
    for (const auto& t : tensors) {
        j["weights"][t.name] = t.w;
        j["moments"][t.name] = {{"m", t.m}, {"v", t.v}};
    }
    return j;
}
void Model::restore(const Json& j) {
    require(canonical.empty(), "FLP2 is inference-only");
    require(j.at("config") == cfg.json(), "checkpoint config mismatch");
    for (auto& t : tensors) {
        auto w = j.at("weights").at(t.name).get<std::vector<float>>();
        auto m = j.at("moments").at(t.name).at("m").get<std::vector<float>>();
        auto v = j.at("moments").at(t.name).at("v").get<std::vector<float>>();
        require(w.size() == t.w.size() && m.size() == w.size() && v.size() == w.size(),
                "checkpoint shape mismatch");
        for (auto x : w)
            require(std::isfinite(x), "nonfinite checkpoint");
        for (auto x : m)
            require(std::isfinite(x), "nonfinite moment");
        for (auto x : v)
            require(std::isfinite(x) && x >= 0, "invalid second moment");
        t.w = std::move(w);
        t.m = std::move(m);
        t.v = std::move(v);
    }
}
float half_round(float x) {
    return ggml_fp16_to_fp32(ggml_fp32_to_fp16(x));
}
static void u16(Bytes& b, uint16_t x) {
    b.push_back(uint8_t(x));
    b.push_back(uint8_t(x >> 8));
}
static void u32(Bytes& b, uint32_t x) {
    for (int i = 0; i < 4; ++i)
        b.push_back(uint8_t(x >> (8 * i)));
}
static float stored_scale(float x) {
    require(std::isfinite(x) && x >= 0 && x <= 65504, "invalid fp16 scale");
    return half_round(x > 0 ? std::max(x, 0.00006103515625f) : 0);
}
// Reduction structure adapted from PyTorch v2.11.0 SumKernel.cpp (BSD-3-Clause).
// Attribution and license: training/floppylm/PYTORCH-LICENSE.
// Reproduce the pinned PyTorch CPU sum reduction order for contiguous fp32 inputs.
// Fixed logical lanes make the codec independent of compiler SIMD dispatch.
static float reference_sum(const std::vector<float>& x) {
    constexpr size_t lanes = 8;
    if (x.size() < lanes) {
        float partial[4] = {};
        size_t i = 0;
        for (; i + 4 <= x.size(); i += 4)
            for (size_t k = 0; k < 4; ++k)
                partial[k] += x[i + k];
        for (; i < x.size(); ++i)
            partial[0] += x[i];
        for (size_t k = 1; k < 4; ++k)
            partial[0] += partial[k];
        return partial[0];
    }
    size_t vectors = x.size() / lanes, groups = vectors / 4;
    size_t power = 4;
    size_t ceil_log = 0;
    for (size_t n = groups ? groups - 1 : 0; n; n >>= 1)
        ++ceil_log;
    power = std::max(power, ceil_log / 4);
    size_t chunk = size_t(1) << power, mask = chunk - 1;
    float a[4][4][lanes] = {};
    size_t i = 0;
    for (; i + chunk <= groups;) {
        for (size_t j = 0; j < chunk; ++j, ++i)
            for (size_t k = 0; k < 4; ++k)
                for (size_t lane = 0; lane < lanes; ++lane)
                    a[0][k][lane] += x[(i * 4 + k) * lanes + lane];
        for (size_t level = 1; level < 4; ++level) {
            for (size_t k = 0; k < 4; ++k)
                for (size_t lane = 0; lane < lanes; ++lane) {
                    a[level][k][lane] += a[level - 1][k][lane];
                    a[level - 1][k][lane] = 0;
                }
            if (i & (mask << (level * power)))
                break;
        }
    }
    for (; i < groups; ++i)
        for (size_t k = 0; k < 4; ++k)
            for (size_t lane = 0; lane < lanes; ++lane)
                a[0][k][lane] += x[(i * 4 + k) * lanes + lane];
    for (size_t level = 1; level < 4; ++level)
        for (size_t k = 0; k < 4; ++k)
            for (size_t lane = 0; lane < lanes; ++lane)
                a[0][k][lane] += a[level][k][lane];
    for (size_t v = groups * 4; v < vectors; ++v)
        for (size_t lane = 0; lane < lanes; ++lane)
            a[0][0][lane] += x[v * lanes + lane];
    for (size_t k = 1; k < 4; ++k)
        for (size_t lane = 0; lane < lanes; ++lane)
            a[0][0][lane] += a[0][k][lane];
    float sum = 0;
    for (size_t tail = vectors * lanes; tail < x.size(); ++tail)
        sum += x[tail];
    for (size_t lane = 0; lane < lanes; ++lane)
        sum += a[0][0][lane];
    return sum;
}
Quant quantize(const Tensor& t, const Config& c, bool embedding) {
    std::string fmt = embedding ? c.emb : c.core;
    int levels = fmt == "ternary" ? 3 : fmt == "2bit" ? 4 : 16;
    Quant q;
    q.levels = levels;
    q.values.resize(t.w.size());
    q.symbols.resize(t.w.size());
    std::vector<float> mean(t.rows), scale(t.rows), absolute(t.w.size());
    for (size_t i = 0; i < t.w.size(); ++i) {
        require(std::isfinite(t.w[i]), "nonfinite weight");
        absolute[i] = std::abs(t.w[i]);
    }
    float global = reference_sum(absolute);
    for (int r = 0; r < t.rows; ++r) {
        std::vector<float> row(absolute.begin() + size_t(r) * t.cols,
                               absolute.begin() + size_t(r + 1) * t.cols);
        mean[r] = reference_sum(row) / t.cols;
    }
    if (c.policy == "tensor16")
        std::fill(mean.begin(), mean.end(), global / float(t.w.size()));
    float kept_global = 0;
    size_t count_global = 0;
    std::vector<float> row_kept(t.rows);
    for (int r = 0; r < t.rows; ++r) {
        if (levels == 3) {
            float kept = 0;
            int count = 0;
            std::vector<float> retained(t.cols);
            for (int i = 0; i < t.cols; ++i) {
                float a = std::abs(t.w[size_t(r) * t.cols + i]);
                if (a > float(c.delta) * mean[r]) {
                    retained[i] = a;
                    ++count;
                }
            }
            kept = reference_sum(retained);
            row_kept[r] = kept;
            scale[r] = count ? kept / count : 0;
            kept_global += kept;
            count_global += count;
        } else
            scale[r] = (levels == 4 ? 1.135f : 0.42f) * mean[r];
    }
    kept_global = reference_sum(row_kept);
    if (levels == 3 && c.policy == "tensor16")
        std::fill(scale.begin(), scale.end(), count_global ? kept_global / count_global : 0);
    if (c.policy == "row8log") {
        float base = stored_scale(*std::max_element(scale.begin(), scale.end()));
        u16(q.scales, ggml_fp32_to_fp16(base));
        for (auto& s : scale) {
            int code = 0;
            if (s > 0 && base > 0)
                code = std::clamp(
                    int(std::nearbyint(std::log2(std::max(s, 1e-30f) / base) * 16) + 128), 1, 255);
            q.scales.push_back(uint8_t(code));
            s = code ? base * std::exp2(float(code - 128) / 16) : 0;
        }
    } else if (c.policy == "tensor16") {
        float s = stored_scale(scale[0]);
        u16(q.scales, ggml_fp32_to_fp16(s));
        std::fill(scale.begin(), scale.end(), s);
    } else {
        for (auto& s : scale) {
            s = stored_scale(s);
            u16(q.scales, ggml_fp32_to_fp16(s));
        }
    }
    float h = float(levels - 1) / 2;
    for (int r = 0; r < t.rows; ++r)
        for (int i = 0; i < t.cols; ++i) {
            size_t k = size_t(r) * t.cols + i;
            float w = t.w[k];
            int sym;
            if (levels == 3)
                sym = 1 + (std::abs(w) > float(c.delta) * mean[r] ? ((w > 0) - (w < 0)) : 0);
            else
                sym = scale[r] == 0 ? levels / 2
                                    : int(std::nearbyint(
                                          std::clamp(std::floor(w / scale[r]) + 0.5f, -h, h) + h));
            q.symbols[k] = uint16_t(sym);
            q.values[k] = (sym - h) * scale[r];
        }
    return q;
}
static Bytes symbols(const std::vector<uint16_t>& s, int a) {
    Bytes bp{1};
    u32(bp, uint32_t(s.size()));
    u16(bp, uint16_t(a));
    if (a == 3) {
        const int k = 40;
        for (size_t i = 0; i < s.size(); i += k) {
            uint64_t v = 0;
            for (int j = 0; j < k; ++j)
                v = v * 3 + (i + j < s.size() ? s[i + j] : 0);
            for (int j = 0; j < 8; ++j)
                bp.push_back(uint8_t(v >> (8 * j)));
        }
    } else {
        int bits = a == 4 ? 2 : 4;
        uint8_t v = 0;
        int used = 0;
        for (auto x : s) {
            v = uint8_t((v << bits) | x);
            used += bits;
            if (used == 8) {
                bp.push_back(v);
                used = 0;
                v = 0;
            }
        }
        if (used)
            bp.push_back(uint8_t(v << (8 - used)));
    }
    std::vector<uint32_t> counts(a), freq(a), cum(a);
    for (auto x : s)
        ++counts[x];
    uint32_t sum = 0;
    for (int i = 0; i < a; ++i) {
        freq[i] = uint32_t(uint64_t(counts[i]) * 32768 / s.size());
        if (counts[i] && !freq[i])
            freq[i] = 1;
        sum += freq[i];
    }
    auto biggest = std::max_element(freq.begin(), freq.end());
    *biggest += 32768 - sum;
    for (int i = 1; i < a; ++i)
        cum[i] = cum[i - 1] + freq[i - 1];
    Bytes ra{0};
    u32(ra, uint32_t(s.size()));
    u16(ra, uint16_t(a));
    for (auto f : freq)
        u16(ra, uint16_t(f));
    uint32_t state = 1 << 23;
    Bytes tail;
    for (auto it = s.rbegin(); it != s.rend(); ++it) {
        uint32_t f = freq[*it];
        while (state >= ((uint32_t(1 << 23) >> 15) << 8) * f) {
            tail.push_back(uint8_t(state));
            state >>= 8;
        }
        state = ((state / f) << 15) + (state % f) + cum[*it];
    }
    u32(tail, state);
    std::reverse(tail.begin(), tail.end());
    ra.insert(ra.end(), tail.begin(), tail.end());
    return ra.size() < bp.size() ? ra : bp;
}
void Model::verify_frozen() const {
    if (canonical.empty())
        return;
    require(cfg.json() == frozen_config, "FLP2 config was modified");
    require(snapshot.size() == tensors.size(), "invalid frozen model");
    for (size_t i = 0; i < tensors.size(); ++i)
        require(snapshot[i] == tensors[i].w,
                "FLP2 model was modified; resume a native checkpoint instead");
}
Bytes pack(const Model& m) {
    if (!m.canonical.empty()) {
        m.verify_frozen();
        return m.canonical;
    }
    std::string h = m.cfg.json().dump();
    require(h.size() < 65536, "header too long");
    Bytes b = {'F', 'L', 'P', '2'};
    u16(b, uint16_t(h.size()));
    b.insert(b.end(), h.begin(), h.end());
    u32(b, 0);
    for (size_t i = 0; i < m.tensors.size(); ++i) {
        const auto& t = m.tensors[i];
        if (t.norm) {
            for (float v : t.w) {
                require(std::isfinite(v) && std::abs(v) <= 65504, "invalid norm");
                u16(b, ggml_fp32_to_fp16(v));
            }
        } else {
            auto q = quantize(t, m.cfg, i == 0);
            auto stream = symbols(q.symbols, q.levels);
            b.insert(b.end(), q.scales.begin(), q.scales.end());
            u32(b, uint32_t(stream.size()));
            b.insert(b.end(), stream.begin(), stream.end());
        }
    }
    return b;
}
struct Reader {
    const Bytes& b;
    size_t p = 0;
    uint64_t take(int n) {
        require(n > 0 && n <= 8 && p + size_t(n) <= b.size(), "truncated FLP2");
        uint64_t v = 0;
        for (int i = 0; i < n; ++i)
            v |= uint64_t(b[p++]) << (8 * i);
        return v;
    }
};
Model unpack(const Bytes& b) {
    Reader r{b};
    require(r.take(4) == 0x32504c46, "FLP2 required");
    size_t len = r.take(2);
    require(r.p + len <= b.size(), "truncated header");
    std::string h(b.begin() + r.p, b.begin() + r.p + len);
    r.p += len;
    Json cfg = Json::parse(h);
    require(cfg.dump() == h, "noncanonical header");
    Model m(Json{{"config", cfg}});
    require(r.take(4) == 0, "unexpected shared section");
    for (size_t ti = 0; ti < m.tensors.size(); ++ti) {
        auto& t = m.tensors[ti];
        if (t.norm) {
            for (auto& w : t.w) {
                w = ggml_fp16_to_fp32(uint16_t(r.take(2)));
                require(std::isfinite(w), "nonfinite norm");
            }
            continue;
        }
        std::vector<float> scale(t.rows);
        if (m.cfg.policy == "tensor16") {
            float v = ggml_fp16_to_fp32(uint16_t(r.take(2)));
            std::fill(scale.begin(), scale.end(), v);
        } else if (m.cfg.policy == "row16") {
            for (auto& s : scale)
                s = ggml_fp16_to_fp32(uint16_t(r.take(2)));
        } else {
            float base = ggml_fp16_to_fp32(uint16_t(r.take(2)));
            require(std::isfinite(base) && base >= 0, "invalid log-scale base");
            for (auto& s : scale) {
                int code = int(r.take(1));
                s = code ? base * std::exp2(float(code - 128) / 16) : 0;
            }
        }
        for (float s : scale)
            require(std::isfinite(s) && s >= 0, "invalid scale");
        size_t bytes = r.take(4), end = r.p + bytes;
        require(end <= b.size(), "truncated symbols");
        int tag = int(r.take(1));
        size_t n = r.take(4);
        int a = int(r.take(2));
        std::string fmt = ti ? m.cfg.core : m.cfg.emb;
        require(a == (fmt == "ternary" ? 3
                      : fmt == "2bit"  ? 4
                                       : 16) &&
                    n == t.w.size(),
                "symbol shape mismatch");
        std::vector<uint16_t> sy(n);
        if (tag == 1) {
            if (a == 3) {
                for (size_t i = 0; i < n; i += 40) {
                    uint64_t v = r.take(8);
                    uint16_t row[40];
                    for (int j = 39; j >= 0; --j) {
                        row[j] = v % 3;
                        v /= 3;
                    }
                    require(v == 0, "invalid ternary word");
                    for (int j = 0; j < 40; ++j)
                        if (i + j < n)
                            sy[i + j] = row[j];
                        else
                            require(row[j] == 0, "nonzero ternary padding");
                }
            } else {
                int bits = a == 4 ? 2 : 4, left = 0;
                uint32_t v = 0;
                for (auto& s : sy) {
                    if (!left) {
                        v = uint32_t(r.take(1));
                        left = 8;
                    }
                    left -= bits;
                    s = (v >> left) & (a - 1);
                }
                require(!left || (v & ((1u << left) - 1)) == 0, "nonzero bit padding");
            }
        } else if (tag == 0) {
            std::vector<uint32_t> f(a), c(a), lookup;
            for (auto& v : f)
                v = uint32_t(r.take(2));
            for (int i = 0; i < a; ++i) {
                c[i] = uint32_t(lookup.size());
                lookup.insert(lookup.end(), f[i], uint32_t(i));
            }
            require(lookup.size() == 32768, "invalid frequencies");
            uint32_t x = 0;
            for (int i = 0; i < 4; ++i)
                x = (x << 8) | uint32_t(r.take(1));
            for (auto& s : sy) {
                uint32_t slot = x & 32767;
                s = uint16_t(lookup[slot]);
                x = f[s] * (x >> 15) + slot - c[s];
                while (x < (1 << 23))
                    x = (x << 8) | uint32_t(r.take(1));
            }
            require(x == (1 << 23), "invalid rANS state");
        } else
            throw std::runtime_error("unknown symbol coder");
        require(r.p == end, "symbol stream length mismatch");
        for (size_t k = 0; k < n; ++k)
            t.w[k] = (float(sy[k]) - float(a - 1) / 2) * scale[k / t.cols];
    }
    require(r.p == b.size(), "trailing FLP2 bytes");
    m.canonical = b;
    m.frozen_config = m.cfg.json();
    for (const auto& t : m.tensors)
        m.snapshot.push_back(t.w);
    return m;
}
Json artifact_sections(const Bytes& b) {
    Reader r{b};
    require(r.take(4) == 0x32504c46, "FLP2 required");
    size_t len = r.take(2);
    require(r.p + len <= b.size(), "truncated header");
    auto config = Json::parse(b.begin() + r.p, b.begin() + r.p + len);
    r.p += len;
    size_t shared = r.take(4);
    require(shared == 0, "unexpected shared state");
    Model shape(Json{{"config", config}});
    Json sections = {{"header", 6 + len + 4}, {"shared", shared}, {"embedding", 0}, {"core", 0},
                     {"scales", 0},           {"norms", 0}};
    for (size_t i = 0; i < shape.tensors.size(); ++i) {
        const auto& t = shape.tensors[i];
        if (t.norm) {
            size_t n = t.w.size() * 2;
            sections["norms"] = sections["norms"].get<size_t>() + n;
            r.p += n;
        } else {
            size_t scale = shape.cfg.policy == "tensor16" ? 2
                           : shape.cfg.policy == "row16"  ? 2 * t.rows
                                                          : 2 + t.rows;
            sections["scales"] = sections["scales"].get<size_t>() + scale;
            r.p += scale;
            size_t n = r.take(4);
            auto key = i == 0 ? "embedding" : "core";
            sections[key] = sections[key].get<size_t>() + 4 + n;
            r.p += n;
        }
        require(r.p <= b.size(), "truncated section");
    }
    require(r.p == b.size(), "invalid section accounting");
    return sections;
}
Bytes read_bytes(const std::string& p) {
    require(std::filesystem::file_size(std::filesystem::u8path(p)) <= 512ull * 1024 * 1024,
            "input exceeds 512 MiB limit");
    std::ifstream f(std::filesystem::u8path(p), std::ios::binary);
    require(bool(f), "cannot read input file");
    return Bytes(std::istreambuf_iterator<char>(f), {});
}
void write_bytes(const std::string& p, const Bytes& b) {
    std::ofstream f(std::filesystem::u8path(p), std::ios::binary | std::ios::trunc);
    require(bool(f), "cannot create output file");
    f.write(reinterpret_cast<const char*>(b.data()), std::streamsize(b.size()));
    require(bool(f), "output write failed");
}
} // namespace xllama::floppy
