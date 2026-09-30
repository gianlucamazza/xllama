// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
#include "floppylm_internal.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
namespace xllama::floppy {
struct Engine::Impl {
    Config cfg;
    int batch, length, threads;
    bool backward;
    ggml_context* ctx = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_cgraph* graph = nullptr;
    ggml_tensor *ids = nullptr, *positions = nullptr, *labels = nullptr, *output = nullptr,
                *loss = nullptr;
    std::vector<ggml_tensor*> weights;
    Impl(const Config& c, int b, int t, int nt, bool back)
        : cfg(c), batch(b), length(t), threads(nt), backward(back) {
        if (b < 1 || b > 32 || t < 1 || t > c.ctx || nt < 1 || nt > 16)
            throw std::runtime_error("invalid execution shape");
        try {
            ggml_init_params ip{32 * 1024 * 1024, nullptr, true};
            ctx = ggml_init(ip);
            if (!ctx)
                throw std::runtime_error("ggml metadata allocation failed");
            backend = ggml_backend_cpu_init();
            if (!backend)
                throw std::runtime_error("CPU backend unavailable");
            ggml_backend_cpu_set_n_threads(backend, nt);
            Model shape(Json{{"config", c.json()}});
            for (auto& w : shape.tensors) {
                auto* p = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, w.cols, w.rows);
                ggml_set_name(p, w.name.c_str());
                if (back)
                    ggml_set_param(p);
                weights.push_back(p);
            }
            ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, t * b);
            positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, t);
            labels = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.vocab, t * b);
            auto* x = ggml_reshape_3d(ctx, ggml_get_rows(ctx, weights[0], ids), c.d, t, b);
            size_t wi = 1;
            int hd = c.d / c.heads;
            auto norm = [&](ggml_tensor* z, ggml_tensor* w) {
                return ggml_mul(ctx, ggml_rms_norm(ctx, z, 1.1920928955078125e-7f), w);
            };
            for (int l = 0; l < c.layers; ++l) {
                auto* n1 = weights[wi++];
                auto* wq = weights[wi++];
                auto* wp = weights[wi++];
                auto* n2 = weights[wi++];
                auto* wf = weights[wi++];
                auto* wf2 = weights[wi++];
                auto* qkv = ggml_mul_mat(ctx, wq, norm(x, n1));
                auto slice = [&](int k) {
                    auto* z = ggml_view_3d(ctx, qkv, c.d, t, b, qkv->nb[1], qkv->nb[2],
                                           size_t(k * c.d) * sizeof(float));
                    return ggml_reshape_4d(ctx, ggml_cont(ctx, z), hd, c.heads, t, b);
                };
                auto* q = slice(0);
                auto* k = slice(1);
                auto* v = slice(2);
                if (c.qknorm) {
                    q = ggml_rms_norm(ctx, q, 1.1920928955078125e-7f);
                    k = ggml_rms_norm(ctx, k, 1.1920928955078125e-7f);
                }
                q = ggml_rope(ctx, q, positions, hd, 0);
                k = ggml_rope(ctx, k, positions, hd, 0);
                q = ggml_permute(ctx, q, 0, 2, 1, 3);
                k = ggml_permute(ctx, k, 0, 2, 1, 3);
                v = ggml_permute(ctx, v, 0, 2, 1, 3);
                auto* s = ggml_scale(ctx, ggml_mul_mat(ctx, k, q), 1 / std::sqrt(float(hd)));
                auto* a = ggml_soft_max(ctx, ggml_diag_mask_inf(ctx, s, 0));
                auto* av = ggml_mul_mat(ctx, ggml_cont(ctx, ggml_transpose(ctx, v)), a);
                av = ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_permute(ctx, av, 0, 2, 1, 3)), c.d, t,
                                     b);
                x = ggml_add(ctx, x, ggml_mul_mat(ctx, wp, av));
                auto* h = ggml_mul_mat(ctx, wf, norm(x, n2));
                if (c.mlp == "gelu")
                    h = ggml_gelu_erf(ctx, h);
                else if (c.mlp == "relu2")
                    h = ggml_sqr(ctx, ggml_relu(ctx, h));
                else {
                    auto* left =
                        ggml_cont(ctx, ggml_view_3d(ctx, h, c.ff, t, b, h->nb[1], h->nb[2], 0));
                    auto* right = ggml_cont(ctx, ggml_view_3d(ctx, h, c.ff, t, b, h->nb[1],
                                                              h->nb[2], c.ff * sizeof(float)));
                    h = ggml_mul(ctx, ggml_silu(ctx, left), right);
                }
                x = ggml_add(ctx, x, ggml_mul_mat(ctx, wf2, h));
            }
            output = ggml_reshape_2d(ctx, ggml_mul_mat(ctx, weights[0], norm(x, weights[wi])),
                                     c.vocab, t * b);
            loss = ggml_cross_entropy_loss(ctx, output, labels);
            ggml_set_loss(loss);
            graph = ggml_new_graph_custom(ctx, 16384, back);
            ggml_build_forward_expand(graph, loss);
            if (back) {
                std::vector<ggml_tensor*> acc(ggml_graph_n_nodes(graph), nullptr);
                for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
                    auto* n = ggml_graph_node(graph, i);
                    if (n->flags & GGML_TENSOR_FLAG_LOSS)
                        acc[i] = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, n->ne);
                }
                ggml_build_backward_expand(ctx, graph, acc.data());
            }
            size_t bytes = 0;
            for (auto* t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
                if (!t->view_src)
                    bytes += ggml_nbytes(t) + 64;
            }
            if (bytes > 512ull * 1024 * 1024)
                throw std::runtime_error(
                    "graph exceeds the 512 MiB activation budget; reduce model or context");
            buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
            if (!buffer)
                throw std::runtime_error("training tensors exceed memory budget");
            std::vector<int32_t> pos(t);
            for (int i = 0; i < t; ++i)
                pos[i] = i;
            ggml_backend_tensor_set(positions, pos.data(), 0, pos.size() * 4);
        } catch (...) {
            release();
            throw;
        }
    }
    ~Impl() {
        release();
    }
    void release() {
        if (buffer)
            ggml_backend_buffer_free(buffer);
        if (backend)
            ggml_backend_free(backend);
        if (ctx)
            ggml_free(ctx);
    }
};
Engine::Engine(const Config& c, int b, int t, int threads, bool back)
    : impl(new Impl(c, b, t, threads, back)) {}
Engine::~Engine() = default;
Step Engine::evaluate(const Model& m, const std::vector<int32_t>& x,
                      const std::vector<int32_t>& y) {
    auto& p = *impl;
    if (x.size() != size_t(p.batch * p.length) || y.size() != x.size())
        throw std::runtime_error("batch shape mismatch");
    if (m.cfg.json() != p.cfg.json())
        throw std::runtime_error("model config mismatch");
    m.verify_frozen();
    if (p.backward && !m.canonical.empty())
        throw std::runtime_error("FLP2 is inference-only");
    if (p.backward)
        ggml_graph_reset(p.graph);
    for (size_t i = 0; i < m.tensors.size(); ++i) {
        const auto& t = m.tensors[i];
        std::vector<float> w;
        if (!m.canonical.empty())
            w = t.w;
        else if (t.norm) {
            w = t.w;
            for (auto& v : w)
                v = half_round(v);
        } else
            w = quantize(t, m.cfg, i == 0).values;
        ggml_backend_tensor_set(p.weights[i], w.data(), 0, w.size() * 4);
    }
    std::vector<float> labels(size_t(p.cfg.vocab) * y.size(), 0);
    for (size_t i = 0; i < y.size(); ++i) {
        if (x[i] < 0 || x[i] >= p.cfg.vocab || y[i] < 0 || y[i] >= p.cfg.vocab)
            throw std::runtime_error("token out of range");
        labels[i * p.cfg.vocab + y[i]] = 1;
    }
    ggml_backend_tensor_set(p.ids, x.data(), 0, x.size() * 4);
    ggml_backend_tensor_set(p.labels, labels.data(), 0, labels.size() * 4);
    if (ggml_backend_graph_compute(p.backend, p.graph) != GGML_STATUS_SUCCESS)
        throw std::runtime_error("graph compute failed");
    Step s;
    ggml_backend_tensor_get(p.loss, &s.loss, 0, 4);
    if (!std::isfinite(s.loss))
        throw std::runtime_error("nonfinite loss");
    s.logits.resize(labels.size());
    ggml_backend_tensor_get(p.output, s.logits.data(), 0, s.logits.size() * 4);
    if (p.backward) {
        for (auto* w : p.weights) {
            auto* g = ggml_graph_get_grad(p.graph, w);
            if (!g)
                throw std::runtime_error("missing weight gradient");
            std::vector<float> v(size_t(ggml_nelements(w)));
            ggml_backend_tensor_get(g, v.data(), 0, v.size() * 4);
            s.grads.push_back(std::move(v));
        }
    }
    return s;
}
void adam(Model& m, const std::vector<std::vector<float>>& g, int step, float lr, float wd) {
    if (!m.canonical.empty())
        throw std::runtime_error("FLP2 is inference-only");
    if (step < 1 || g.size() != m.tensors.size() || !std::isfinite(lr) || lr < 0 ||
        !std::isfinite(wd) || wd < 0)
        throw std::runtime_error("invalid optimizer input");
    double norm = 0;
    for (const auto& a : g)
        for (float x : a) {
            if (!std::isfinite(x))
                throw std::runtime_error("nonfinite gradient");
            norm += double(x) * x;
        }
    float clip = std::min(1.0f, float(1 / (std::sqrt(norm) + 1e-6)));
    float bc1 = 1 - std::pow(0.9f, step), bc2 = 1 - std::pow(0.95f, step);
    for (size_t i = 0; i < g.size(); ++i) {
        auto& t = m.tensors[i];
        if (g[i].size() != t.w.size())
            throw std::runtime_error("gradient shape mismatch");
        for (size_t k = 0; k < t.w.size(); ++k) {
            float x = g[i][k] * clip;
            t.m[k] = .9f * t.m[k] + .1f * x;
            t.v[k] = .95f * t.v[k] + .05f * x * x;
            t.w[k] *= 1 - lr * (t.norm ? 0 : wd);
            t.w[k] -= lr / bc1 * t.m[k] / (std::sqrt(t.v[k]) / std::sqrt(bc2) + 1e-8f);
        }
    }
}
} // namespace xllama::floppy
