// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

// Built only with the llama.cpp backend (it links ggml); the ORT-only UWP
// variant compiles this file to nothing.
#ifdef XLLAMA_USE_LLAMA

    #include "xllama/ggml_d3d12.h"

    #include "ggml-cpu.h"

    #include <algorithm>
    #include <cmath>
    #include <cstdio>
    #include <cstring>
    #include <ctime>
    #include <string>
    #include <vector>

namespace xllama {

// --- Pure rules (host-tested) ---

bool d3d12_weight_type_supported(ggml_type t) {
    return t == GGML_TYPE_Q4_0 || t == GGML_TYPE_Q4_K || t == GGML_TYPE_Q6_K;
}

D3d12Dispatch d3d12_mm_dispatch(std::int64_t n, std::int64_t ncols) {
    D3d12Dispatch d;
    if (n <= 0 || ncols <= 0)
        return d;
    const std::int64_t gx = (n + kD3d12MmvRows - 1) / kD3d12MmvRows;
    d.ok = gx <= kD3d12MaxGroups && ncols <= kD3d12MaxGroups;
    if (d.ok) {
        d.groups_x = static_cast<std::uint32_t>(gx);
        d.groups_y = static_cast<std::uint32_t>(ncols);
    }
    return d;
}

int d3d12_mm_threads(std::int64_t k) {
    return k / kD3d12Chunk >= kD3d12LongKChunks ? kD3d12MmvThreadsLong : kD3d12MmvThreadsShort;
}

bool d3d12_mm_supported(const D3d12MatmulDesc& d) {
    return d3d12_weight_type_supported(d.src0_type) && d.src0_in_weight_buffer &&
           d.src0_contiguous && d.src1_contiguous && d.src1_type == GGML_TYPE_F32 &&
           d.dst_type == GGML_TYPE_F32 && d.ne00 > 0 && d.ne00 % kD3d12Chunk == 0 &&
           d.ne00 == d.ne10 && d.ne02 == 1 && d.ne03 == 1 && d.ne12 == 1 && d.ne13 == 1 &&
           d3d12_mm_dispatch(d.ne01, d.ne11).ok;
}

bool d3d12_get_rows_type_supported(ggml_type t) {
    return t == GGML_TYPE_Q6_K;
}

bool d3d12_get_rows_supported(const D3d12GetRowsDesc& d) {
    return d3d12_get_rows_type_supported(d.src0_type) && d.src0_in_weight_buffer &&
           d.src0_contiguous && d.src1_type == GGML_TYPE_I32 && d.dst_type == GGML_TYPE_F32 &&
           d.ne00 > 0 && d.ne00 % kD3d12Chunk == 0 && d.ne02 == 1 && d.ne03 == 1 && d.ne10 > 0 &&
           d.ne10 <= kD3d12MaxGroups && d.ne11 == 1 && d.ne12 == 1 &&
           d.ne00 / kD3d12Chunk <= kD3d12MaxGroups;
}

bool d3d12_place_tied_embedding(ggml_type embd_type, bool has_output_weight) {
    return !has_output_weight && d3d12_get_rows_type_supported(embd_type);
}

// --- Host emulation of shaders/ggml_d3d12_mmv_*.hlsl ---

namespace {

// Little-endian loads with the shaders' 2-byte-aligned dword trick.
std::uint32_t ld32(const std::uint8_t* w, std::uint32_t a) {
    auto dw = [&](std::uint32_t at) {
        std::uint32_t v;
        std::memcpy(&v, w + at, 4);
        return v;
    };
    const std::uint32_t b = a & ~3u;
    const std::uint32_t lo = dw(b);
    if ((a & 3u) == 0u)
        return lo;
    return (lo >> 16) | (dw(b + 4u) << 16);
}

std::uint32_t ld16(const std::uint8_t* w, std::uint32_t a) {
    std::uint32_t v;
    std::memcpy(&v, w + (a & ~3u), 4);
    return (a & 2u) != 0u ? (v >> 16) : (v & 0xffffu);
}

float h2f(std::uint32_t h) {
    return ggml_fp16_to_fp32(static_cast<ggml_fp16_t>(h));
}

std::int32_t nib(std::uint32_t word, std::uint32_t i, std::uint32_t shift) {
    return static_cast<std::int32_t>((word >> (8u * i + shift)) & 0xFu);
}

std::int32_t sbyte(std::uint32_t word, std::uint32_t i) {
    return static_cast<std::int8_t>((word >> (8u * i)) & 0xffu);
}

void q4k_scale_min(std::uint32_t j, const std::uint8_t* sc, std::int32_t* d, std::int32_t* m) {
    std::uint32_t dd, mm;
    if (j < 4u) {
        dd = sc[j] & 63u;
        mm = sc[j + 4u] & 63u;
    } else {
        dd = (sc[j + 4u] & 0xFu) | ((sc[j - 4u] >> 6) << 4);
        mm = (sc[j + 4u] >> 4) | ((sc[j] >> 6) << 4);
    }
    *d = static_cast<std::int32_t>(dd);
    *m = static_cast<std::int32_t>(mm);
}

// One thread's contribution for one row and one 256-element chunk. `xq` is
// one activation column in the vec_dot type's ggml blocks (block_q8_0 for
// Q4_0, block_q8_K for Q4_K / Q6_K). Integer sums are exact; the float terms
// follow ggml's vec_dot (#312).
float thread_chunk(ggml_type t, const std::uint8_t* w, std::uint32_t row_off, std::uint32_t blk,
                   std::uint32_t itid, const std::uint8_t* xq) {
    if (t == GGML_TYPE_Q4_0) {
        const std::uint32_t b = itid >> 1, h = itid & 1u;
        const std::uint8_t* xb = xq + (blk * 8u + b) * 34u; // block_q8_0
        const float dx = h2f(ld16(xb, 0));
        const auto* qx = reinterpret_cast<const std::int8_t*>(xb + 2);
        const std::uint32_t bb = row_off + (blk * 8u + b) * 18u;
        const float d = h2f(ld16(w, bb));
        const std::uint32_t q0 = ld32(w, bb + 2u + 8u * h), q1 = ld32(w, bb + 6u + 8u * h);
        std::int32_t sumi = 0;
        for (std::uint32_t i = 0; i < 4; ++i) {
            const std::uint32_t lo = 8u * h + i, hi = 16u + 8u * h + i;
            sumi += (nib(q0, i, 0) - 8) * qx[lo] + (nib(q1, i, 0) - 8) * qx[lo + 4] +
                    (nib(q0, i, 4) - 8) * qx[hi] + (nib(q1, i, 4) - 8) * qx[hi + 4];
        }
        return (d * dx) * static_cast<float>(sumi);
    }
    float dy;
    std::memcpy(&dy, xq + blk * 292u, 4); // block_q8_K: float d; int8 qs[256]; ...
    const auto* qx = reinterpret_cast<const std::int8_t*>(xq + blk * 292u + 4u);
    if (t == GGML_TYPE_Q4_K) {
        const std::uint32_t il = itid >> 2, ir = itid & 3u;
        const std::uint32_t e_lo = il * 64u + ir * 8u, e_hi = e_lo + 32u;
        const std::uint32_t bb = row_off + blk * 144u;
        const float d = h2f(ld16(w, bb)), dmin = h2f(ld16(w, bb + 2u));
        std::int32_t sc1, m1, sc2, m2;
        q4k_scale_min(2u * il, w + bb + 4u, &sc1, &m1);
        q4k_scale_min(2u * il + 1u, w + bb + 4u, &sc2, &m2);
        const std::uint32_t qa = ld32(w, bb + 16u + il * 32u + ir * 8u);
        const std::uint32_t qb = ld32(w, bb + 20u + il * 32u + ir * 8u);
        std::int32_t lo = 0, hi = 0, sl = 0, sh = 0;
        for (std::uint32_t i = 0; i < 4; ++i) {
            lo += nib(qa, i, 0) * qx[e_lo + i] + nib(qb, i, 0) * qx[e_lo + 4 + i];
            hi += nib(qa, i, 4) * qx[e_hi + i] + nib(qb, i, 4) * qx[e_hi + 4 + i];
            sl += qx[e_lo + i] + qx[e_lo + 4 + i];
            sh += qx[e_hi + i] + qx[e_hi + 4 + i];
        }
        return (d * dy) * static_cast<float>(sc1 * lo + sc2 * hi) -
               (dmin * dy) * static_cast<float>(m1 * sl + m2 * sh);
    }
    // Q6_K
    const std::uint32_t v = itid >> 3, l0 = 4u * (itid & 7u), is = l0 >> 4;
    const std::uint32_t e = 128u * v + l0;
    const std::uint32_t bb = row_off + blk * 210u;
    const std::uint32_t qla = ld32(w, bb + 64u * v + l0);
    const std::uint32_t qlb = ld32(w, bb + 64u * v + 32u + l0);
    const std::uint32_t qh = ld32(w, bb + 128u + 32u * v + l0);
    const std::uint32_t s0 = ld32(w, bb + 192u + 8u * v), s1 = ld32(w, bb + 196u + 8u * v);
    const float d = h2f(ld16(w, bb + 208u));
    auto q6 = [](std::uint32_t ql, std::uint32_t qhw, std::uint32_t i, std::uint32_t qs,
                 std::uint32_t hs) {
        const std::uint32_t lo4 = (ql >> (8u * i + qs)) & 0xFu;
        const std::uint32_t hi2 = (qhw >> (8u * i + hs)) & 3u;
        return static_cast<std::int32_t>(lo4 | (hi2 << 4)) - 32;
    };
    std::int32_t t1 = 0, t2 = 0, t3 = 0, t4 = 0;
    for (std::uint32_t i = 0; i < 4; ++i) {
        t1 += q6(qla, qh, i, 0, 0) * qx[e + i];
        t2 += q6(qlb, qh, i, 0, 2) * qx[e + 32 + i];
        t3 += q6(qla, qh, i, 4, 4) * qx[e + 64 + i];
        t4 += q6(qlb, qh, i, 4, 6) * qx[e + 96 + i];
    }
    const std::int32_t tsum =
        sbyte(s0, is) * t1 + sbyte(s0, is + 2) * t2 + sbyte(s1, is) * t3 + sbyte(s1, is + 2) * t4;
    return (d * dy) * static_cast<float>(tsum);
}

} // namespace

ggml_type d3d12_activation_type(ggml_type weight) {
    return weight == GGML_TYPE_Q4_0 ? GGML_TYPE_Q8_0 : GGML_TYPE_Q8_K;
}

void d3d12_quantize_activations(ggml_type weight, const float* x, std::size_t x_stride, int k,
                                int ncols, std::uint8_t* out) {
    const ggml_type at = d3d12_activation_type(weight);
    const auto from_float = ggml_get_type_traits_cpu(at)->from_float;
    const std::size_t row = ggml_row_size(at, k);
    for (int c = 0; c < ncols; ++c)
        from_float(x + static_cast<std::size_t>(c) * x_stride, out + c * row, k);
}

void d3d12_mmv_emulate(ggml_type t, const std::uint8_t* w, std::size_t w_row_bytes, const float* x,
                       std::size_t x_stride, float* y, std::size_t y_stride, int n, int k,
                       int ncols) {
    if (!w || !x || !y || n <= 0 || k <= 0 || k % kD3d12Chunk != 0 || ncols <= 0 ||
        !d3d12_weight_type_supported(t))
        return;
    const std::size_t xq_row = ggml_row_size(d3d12_activation_type(t), k);
    std::vector<std::uint8_t> xq(xq_row * static_cast<std::size_t>(ncols) + 4);
    d3d12_quantize_activations(t, x, x_stride, k, ncols, xq.data());
    const std::uint32_t nchunk = static_cast<std::uint32_t>(k / kD3d12Chunk);
    const std::uint32_t threads = static_cast<std::uint32_t>(d3d12_mm_threads(k));
    const std::uint32_t in_flight = threads / 16u;
    float acc[kD3d12MmvThreadsLong];
    for (int c = 0; c < ncols; ++c) {
        const std::uint8_t* xc = xq.data() + static_cast<std::size_t>(c) * xq_row;
        for (int row = 0; row < n; ++row) {
            const std::uint32_t row_off = static_cast<std::uint32_t>(row * w_row_bytes);
            for (std::uint32_t tid = 0; tid < threads; ++tid) {
                float a = 0.f;
                for (std::uint32_t blk = tid >> 4; blk < nchunk; blk += in_flight)
                    a += thread_chunk(t, w, row_off, blk, tid & 15u, xc);
                acc[tid] = a;
            }
            for (int stride = static_cast<int>(threads) / 2; stride > 0; stride >>= 1)
                for (int i = 0; i < stride; ++i)
                    acc[i] += acc[i + stride];
            y[static_cast<std::size_t>(c) * y_stride + row] = acc[0];
        }
    }
}

void d3d12_get_rows_emulate(ggml_type t, const std::uint8_t* w, std::size_t w_row_bytes,
                            const std::int32_t* ids, int n_ids, int k, float* y,
                            std::size_t y_stride) {
    if (!w || !ids || !y || n_ids <= 0 || k <= 0 || k % kD3d12Chunk != 0 ||
        !d3d12_get_rows_type_supported(t))
        return;
    auto ld8 = [&](std::uint32_t a) { return (ld32(w, a & ~3u) >> ((a & 3u) * 8u)) & 0xffu; };
    // One thread group per (256-element chunk, id); thread tid = 32v + l writes
    // elements 128v + l + {0, 32, 64, 96}, as dequantize_row_q6_K's inner loop.
    for (int i = 0; i < n_ids; ++i) {
        for (std::uint32_t blk = 0; blk < static_cast<std::uint32_t>(k / kD3d12Chunk); ++blk) {
            const std::uint32_t bb = static_cast<std::uint32_t>(ids[i] * w_row_bytes) + blk * 210u;
            const float d = h2f(ld16(w, bb + 208u));
            for (std::uint32_t tid = 0; tid < 64u; ++tid) {
                const std::uint32_t v = tid >> 5, l = tid & 31u, is = l >> 4;
                const std::uint32_t qla = ld8(bb + 64u * v + l), qlb = ld8(bb + 64u * v + l + 32u);
                const std::uint32_t qh = ld8(bb + 128u + 32u * v + l);
                auto sc = [&](std::uint32_t j) {
                    return static_cast<float>(
                        static_cast<std::int8_t>(ld8(bb + 192u + 8u * v + is + j)));
                };
                const int q1 = static_cast<int>((qla & 0xFu) | (((qh >> 0) & 3u) << 4)) - 32;
                const int q2 = static_cast<int>((qlb & 0xFu) | (((qh >> 2) & 3u) << 4)) - 32;
                const int q3 = static_cast<int>((qla >> 4) | (((qh >> 4) & 3u) << 4)) - 32;
                const int q4 = static_cast<int>((qlb >> 4) | (((qh >> 6) & 3u) << 4)) - 32;
                float* out = y + static_cast<std::size_t>(i) * y_stride + blk * 256u + 128u * v + l;
                out[0] = d * sc(0) * static_cast<float>(q1);
                out[32] = d * sc(2) * static_cast<float>(q2);
                out[64] = d * sc(4) * static_cast<float>(q3);
                out[96] = d * sc(6) * static_cast<float>(q4);
            }
        }
    }
}

// --- Selftest CSV ---

const char* d3d12_selftest_csv_header() {
    return "type,n,k,ncols,rel_err,gpu_ms,packed_gbs,ok,d3d12_ran,host,date,error\n";
}

std::string format_d3d12_selftest_row(const D3d12SelftestRow& r, const char* host_label) {
    char date_buf[32];
    std::time_t now = std::time(nullptr);
    std::strftime(date_buf, sizeof(date_buf), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&now));
    std::string err = r.error.empty() ? "-" : r.error;
    for (char& ch : err)
        if (ch == ',' || ch == '\n' || ch == '\r')
            ch = ' ';
    char buf[512];
    std::snprintf(buf, sizeof(buf), "%s,%d,%d,%d,%.3g,%.4f,%.2f,%d,%d,%s,%s,%s\n", r.type.c_str(),
                  r.n, r.k, r.ncols, r.rel_err, r.gpu_ms, r.packed_gbs, r.ok ? 1 : 0,
                  r.d3d12_ran ? 1 : 0, host_label ? host_label : "unknown", date_buf, err.c_str());
    return buf;
}

} // namespace xllama

    #if !defined(_WIN32)

namespace xllama {

bool ggml_d3d12_register() {
    return false;
}

ggml_backend_buffer_type_t ggml_d3d12_weights_buft() {
    return nullptr;
}

void run_d3d12_selftest(std::vector<D3d12SelftestRow>* out) {
    if (!out)
        return;
    D3d12SelftestRow r;
    r.type = "-";
    r.error = "d3d12 unavailable on this platform";
    out->push_back(std::move(r));
}

} // namespace xllama

    #else // _WIN32 — the backend itself

        #include <chrono>
        #include <cstdint>
        #include <mutex>
        #include <random>

        #include <dxgi1_4.h>

        #include "d3d12_compute.h"
        #include "ggml-alloc.h"
        #include "ggml-backend-impl.h"
        #include "ggml-backend.h"
        #include "ggml_d3d12_get_rows_q6_k_dxil.h"
        #include "ggml_d3d12_mmv_q4_0_t128_dxil.h"
        #include "ggml_d3d12_mmv_q4_0_t64_dxil.h"
        #include "ggml_d3d12_mmv_q4_k_t128_dxil.h"
        #include "ggml_d3d12_mmv_q4_k_t64_dxil.h"
        #include "ggml_d3d12_mmv_q6_k_t128_dxil.h"
        #include "ggml_d3d12_mmv_q6_k_t64_dxil.h"
        #include "xllama/d3d12_dyn.h"
        #include "xllama/platform.h"

namespace xllama {
namespace {

using d3d12c::ComPtr;

enum Pso { kPsoQ40 = 0, kPsoQ4K = 1, kPsoQ6K = 2, kPsoCount = 3 };

int pso_for(ggml_type t) {
    switch (t) {
    case GGML_TYPE_Q4_0:
        return kPsoQ40;
    case GGML_TYPE_Q4_K:
        return kPsoQ4K;
    case GGML_TYPE_Q6_K:
        return kPsoQ6K;
    default:
        return -1;
    }
}

// Process-wide D3D12 state behind the ggml device. One queue, serialised.
struct Gpu {
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> alloc;
    ComPtr<ID3D12GraphicsCommandList> cl;
    ComPtr<ID3D12RootSignature> root;
    ComPtr<ID3D12PipelineState> pso[kPsoCount][2]; // [type][0 = 64 threads, 1 = 128]
    ComPtr<ID3D12PipelineState> pso_get_rows;      // Q6_K
    ComPtr<ID3D12QueryHeap> ts;
    ComPtr<ID3D12Resource> ts_rb;
    ComPtr<ID3D12Resource> staging;  // weight upload ring (kStagingBytes)
    ComPtr<ID3D12Resource> readback; // get_tensor on weights (kStagingBytes)
    std::uint8_t* staging_ptr = nullptr;
    std::uint8_t* readback_ptr = nullptr;
    // q8 activations for the next submission (CPU-written, GPU-read; grows on
    // demand to the largest single input, ~6 MiB for a 512-token ubatch).
    ComPtr<ID3D12Resource> xq;
    std::uint8_t* xq_ptr = nullptr;
    std::size_t xq_bytes = 0;
    d3d12c::QueueFence fence;
    UINT64 ts_freq = 0;
    LUID luid = {};
    double last_gpu_ms = 0.0;
    // Per-backend-lifetime counters, logged when the backend is freed.
    std::uint64_t n_calls = 0;
    std::uint64_t n_matmuls = 0;
    double wall_ms = 0.0;
    double gpu_ms = 0.0;
    std::mutex mu;
    bool ok = false;
    std::string error;
};

// Weight upload / readback ring. Mapped for the process lifetime and counted in
// its working set, so it stays small: 64 MiB each put the first GPU-layer
// smoke at 640 MB peak vs 311 MB on the CPU (D2b); 8 MiB costs a few more
// round trips at load only.
constexpr UINT64 kStagingBytes = 8ull << 20;

ComPtr<ID3D12RootSignature> make_root_sig(ID3D12Device* device, std::string* err) {
    D3D12_ROOT_PARAMETER p[4] = {};
    p[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    p[0].Constants.ShaderRegister = 0;
    p[0].Constants.Num32BitValues = 8;
    p[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV; // t0 weights
    p[1].Descriptor.ShaderRegister = 0;
    p[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV; // u0 output
    p[2].Descriptor.ShaderRegister = 0;
    p[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV; // u1 activations
    p[3].Descriptor.ShaderRegister = 1;
    for (auto& q : p)
        q.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    D3D12_ROOT_SIGNATURE_DESC desc = {};
    desc.NumParameters = 4;
    desc.pParameters = p;
    auto serialize = d3d12_dyn::SerializeRootSignature();
    if (!serialize) {
        *err = "D3D12SerializeRootSignature not available";
        return {};
    }
    ComPtr<ID3DBlob> sig, blob_err;
    HRESULT hr = serialize(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &sig, &blob_err);
    if (FAILED(hr)) {
        *err = d3d12c::hr_message("D3D12SerializeRootSignature", hr);
        return {};
    }
    ComPtr<ID3D12RootSignature> root;
    hr = device->CreateRootSignature(0, sig->GetBufferPointer(), sig->GetBufferSize(),
                                     IID_PPV_ARGS(&root));
    if (FAILED(hr)) {
        *err = d3d12c::hr_message("CreateRootSignature", hr);
        return {};
    }
    return root;
}

bool init_gpu(Gpu& g) {
    std::string& err = g.error;
    g.device = d3d12c::create_device(&err);
    if (!g.device)
        return false;
    D3D12_FEATURE_DATA_ARCHITECTURE1 arch = {};
    if (FAILED(g.device->CheckFeatureSupport(D3D12_FEATURE_ARCHITECTURE1, &arch, sizeof(arch))) ||
        !arch.CacheCoherentUMA) {
        // D3D12_Host relies on a CPU-cached GPU-visible heap (D1: cc_uma=1 on Series S).
        err = "adapter is not cache-coherent UMA";
        return false;
    }
    g.luid = g.device->GetAdapterLuid();
    D3D12_COMMAND_QUEUE_DESC qd = {};
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    HRESULT hr = g.device->CreateCommandQueue(&qd, IID_PPV_ARGS(&g.queue));
    if (FAILED(hr)) {
        err = d3d12c::hr_message("CreateCommandQueue", hr);
        return false;
    }
    hr = g.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&g.alloc));
    if (SUCCEEDED(hr))
        hr = g.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, g.alloc.Get(), nullptr,
                                         IID_PPV_ARGS(&g.cl));
    if (FAILED(hr)) {
        err = d3d12c::hr_message("CreateCommandList", hr);
        return false;
    }
    g.cl->Close();
    if (!g.fence.init(g.device.Get(), &err))
        return false;
    g.root = make_root_sig(g.device.Get(), &err);
    if (!g.root)
        return false;
    const void* blobs[kPsoCount][2] = {{kGgmlD3d12MmvQ40T64Dxil, kGgmlD3d12MmvQ40T128Dxil},
                                       {kGgmlD3d12MmvQ4KT64Dxil, kGgmlD3d12MmvQ4KT128Dxil},
                                       {kGgmlD3d12MmvQ6KT64Dxil, kGgmlD3d12MmvQ6KT128Dxil}};
    const size_t sizes[kPsoCount][2] = {
        {kGgmlD3d12MmvQ40T64DxilSize, kGgmlD3d12MmvQ40T128DxilSize},
        {kGgmlD3d12MmvQ4KT64DxilSize, kGgmlD3d12MmvQ4KT128DxilSize},
        {kGgmlD3d12MmvQ6KT64DxilSize, kGgmlD3d12MmvQ6KT128DxilSize}};
    for (int i = 0; i < kPsoCount; ++i) {
        for (int wd = 0; wd < 2; ++wd) {
            D3D12_COMPUTE_PIPELINE_STATE_DESC pd = {};
            pd.pRootSignature = g.root.Get();
            pd.CS.pShaderBytecode = blobs[i][wd];
            pd.CS.BytecodeLength = sizes[i][wd];
            hr = g.device->CreateComputePipelineState(&pd, IID_PPV_ARGS(&g.pso[i][wd]));
            if (FAILED(hr)) {
                err = d3d12c::hr_message("CreateComputePipelineState", hr);
                return false;
            }
        }
    }
    {
        D3D12_COMPUTE_PIPELINE_STATE_DESC pd = {};
        pd.pRootSignature = g.root.Get();
        pd.CS.pShaderBytecode = kGgmlD3d12GetRowsQ6KDxil;
        pd.CS.BytecodeLength = kGgmlD3d12GetRowsQ6KDxilSize;
        hr = g.device->CreateComputePipelineState(&pd, IID_PPV_ARGS(&g.pso_get_rows));
        if (FAILED(hr)) {
            err = d3d12c::hr_message("CreateComputePipelineState", hr);
            return false;
        }
    }
    if (SUCCEEDED(g.queue->GetTimestampFrequency(&g.ts_freq)) && g.ts_freq > 0) {
        D3D12_QUERY_HEAP_DESC qhd = {};
        qhd.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
        qhd.Count = 2;
        g.device->CreateQueryHeap(&qhd, IID_PPV_ARGS(&g.ts));
        g.ts_rb = d3d12c::create_buffer(g.device.Get(), 256, D3D12_HEAP_TYPE_READBACK,
                                        D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST,
                                        "ts_rb", &err);
        if (!g.ts || !g.ts_rb) {
            g.ts.Reset();
            g.ts_rb.Reset();
            err.clear();
        }
    }
    g.staging = d3d12c::create_buffer(g.device.Get(), kStagingBytes, D3D12_HEAP_TYPE_UPLOAD,
                                      D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ,
                                      "staging", &err);
    g.readback = d3d12c::create_buffer(g.device.Get(), kStagingBytes, D3D12_HEAP_TYPE_READBACK,
                                       D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST,
                                       "readback", &err);
    if (!g.staging || !g.readback)
        return false;
    void* p = nullptr;
    if (FAILED(g.staging->Map(0, nullptr, &p)))
        return false;
    g.staging_ptr = static_cast<std::uint8_t*>(p);
    if (FAILED(g.readback->Map(0, nullptr, &p)))
        return false;
    g.readback_ptr = static_cast<std::uint8_t*>(p);
    return true;
}

Gpu& gpu() {
    static Gpu g;
    static std::once_flag once;
    std::call_once(once, [] { g.ok = init_gpu(g); });
    return g;
}

// Record into the shared list and run it to completion. Caller holds g.mu.
template <typename F> bool run_now(Gpu& g, F&& record) {
    g.alloc->Reset();
    g.cl->Reset(g.alloc.Get(), nullptr);
    record(g.cl.Get());
    g.cl->Close();
    ID3D12CommandList* lists[] = {g.cl.Get()};
    g.queue->ExecuteCommandLists(1, lists);
    return g.fence.signal_and_wait(g.queue.Get(), /*spin=*/true);
}

// --- Buffers ---

struct BufCtx {
    ComPtr<ID3D12Resource> res;
    D3D12_GPU_VIRTUAL_ADDRESS va = 0;
    std::uint8_t* host = nullptr; // mapped pointer (D3D12_Host only)
    bool is_host = false;
};

void* buf_base(ggml_backend_buffer_t b) {
    auto* c = static_cast<BufCtx*>(b->context);
    return c->is_host ? static_cast<void*>(c->host)
                      : reinterpret_cast<void*>(static_cast<std::uintptr_t>(c->va));
}

void buf_free(ggml_backend_buffer_t b) {
    delete static_cast<BufCtx*>(b->context);
}

std::size_t tensor_offset(const ggml_tensor* t) {
    return static_cast<std::size_t>(static_cast<const std::uint8_t*>(t->data) -
                                    static_cast<const std::uint8_t*>(buf_base(t->buffer)));
}

D3D12_GPU_VIRTUAL_ADDRESS tensor_va(const ggml_tensor* t) {
    return static_cast<BufCtx*>(t->buffer->context)->va + tensor_offset(t);
}

// D3D12_Host: plain memory.
void host_memset(ggml_backend_buffer_t, ggml_tensor* t, uint8_t v, size_t off, size_t size) {
    std::memset(static_cast<std::uint8_t*>(t->data) + off, v, size);
}
void host_set(ggml_backend_buffer_t, ggml_tensor* t, const void* data, size_t off, size_t size) {
    std::memcpy(static_cast<std::uint8_t*>(t->data) + off, data, size);
}
void host_get(ggml_backend_buffer_t, const ggml_tensor* t, void* data, size_t off, size_t size) {
    std::memcpy(data, static_cast<const std::uint8_t*>(t->data) + off, size);
}
bool host_cpy(ggml_backend_buffer_t, const ggml_tensor* src, ggml_tensor* dst) {
    if (!ggml_backend_buffer_is_host(src->buffer))
        return false;
    std::memcpy(dst->data, src->data, ggml_nbytes(src));
    return true;
}
void host_clear(ggml_backend_buffer_t b, uint8_t v) {
    std::memset(static_cast<BufCtx*>(b->context)->host, v, b->size);
}

const ggml_backend_buffer_i kHostBufIface = {
    /* .free_buffer   = */ buf_free,
    /* .get_base      = */ buf_base,
    /* .init_tensor   = */ nullptr,
    /* .memset_tensor = */ host_memset,
    /* .set_tensor    = */ host_set,
    /* .get_tensor    = */ host_get,
    /* .set_tensor_2d = */ nullptr,
    /* .get_tensor_2d = */ nullptr,
    /* .cpy_tensor    = */ host_cpy,
    /* .clear         = */ host_clear,
    /* .reset         = */ nullptr,
};

// D3D12_Weights: DEFAULT heap, filled through the staging ring. The buffer
// stays in COMMON; buffers promote implicitly to COPY_DEST / shader read and
// decay back at the end of each submission.
void weights_write(ggml_backend_buffer_t b, std::size_t dst_off, const void* data, std::size_t size,
                   int fill) {
    Gpu& g = gpu();
    std::lock_guard<std::mutex> lock(g.mu);
    auto* c = static_cast<BufCtx*>(b->context);
    for (std::size_t done = 0; done < size;) {
        const std::size_t chunk = std::min<std::size_t>(size - done, kStagingBytes);
        if (data)
            std::memcpy(g.staging_ptr, static_cast<const std::uint8_t*>(data) + done, chunk);
        else
            std::memset(g.staging_ptr, fill, chunk);
        run_now(g, [&](ID3D12GraphicsCommandList* cl) {
            cl->CopyBufferRegion(c->res.Get(), dst_off + done, g.staging.Get(), 0, chunk);
        });
        done += chunk;
    }
}
void weights_memset(ggml_backend_buffer_t b, ggml_tensor* t, uint8_t v, size_t off, size_t size) {
    weights_write(b, tensor_offset(t) + off, nullptr, size, v);
}
void weights_set(ggml_backend_buffer_t b, ggml_tensor* t, const void* data, size_t off,
                 size_t size) {
    weights_write(b, tensor_offset(t) + off, data, size, 0);
}
void weights_get(ggml_backend_buffer_t b, const ggml_tensor* t, void* data, size_t off,
                 size_t size) {
    Gpu& g = gpu();
    std::lock_guard<std::mutex> lock(g.mu);
    auto* c = static_cast<BufCtx*>(b->context);
    const std::size_t src_off = tensor_offset(t) + off;
    for (std::size_t done = 0; done < size;) {
        const std::size_t chunk = std::min<std::size_t>(size - done, kStagingBytes);
        run_now(g, [&](ID3D12GraphicsCommandList* cl) {
            cl->CopyBufferRegion(g.readback.Get(), 0, c->res.Get(), src_off + done, chunk);
        });
        std::memcpy(static_cast<std::uint8_t*>(data) + done, g.readback_ptr, chunk);
        done += chunk;
    }
}
void weights_clear(ggml_backend_buffer_t b, uint8_t v) {
    weights_write(b, 0, nullptr, b->size, v);
}

const ggml_backend_buffer_i kWeightsBufIface = {
    /* .free_buffer   = */ buf_free,
    /* .get_base      = */ buf_base,
    /* .init_tensor   = */ nullptr,
    /* .memset_tensor = */ weights_memset,
    /* .set_tensor    = */ weights_set,
    /* .get_tensor    = */ weights_get,
    /* .set_tensor_2d = */ nullptr,
    /* .get_tensor_2d = */ nullptr,
    /* .cpy_tensor    = */ nullptr,
    /* .clear         = */ weights_clear,
    /* .reset         = */ nullptr,
};

// --- Buffer types ---

ggml_backend_dev_t device_ptr();

const char* host_buft_name(ggml_backend_buffer_type_t) {
    return "D3D12_Host";
}
const char* weights_buft_name(ggml_backend_buffer_type_t) {
    return "D3D12_Weights";
}
size_t buft_alignment(ggml_backend_buffer_type_t) {
    return 256;
}
bool buft_is_host_true(ggml_backend_buffer_type_t) {
    return true;
}
// The Q4_0/Q6_K kernels read whole aligned dwords, up to 2 bytes past a
// tensor's last block; root descriptors have no bounds check, so pad.
size_t weights_alloc_size(ggml_backend_buffer_type_t, const ggml_tensor* t) {
    return ggml_nbytes(t) + 16;
}

ggml_backend_buffer_t alloc_buf(ggml_backend_buffer_type_t buft, size_t size, bool host) {
    Gpu& g = gpu();
    if (!g.ok)
        return nullptr;
    size = std::max<size_t>(size, 256);
    auto* c = new BufCtx;
    c->is_host = host;
    std::string err;
    if (host) {
        D3D12_HEAP_PROPERTIES hp = {};
        hp.Type = D3D12_HEAP_TYPE_CUSTOM;
        hp.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_WRITE_BACK;
        hp.MemoryPoolPreference = D3D12_MEMORY_POOL_L0;
        c->res = d3d12c::create_buffer_props(
            g.device.Get(), size, hp, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, "D3D12_Host", &err);
        void* p = nullptr;
        if (c->res && SUCCEEDED(c->res->Map(0, nullptr, &p)))
            c->host = static_cast<std::uint8_t*>(p);
    } else {
        c->res = d3d12c::create_buffer(g.device.Get(), size, D3D12_HEAP_TYPE_DEFAULT,
                                       D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COMMON,
                                       "D3D12_Weights", &err);
    }
    if (!c->res || (host && !c->host)) {
        char msg[256];
        std::snprintf(msg, sizeof(msg), "[xllama] d3d12: allocation of %zu bytes failed: %s\n",
                      size, err.c_str());
        log_output(msg);
        delete c;
        return nullptr;
    }
    c->va = c->res->GetGPUVirtualAddress();
    if (size >= (1u << 20)) {
        char msg[128];
        std::snprintf(msg, sizeof(msg), "[xllama] d3d12: %s buffer %.1f MiB\n",
                      host ? "D3D12_Host" : "D3D12_Weights", static_cast<double>(size) / 1048576.0);
        log_output(msg);
    }
    return ggml_backend_buffer_init(buft, host ? kHostBufIface : kWeightsBufIface, c, size);
}

ggml_backend_buffer_t host_alloc(ggml_backend_buffer_type_t buft, size_t size) {
    return alloc_buf(buft, size, true);
}
ggml_backend_buffer_t weights_alloc(ggml_backend_buffer_type_t buft, size_t size) {
    return alloc_buf(buft, size, false);
}

ggml_backend_buffer_type kHostBuft = {
    /* .iface = */ {host_buft_name, host_alloc, buft_alignment, nullptr, nullptr,
                    buft_is_host_true},
    /* .device  = */ nullptr,
    /* .context = */ nullptr,
};
ggml_backend_buffer_type kWeightsBuft = {
    /* .iface = */ {weights_buft_name, weights_alloc, buft_alignment, nullptr, weights_alloc_size,
                    nullptr},
    /* .device  = */ nullptr,
    /* .context = */ nullptr,
};

bool is_ours(ggml_backend_buffer_type_t buft) {
    return buft == &kHostBuft || buft == &kWeightsBuft;
}

// --- Backend ---

const char* backend_name(ggml_backend_t) {
    return "D3D12";
}
void backend_free(ggml_backend_t b) {
    Gpu& g = gpu();
    {
        std::lock_guard<std::mutex> lock(g.mu);
        char msg[256];
        std::snprintf(msg, sizeof(msg),
                      "[xllama] d3d12: %llu graph_compute calls, %llu matmuls, %.1f ms wall "
                      "(%.1f ms GPU)\n",
                      static_cast<unsigned long long>(g.n_calls),
                      static_cast<unsigned long long>(g.n_matmuls), g.wall_ms, g.gpu_ms);
        log_output(msg);
        g.n_calls = g.n_matmuls = 0;
        g.wall_ms = g.gpu_ms = 0.0;
    }
    delete b;
}

// Grow the q8 activation scratch. Caller holds g.mu and has nothing in flight.
bool ensure_xq(Gpu& g, std::size_t bytes) {
    if (bytes <= g.xq_bytes)
        return true;
    const std::size_t want = std::max<std::size_t>(bytes, std::size_t(1) << 20);
    g.xq.Reset();
    g.xq_ptr = nullptr;
    g.xq_bytes = 0;
    D3D12_HEAP_PROPERTIES hp = {};
    hp.Type = D3D12_HEAP_TYPE_CUSTOM;
    hp.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_WRITE_BACK;
    hp.MemoryPoolPreference = D3D12_MEMORY_POOL_L0;
    std::string err;
    g.xq = d3d12c::create_buffer_props(g.device.Get(), want, hp,
                                       D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                       D3D12_RESOURCE_STATE_UNORDERED_ACCESS, "D3D12_Xq", &err);
    void* p = nullptr;
    if (!g.xq || FAILED(g.xq->Map(0, nullptr, &p))) {
        log_output("[xllama] d3d12: q8 activation scratch allocation failed: " + err + "\n");
        g.xq.Reset();
        return false;
    }
    g.xq_ptr = static_cast<std::uint8_t*>(p);
    g.xq_bytes = want;
    return true;
}

// The tensor that owns a node's memory (follows view chains).
const ggml_tensor* base_of(const ggml_tensor* t) {
    while (t->view_src)
        t = t->view_src;
    return t;
}

// One dispatch: a MUL_MAT (u1 = its quantized input) or a GET_ROWS (u1 = its
// row ids), both staged in g.xq.
struct Op {
    const ggml_tensor* node;
    std::size_t xq_off;   // staged input in g.xq
    std::uint32_t xq_row; // bytes per quantized column (MUL_MAT)
};

// Record and run one submission. Caller holds g.mu.
bool submit_ops(Gpu& g, const std::vector<Op>& ops) {
    const bool ts = g.ts && g.ts_rb;
    const bool ran = run_now(g, [&](ID3D12GraphicsCommandList* cl) {
        cl->SetComputeRootSignature(g.root.Get());
        if (ts)
            cl->EndQuery(g.ts.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0);
        const D3D12_GPU_VIRTUAL_ADDRESS xq_va = g.xq->GetGPUVirtualAddress();
        for (const Op& op : ops) {
            const ggml_tensor* node = op.node;
            const ggml_tensor* w = node->src[0];
            if (node->op == GGML_OP_GET_ROWS) {
                const std::uint32_t c[8] = {static_cast<std::uint32_t>(node->ne[1]),
                                            static_cast<std::uint32_t>(w->ne[0]),
                                            static_cast<std::uint32_t>(w->ne[0] / kD3d12Chunk),
                                            static_cast<std::uint32_t>(w->nb[1]),
                                            0,
                                            static_cast<std::uint32_t>(node->nb[1] / sizeof(float)),
                                            0,
                                            0};
                cl->SetPipelineState(g.pso_get_rows.Get());
                cl->SetComputeRoot32BitConstants(0, 8, c, 0);
                cl->SetComputeRootShaderResourceView(1, tensor_va(w));
                cl->SetComputeRootUnorderedAccessView(2, tensor_va(node));
                cl->SetComputeRootUnorderedAccessView(3, xq_va + op.xq_off);
                cl->Dispatch(static_cast<UINT>(w->ne[0] / kD3d12Chunk),
                             static_cast<UINT>(node->ne[1]), 1);
                continue;
            }
            const D3d12Dispatch d = d3d12_mm_dispatch(node->ne[0], node->ne[1]);
            const std::uint32_t c[8] = {static_cast<std::uint32_t>(w->ne[1]),
                                        static_cast<std::uint32_t>(w->ne[0]),
                                        static_cast<std::uint32_t>(w->ne[0] / kD3d12Chunk),
                                        static_cast<std::uint32_t>(w->nb[1]),
                                        op.xq_row,
                                        static_cast<std::uint32_t>(node->nb[1] / sizeof(float)),
                                        0,
                                        0};
            const int wide = d3d12_mm_threads(w->ne[0]) == kD3d12MmvThreadsLong ? 1 : 0;
            cl->SetPipelineState(g.pso[pso_for(w->type)][wide].Get());
            cl->SetComputeRoot32BitConstants(0, 8, c, 0);
            cl->SetComputeRootShaderResourceView(1, tensor_va(w));
            cl->SetComputeRootUnorderedAccessView(2, tensor_va(node));
            cl->SetComputeRootUnorderedAccessView(3, xq_va + op.xq_off);
            cl->Dispatch(d.groups_x, d.groups_y, 1);
        }
        if (ts) {
            cl->EndQuery(g.ts.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 1);
            cl->ResolveQueryData(g.ts.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, g.ts_rb.Get(), 0);
        }
    });
    if (!ran)
        return false;
    if (ts) {
        void* p = nullptr;
        if (SUCCEEDED(g.ts_rb->Map(0, nullptr, &p))) {
            const auto* t = static_cast<const std::uint64_t*>(p);
            g.last_gpu_ms += t[1] > t[0] ? 1000.0 * static_cast<double>(t[1] - t[0]) /
                                               static_cast<double>(g.ts_freq)
                                         : 0.0;
            g.ts_rb->Unmap(0, nullptr);
        }
    }
    g.n_matmuls += ops.size();
    return true;
}

ggml_status backend_graph_compute(ggml_backend_t, ggml_cgraph* cgraph) {
    Gpu& g = gpu();
    std::vector<const ggml_tensor*> mm;
    for (int i = 0; i < ggml_graph_n_nodes(cgraph); ++i) {
        const ggml_tensor* node = ggml_graph_node(cgraph, i);
        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0)
            continue;
        switch (node->op) {
        case GGML_OP_MUL_MAT:
        case GGML_OP_GET_ROWS:
            mm.push_back(node);
            break;
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            break;
        default:
            GGML_ABORT("d3d12: unsupported op %s", ggml_op_desc(node));
        }
    }
    if (mm.empty())
        return GGML_STATUS_SUCCESS;

    std::lock_guard<std::mutex> lock(g.mu);
    const auto t0 = std::chrono::steady_clock::now();
    g.last_gpu_ms = 0.0;
    // Activations are quantized on the CPU before the GPU reads them, so a
    // matmul whose input another matmul of the same batch produces (a LoRA
    // chain) waits for that batch. Matmuls sharing an input (q/k/v, gate/up)
    // share one quantized copy.
    std::vector<Op> ops;
    std::vector<const ggml_tensor*> produced; // outputs of `ops`
    struct Quantized {
        const void* data;
        ggml_type type;
        std::size_t off;
        std::uint32_t row;
    };
    std::vector<Quantized> done;
    std::size_t used = 0;
    auto flush = [&]() -> bool {
        if (ops.empty())
            return true;
        const bool ok = submit_ops(g, ops);
        ops.clear();
        produced.clear();
        done.clear();
        used = 0;
        return ok;
    };
    for (const ggml_tensor* node : mm) {
        const ggml_tensor* w = node->src[0];
        const ggml_tensor* x = node->src[1];
        GGML_ASSERT(w->buffer && w->buffer->buft == &kWeightsBuft);
        GGML_ASSERT(x->buffer && ggml_backend_buffer_is_host(x->buffer));
        GGML_ASSERT(node->buffer && is_ours(node->buffer->buft));
        if (node->op == GGML_OP_GET_ROWS) {
            // Row ids are tiny: stage them next to the quantized activations.
            const std::size_t bytes = static_cast<std::size_t>(x->ne[0]) * sizeof(std::int32_t);
            if (used + bytes > g.xq_bytes && !flush())
                return GGML_STATUS_FAILED;
            if (!ensure_xq(g, used + bytes))
                return GGML_STATUS_FAILED;
            std::memcpy(g.xq_ptr + used, x->data, bytes);
            ops.push_back({node, used, 0});
            produced.push_back(base_of(node));
            used += (bytes + 255) & ~std::size_t(255);
            continue;
        }
        const ggml_tensor* xb = base_of(x);
        for (const ggml_tensor* p : produced)
            if (p == xb) {
                if (!flush())
                    return GGML_STATUS_FAILED;
                break;
            }
        const ggml_type at = d3d12_activation_type(w->type);
        const int k = static_cast<int>(x->ne[0]);
        const int ncols = static_cast<int>(x->ne[1]);
        const std::uint32_t row = static_cast<std::uint32_t>(ggml_row_size(at, k));
        const Quantized* q = nullptr;
        for (const Quantized& d : done)
            if (d.data == x->data && d.type == at && d.row == row)
                q = &d;
        if (!q) {
            const std::size_t bytes = static_cast<std::size_t>(row) * ncols;
            if (used + bytes > g.xq_bytes && !flush())
                return GGML_STATUS_FAILED;
            if (!ensure_xq(g, used + bytes))
                return GGML_STATUS_FAILED;
            d3d12_quantize_activations(w->type, static_cast<const float*>(x->data),
                                       x->nb[1] / sizeof(float), k, ncols, g.xq_ptr + used);
            done.push_back({x->data, at, used, row});
            q = &done.back();
            used += (bytes + 255) & ~std::size_t(255); // root UAV addresses stay aligned
        }
        ops.push_back({node, q->off, q->row});
        produced.push_back(base_of(node));
    }
    if (!flush())
        return GGML_STATUS_FAILED;
    ++g.n_calls;
    g.gpu_ms += g.last_gpu_ms;
    g.wall_ms +=
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return GGML_STATUS_SUCCESS;
}

ggml_guid_t backend_guid() {
    static ggml_guid guid = {0x78, 0x6c, 0x6c, 0x61, 0x6d, 0x61, 0x2d, 0x64,
                             0x33, 0x64, 0x31, 0x32, 0x2d, 0x76, 0x30, 0x31};
    return &guid;
}

const ggml_backend_i kBackendIface = {
    /* .get_name            = */ backend_name,
    /* .free                = */ backend_free,
    /* .set_tensor_async    = */ nullptr,
    /* .get_tensor_async    = */ nullptr,
    /* .set_tensor_2d_async = */ nullptr,
    /* .get_tensor_2d_async = */ nullptr,
    /* .cpy_tensor_async    = */ nullptr,
    /* .synchronize         = */ nullptr, // graph_compute completes before returning
    /* .graph_plan_create   = */ nullptr,
    /* .graph_plan_free     = */ nullptr,
    /* .graph_plan_update   = */ nullptr,
    /* .graph_plan_compute  = */ nullptr,
    /* .graph_compute       = */ backend_graph_compute,
    /* .event_record        = */ nullptr,
    /* .event_wait          = */ nullptr,
    /* .graph_optimize      = */ nullptr,
};

// --- Device ---

const char* dev_name(ggml_backend_dev_t) {
    return "D3D12";
}
const char* dev_description(ggml_backend_dev_t) {
    return "xllama D3D12 compute (Q4_0/Q4_K/Q6_K matmul, Q6_K get_rows)";
}
void dev_memory(ggml_backend_dev_t, size_t* free, size_t* total) {
    *free = *total = 0;
    Gpu& g = gpu();
    ComPtr<IDXGIFactory4> factory;
    ComPtr<IDXGIAdapter3> adapter;
    if (!g.ok || FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))) ||
        FAILED(factory->EnumAdapterByLuid(g.luid, IID_PPV_ARGS(&adapter))))
        return;
    DXGI_QUERY_VIDEO_MEMORY_INFO info = {};
    if (SUCCEEDED(adapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &info))) {
        *total = static_cast<size_t>(info.Budget);
        *free = info.Budget > info.CurrentUsage
                    ? static_cast<size_t>(info.Budget - info.CurrentUsage)
                    : 0;
    }
}
enum ggml_backend_dev_type dev_type(ggml_backend_dev_t) {
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}
void dev_props(ggml_backend_dev_t dev, ggml_backend_dev_props* props) {
    props->name = dev_name(dev);
    props->description = dev_description(dev);
    props->type = dev_type(dev);
    dev_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {/* async */ false, /* host_buffer */ false, /* buffer_from_host_ptr */ false,
                   /* events */ false, /* mmap_support */ false};
}
ggml_backend_t dev_init_backend(ggml_backend_dev_t dev, const char*) {
    if (!gpu().ok)
        return nullptr;
    return new ggml_backend{backend_guid(), kBackendIface, dev, nullptr};
}
ggml_backend_buffer_type_t dev_buffer_type(ggml_backend_dev_t) {
    return &kHostBuft;
}

bool dev_supports_op(ggml_backend_dev_t, const ggml_tensor* op) {
    switch (op->op) {
    case GGML_OP_NONE:
    case GGML_OP_RESHAPE:
    case GGML_OP_VIEW:
    case GGML_OP_PERMUTE:
    case GGML_OP_TRANSPOSE:
        return true;
    case GGML_OP_MUL_MAT: {
        const ggml_tensor* w = op->src[0];
        const ggml_tensor* x = op->src[1];
        D3d12MatmulDesc d;
        d.src0_type = w->type;
        d.ne00 = w->ne[0];
        d.ne01 = w->ne[1];
        d.ne02 = w->ne[2];
        d.ne03 = w->ne[3];
        d.ne10 = x->ne[0];
        d.ne11 = x->ne[1];
        d.ne12 = x->ne[2];
        d.ne13 = x->ne[3];
        d.src1_type = x->type;
        d.dst_type = op->type;
        d.src0_contiguous = ggml_is_contiguous(w);
        d.src1_contiguous = ggml_is_contiguous(x);
        d.src0_in_weight_buffer = w->buffer && w->buffer->buft == &kWeightsBuft;
        const bool ok = d3d12_mm_supported(d);
        // A refused matmul whose weight already sits in D3D12_Weights means a
        // placement the backend then cannot run — log the first few (D2b #309).
        static int logged = 0;
        if (!ok && w->buffer && w->buffer->buft == &kWeightsBuft && logged < 8) {
            ++logged;
            char msg[256];
            std::snprintf(msg, sizeof(msg),
                          "[xllama] d3d12: MUL_MAT refused: %s %lldx%lld x [%lld,%lld,%lld,%lld] "
                          "src1=%s dst=%s\n",
                          ggml_type_name(w->type), static_cast<long long>(w->ne[0]),
                          static_cast<long long>(w->ne[1]), static_cast<long long>(x->ne[0]),
                          static_cast<long long>(x->ne[1]), static_cast<long long>(x->ne[2]),
                          static_cast<long long>(x->ne[3]), ggml_type_name(x->type),
                          ggml_type_name(op->type));
            log_output(msg);
        }
        return ok;
    }
    case GGML_OP_GET_ROWS: {
        const ggml_tensor* w = op->src[0];
        const ggml_tensor* ids = op->src[1];
        D3d12GetRowsDesc d;
        d.src0_type = w->type;
        d.ne00 = w->ne[0];
        d.ne01 = w->ne[1];
        d.ne02 = w->ne[2];
        d.ne03 = w->ne[3];
        d.src1_type = ids->type;
        d.ne10 = ids->ne[0];
        d.ne11 = ids->ne[1];
        d.ne12 = ids->ne[2];
        d.dst_type = op->type;
        d.src0_contiguous = ggml_is_contiguous(w);
        d.src0_in_weight_buffer = w->buffer && w->buffer->buft == &kWeightsBuft;
        return d3d12_get_rows_supported(d);
    }
    default:
        return false;
    }
}
bool dev_supports_buft(ggml_backend_dev_t, ggml_backend_buffer_type_t buft) {
    return is_ours(buft);
}

const ggml_backend_device_i kDeviceIface = {
    /* .get_name             = */ dev_name,
    /* .get_description      = */ dev_description,
    /* .get_memory           = */ dev_memory,
    /* .get_type             = */ dev_type,
    /* .get_props            = */ dev_props,
    /* .init_backend         = */ dev_init_backend,
    /* .get_buffer_type      = */ dev_buffer_type,
    // The default buft doubles as the host buft: llama.cpp then allocates the
    // CPU backend's compute buffer in D3D12_Host, which supports_buft accepts,
    // so the scheduler stops copying every matmul input into a second buffer.
    /* .get_host_buffer_type = */ dev_buffer_type,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ dev_supports_op,
    /* .supports_buft        = */ dev_supports_buft,
    /* .offload_op           = */ nullptr,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

// --- Registry ---

const char* reg_name(ggml_backend_reg_t) {
    return "D3D12";
}
size_t reg_device_count(ggml_backend_reg_t) {
    return 1;
}
ggml_backend_dev_t reg_device(ggml_backend_reg_t, size_t index) {
    GGML_ASSERT(index == 0);
    return device_ptr();
}

// Weights live in the extra buft; llama.cpp lists it after the default one.
ggml_backend_buffer_type_t* dev_extra_bufts(ggml_backend_dev_t) {
    static ggml_backend_buffer_type_t list[] = {&kWeightsBuft, nullptr};
    return list;
}

void* reg_proc_address(ggml_backend_reg_t, const char* name) {
    if (std::strcmp(name, "ggml_backend_dev_get_extra_bufts") == 0)
        return reinterpret_cast<void*>(dev_extra_bufts);
    return nullptr;
}

const ggml_backend_reg_i kRegIface = {
    /* .get_name         = */ reg_name,
    /* .get_device_count = */ reg_device_count,
    /* .get_device       = */ reg_device,
    /* .get_proc_address = */ reg_proc_address,
};

ggml_backend_reg kReg = {
    /* .api_version = */ GGML_BACKEND_API_VERSION,
    /* .iface       = */ kRegIface,
    /* .context     = */ nullptr,
};

ggml_backend_device kDevice = {
    /* .iface   = */ kDeviceIface,
    /* .reg     = */ &kReg,
    /* .context = */ nullptr,
};

ggml_backend_dev_t device_ptr() {
    return &kDevice;
}

} // namespace

bool ggml_d3d12_register() {
    static std::once_flag once;
    static bool registered = false;
    std::call_once(once, [] {
        Gpu& g = gpu();
        if (!g.ok) {
            log_output(("[xllama] d3d12: backend unavailable: " + g.error + "\n").c_str());
            return;
        }
        kHostBuft.device = &kDevice;
        kWeightsBuft.device = &kDevice;
        ggml_backend_register(&kReg);
        registered = true;
    });
    return registered;
}

ggml_backend_buffer_type_t ggml_d3d12_weights_buft() {
    return ggml_d3d12_register() ? &kWeightsBuft : nullptr;
}

// --- Selftest ---

namespace {

struct SelftestCase {
    ggml_type type;
    const char* name;
    int n, k, ncols;
};

// Max |gpu - cpu| / max |cpu| against the CPU backend's own path: the
// vec_dot type's from_float, then ggml's vec_dot (#312, gate A).
double reference_rel_err(ggml_type t, const std::vector<std::uint8_t>& q, std::size_t row_bytes,
                         const std::vector<float>& x, const std::vector<float>& y, int n, int k,
                         int ncols) {
    ggml_cpu_init();
    const auto* wt = ggml_get_type_traits_cpu(t);
    std::vector<std::uint8_t> xq(ggml_row_size(wt->vec_dot_type, k));
    double max_ref = 0.0, max_diff = 0.0;
    for (int c = 0; c < ncols; ++c) {
        ggml_get_type_traits_cpu(wt->vec_dot_type)
            ->from_float(x.data() + static_cast<std::size_t>(c) * k, xq.data(), k);
        for (int r = 0; r < n; ++r) {
            float ref = 0.f;
            wt->vec_dot(k, &ref, 0, q.data() + static_cast<std::size_t>(r) * row_bytes, 0,
                        xq.data(), 0, 1);
            const double got = y[static_cast<std::size_t>(c) * n + r];
            max_ref = std::max(max_ref, std::fabs(static_cast<double>(ref)));
            max_diff = std::max(max_diff, std::fabs(static_cast<double>(ref) - got));
        }
    }
    return max_ref > 0.0 ? max_diff / max_ref : max_diff;
}

D3d12SelftestRow run_case(ggml_backend_t backend, const SelftestCase& sc) {
    D3d12SelftestRow row;
    row.type = sc.name;
    row.n = sc.n;
    row.k = sc.k;
    row.ncols = sc.ncols;

    ggml_init_params ip = {};
    ip.mem_size = 8 * ggml_tensor_overhead() + ggml_graph_overhead();
    ip.no_alloc = true;
    ggml_context* ctx_w = ggml_init(ip);
    ggml_context* ctx_x = ggml_init(ip);
    ggml_context* ctx_g = ggml_init(ip);
    ggml_tensor* w = ggml_new_tensor_2d(ctx_w, sc.type, sc.k, sc.n);
    ggml_tensor* x = ggml_new_tensor_2d(ctx_x, GGML_TYPE_F32, sc.k, sc.ncols);
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors_from_buft(ctx_w, &kWeightsBuft);
    ggml_backend_buffer_t xbuf = ggml_backend_alloc_ctx_tensors_from_buft(ctx_x, &kHostBuft);
    ggml_gallocr_t galloc = nullptr;

    auto cleanup = [&] {
        if (galloc)
            ggml_gallocr_free(galloc);
        if (wbuf)
            ggml_backend_buffer_free(wbuf);
        if (xbuf)
            ggml_backend_buffer_free(xbuf);
        ggml_free(ctx_g);
        ggml_free(ctx_x);
        ggml_free(ctx_w);
    };
    if (!wbuf || !xbuf) {
        row.error = "buffer allocation failed";
        cleanup();
        return row;
    }

    std::mt19937 rng(1234u + static_cast<unsigned>(sc.n + sc.k + sc.ncols));
    std::uniform_real_distribution<float> uni(-1.f, 1.f);
    std::vector<float> wf(static_cast<std::size_t>(sc.n) * sc.k);
    for (float& v : wf)
        v = uni(rng);
    std::vector<std::uint8_t> q(ggml_nbytes(w));
    ggml_quantize_chunk(sc.type, wf.data(), q.data(), 0, sc.n, sc.k, nullptr);
    ggml_backend_tensor_set(w, q.data(), 0, q.size());
    std::vector<float> xf(static_cast<std::size_t>(sc.k) * sc.ncols);
    for (float& v : xf)
        v = uni(rng);
    ggml_backend_tensor_set(x, xf.data(), 0, xf.size() * sizeof(float));

    ggml_tensor* y = ggml_mul_mat(ctx_g, w, x);
    ggml_cgraph* gf = ggml_new_graph(ctx_g);
    ggml_build_forward_expand(gf, y);
    galloc = ggml_gallocr_new(&kHostBuft);
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        row.error = "graph allocation failed";
        cleanup();
        return row;
    }
    // Warm-up, then the timed run.
    ggml_status st = ggml_backend_graph_compute(backend, gf);
    if (st == GGML_STATUS_SUCCESS)
        st = ggml_backend_graph_compute(backend, gf);
    row.d3d12_ran = st == GGML_STATUS_SUCCESS;
    if (!row.d3d12_ran) {
        row.error = "graph_compute failed";
        cleanup();
        return row;
    }
    row.gpu_ms = gpu().last_gpu_ms;
    row.packed_gbs = row.gpu_ms > 0.0 ? static_cast<double>(q.size()) / 1e6 / row.gpu_ms : 0.0;
    std::vector<float> yf(static_cast<std::size_t>(sc.n) * sc.ncols);
    ggml_backend_tensor_get(y, yf.data(), 0, yf.size() * sizeof(float));
    row.rel_err = reference_rel_err(sc.type, q, w->nb[1], xf, yf, sc.n, sc.k, sc.ncols);
    row.ok = row.rel_err <= kD3d12SelftestRelTol;
    if (!row.ok)
        row.error = "mismatch vs the CPU backend (q8 vec_dot)";
    cleanup();
    return row;
}

} // namespace

void run_d3d12_selftest(std::vector<D3d12SelftestRow>* out) {
    if (!out)
        return;
    if (!ggml_d3d12_register()) {
        D3d12SelftestRow r;
        r.type = "-";
        r.error = gpu().error.empty() ? "d3d12 backend unavailable" : gpu().error;
        out->push_back(std::move(r));
        return;
    }
    ggml_backend_t backend = dev_init_backend(&kDevice, nullptr);
    // Shapes of the D2 target models; q6_k 2048x11008 has 2-byte-aligned rows.
    const SelftestCase cases[] = {
        {GGML_TYPE_Q4_0, "q4_0", 8192, 2048, 1},   {GGML_TYPE_Q4_0, "q4_0", 2048, 8192, 1},
        {GGML_TYPE_Q4_K, "q4_k", 11008, 2048, 1},  {GGML_TYPE_Q4_K, "q4_k", 2048, 11008, 1},
        {GGML_TYPE_Q6_K, "q6_k", 2048, 11008, 1},  {GGML_TYPE_Q6_K, "q6_k", 65536, 1024, 1},
        {GGML_TYPE_Q4_0, "q4_0", 1024, 1024, 7},   {GGML_TYPE_Q4_K, "q4_k", 1024, 1024, 7},
        {GGML_TYPE_Q6_K, "q6_k", 1024, 1024, 7},   {GGML_TYPE_Q4_0, "q4_0", 1024, 1024, 512},
        {GGML_TYPE_Q4_K, "q4_k", 1024, 1024, 512}, {GGML_TYPE_Q6_K, "q6_k", 1024, 1024, 512},
    };
    for (const auto& sc : cases)
        out->push_back(run_case(backend, sc));
    ggml_backend_free(backend);
}

} // namespace xllama

    #endif // _WIN32

#endif // XLLAMA_USE_LLAMA
