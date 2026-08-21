// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include "xllama/gpugemv.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#if defined(_WIN32)
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #include <Windows.h>
    #include <d3d12.h>
    #include <dxgi1_4.h>
    #include <wrl/client.h>

    #include "gpugemv_q4k_dxil.h"
    #include "gpugemv_q4k_wave32_dxil.h"
    #include "xllama/d3d12_dyn.h"

using Microsoft::WRL::ComPtr;
#endif

namespace xllama {

// --- half <-> float (IEEE 754, round-to-nearest-even simplified) ---

std::uint16_t gpugemv_float_to_half(float f) {
    std::uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    const std::uint32_t sign = (x >> 16) & 0x8000u;
    std::int32_t exp = static_cast<std::int32_t>((x >> 23) & 0xff) - 127 + 15;
    std::uint32_t mant = x & 0x7fffffu;
    if (exp <= 0) {
        if (exp < -10)
            return static_cast<std::uint16_t>(sign);
        mant |= 0x800000u;
        const int shift = 14 - exp;
        std::uint32_t m = mant >> shift;
        if ((mant >> (shift - 1)) & 1u)
            m += 1;
        return static_cast<std::uint16_t>(sign | m);
    }
    if (exp >= 31) {
        // Inf / NaN
        if (mant)
            return static_cast<std::uint16_t>(sign | 0x7e00u);
        return static_cast<std::uint16_t>(sign | 0x7c00u);
    }
    std::uint32_t h = sign | (static_cast<std::uint32_t>(exp) << 10) | (mant >> 13);
    if (mant & 0x1000u)
        h += 1;
    return static_cast<std::uint16_t>(h);
}

float gpugemv_half_to_float(std::uint16_t h) {
    const std::uint32_t sign = (static_cast<std::uint32_t>(h & 0x8000u)) << 16;
    std::uint32_t exp = (h >> 10) & 0x1fu;
    std::uint32_t mant = h & 0x3ffu;
    std::uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign;
        } else {
            exp = 1;
            while ((mant & 0x400u) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x3ffu;
            f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7f800000u | (mant << 13);
    } else {
        f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float out;
    std::memcpy(&out, &f, sizeof(out));
    return out;
}

void gpugemv_get_scale_min_k4(int j, const std::uint8_t* q, std::uint8_t* d, std::uint8_t* m) {
    // Identical to ggml get_scale_min_k4.
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

void gpugemv_dequant_block(const GpugemvQ4KBlock& b, float* y256) {
    const float d = gpugemv_half_to_float(b.d);
    const float minv = gpugemv_half_to_float(b.dmin);
    const std::uint8_t* q = b.qs;
    int is = 0;
    float* y = y256;
    for (int j = 0; j < kGpugemvQK; j += 64) {
        std::uint8_t sc, m;
        gpugemv_get_scale_min_k4(is + 0, b.scales, &sc, &m);
        const float d1 = d * static_cast<float>(sc);
        const float m1 = minv * static_cast<float>(m);
        gpugemv_get_scale_min_k4(is + 1, b.scales, &sc, &m);
        const float d2 = d * static_cast<float>(sc);
        const float m2 = minv * static_cast<float>(m);
        for (int l = 0; l < 32; ++l)
            *y++ = d1 * static_cast<float>(q[l] & 0xF) - m1;
        for (int l = 0; l < 32; ++l)
            *y++ = d2 * static_cast<float>(q[l] >> 4) - m2;
        q += 32;
        is += 2;
    }
}

void gpugemv_fill_weights(GpugemvQ4KBlock* blocks, int n, int k) {
    if (!blocks || n <= 0 || k <= 0 || (k % kGpugemvQK) != 0)
        return;
    const int nb = k / kGpugemvQK;
    for (int row = 0; row < n; ++row) {
        for (int bi = 0; bi < nb; ++bi) {
            GpugemvQ4KBlock& b = blocks[row * nb + bi];
            // Mild scales so dequant stays well-conditioned.
            b.d = gpugemv_float_to_half(0.05f + 0.001f * static_cast<float>((row + bi) % 17));
            b.dmin = gpugemv_float_to_half(0.01f);
            for (int s = 0; s < kGpugemvScaleBytes; ++s)
                b.scales[s] = static_cast<std::uint8_t>((s * 7 + row + bi) & 0x3f);
            // Keep high 2 bits of scales consistent with packing (lower 6 used).
            for (int s = 0; s < kGpugemvScaleBytes; ++s)
                b.scales[s] = static_cast<std::uint8_t>(b.scales[s] & 0x3f);
            for (int q = 0; q < kGpugemvQsBytes; ++q) {
                const std::uint8_t lo = static_cast<std::uint8_t>((q + bi + row) & 0xF);
                const std::uint8_t hi = static_cast<std::uint8_t>((q * 3 + row) & 0xF);
                b.qs[q] = static_cast<std::uint8_t>(lo | (hi << 4));
            }
        }
    }
}

void gpugemv_fill_x(float* x, int k) {
    if (!x || k <= 0)
        return;
    for (int i = 0; i < k; ++i) {
        // Small activations; deterministic.
        const std::uint32_t u = static_cast<std::uint32_t>(i) * 0x9e3779b9u + 1u;
        x[i] = (static_cast<float>(u & 0xFFu) / 255.f) * 0.1f - 0.05f;
    }
}

void gpugemv_cpu_ref(const GpugemvQ4KBlock* blocks, const float* x, float* y, int n, int k) {
    if (!blocks || !x || !y || n <= 0 || k <= 0 || (k % kGpugemvQK) != 0)
        return;
    const int nb = k / kGpugemvQK;
    std::vector<float> w(static_cast<std::size_t>(kGpugemvQK));
    for (int row = 0; row < n; ++row) {
        float acc = 0.f;
        for (int bi = 0; bi < nb; ++bi) {
            gpugemv_dequant_block(blocks[row * nb + bi], w.data());
            const float* xv = x + bi * kGpugemvQK;
            for (int l = 0; l < kGpugemvQK; ++l)
                acc += w[static_cast<std::size_t>(l)] * xv[l];
        }
        y[row] = acc;
    }
}

std::uint32_t gpugemv_checksum_floats(const float* data, std::size_t n) {
    std::uint32_t x = 0;
    if (!data)
        return 0;
    for (std::size_t i = 0; i < n; ++i) {
        std::uint32_t bits;
        std::memcpy(&bits, data + i, sizeof(bits));
        x ^= bits;
    }
    return x;
}

float gpugemv_max_abs_err(const float* a, const float* b, std::size_t n) {
    float m = 0.f;
    if (!a || !b)
        return std::numeric_limits<float>::infinity();
    for (std::size_t i = 0; i < n; ++i)
        m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

const char* gpugemv_csv_header() {
    return "kernel,n,k,iterations,run_index,packed_mb,packed_gbs,packed_gbs_cpu,packed_gbs_h61,"
           "max_abs_err,checksum_ok,d3d12_ran,gpu_timestamp,wave_ops,checksum,expected,host,date,"
           "error\n";
}

std::string format_gpugemv_row(const GpugemvResult& r, const char* host_label) {
    char date_buf[32];
    std::time_t now = std::time(nullptr);
    std::tm* tm_utc = std::gmtime(&now);
    std::strftime(date_buf, sizeof(date_buf), "%Y-%m-%dT%H:%M:%SZ", tm_utc);

    std::string err = r.error_msg;
    for (char& c : err) {
        if (c == ',' || c == '\n' || c == '\r')
            c = ' ';
    }
    if (err.empty())
        err = "-";

    char buf[768];
    std::snprintf(
        buf, sizeof(buf), "%s,%d,%d,%d,%d,%.3f,%.2f,%.2f,%.2f,%.6g,%d,%d,%d,%d,%u,%u,%s,%s,%s\n",
        gpugemv_kernel_name(r.kernel), r.n, r.k, r.iterations, r.run_index,
        static_cast<double>(r.packed_bytes) / (1024.0 * 1024.0), r.packed_gbs, r.packed_gbs_cpu,
        r.packed_gbs_h61, static_cast<double>(r.max_abs_err), r.checksum_ok ? 1 : 0,
        r.d3d12_ran ? 1 : 0, r.gpu_timestamp ? 1 : 0, r.wave_ops ? 1 : 0, r.y_checksum,
        r.expected_y_checksum, host_label ? host_label : "unknown", date_buf, err.c_str());
    return std::string(buf);
}

namespace {

GpugemvResult gpugemv_median_summary(const std::vector<GpugemvResult>& rows) {
    if (rows.empty())
        return {};
    if (rows.size() == 1) {
        GpugemvResult r = rows[0];
        r.run_index = 0;
        return r;
    }
    std::vector<std::size_t> idx(rows.size());
    for (std::size_t i = 0; i < rows.size(); ++i)
        idx[i] = i;
    std::sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) {
        return rows[a].packed_gbs < rows[b].packed_gbs;
    });
    GpugemvResult r = rows[idx[idx.size() / 2]];
    r.run_index = 0;
    r.iterations = rows[0].iterations;
    return r;
}

void gpugemv_host_tiny_ref(GpugemvResult& r, int n, int k) {
    const int tn = std::min(n, 8);
    const int tk = std::min(k, kGpugemvQK);
    if (tn <= 0 || tk <= 0 || (tk % kGpugemvQK) != 0)
        return;
    std::vector<GpugemvQ4KBlock> w(static_cast<std::size_t>(tn * (tk / kGpugemvQK)));
    std::vector<float> x(static_cast<std::size_t>(tk));
    std::vector<float> y(static_cast<std::size_t>(tn));
    gpugemv_fill_weights(w.data(), tn, tk);
    gpugemv_fill_x(x.data(), tk);
    gpugemv_cpu_ref(w.data(), x.data(), y.data(), tn, tk);
    r.expected_y_checksum = gpugemv_checksum_floats(y.data(), static_cast<std::size_t>(tn));
}

} // namespace

#if !defined(_WIN32)

void measure_gpugemv_each(int n, int k, int iterations, GpugemvKernel kernel,
                          std::vector<GpugemvResult>* out) {
    if (!out)
        return;
    GpugemvResult r;
    r.n = n;
    r.k = k;
    r.iterations = iterations < 1 ? 1 : iterations;
    r.run_index = 1;
    r.kernel = kernel;
    r.packed_bytes = gpugemv_packed_bytes(n, k);
    r.packed_gbs = 0.0;
    r.d3d12_ran = false;
    r.error_msg = "d3d12 unavailable on this platform";
    if (r.packed_bytes == 0) {
        r.error_msg = "invalid n/k (need k%256==0, n>0,k>0)";
        out->push_back(std::move(r));
        return;
    }
    gpugemv_host_tiny_ref(r, n, k);
    out->push_back(std::move(r));
}

GpugemvResult measure_gpugemv(int n, int k, int iterations, GpugemvKernel kernel) {
    std::vector<GpugemvResult> rows;
    measure_gpugemv_each(n, k, iterations, kernel, &rows);
    return gpugemv_median_summary(rows);
}

#else

namespace {

void throw_if_failed(HRESULT hr, const char* what, GpugemvResult& r) {
    if (SUCCEEDED(hr))
        return;
    char b[160];
    std::snprintf(b, sizeof(b), "%s hr=0x%08lx", what, static_cast<unsigned long>(hr));
    r.error_msg = b;
}

ComPtr<ID3D12RootSignature> create_root_sig(ID3D12Device* device, GpugemvResult& r) {
    // b0 CBV, t0 weights SRV, t1 x SRV, u0 y UAV
    D3D12_DESCRIPTOR_RANGE ranges[4] = {};
    ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_CBV;
    ranges[0].NumDescriptors = 1;
    ranges[0].BaseShaderRegister = 0;
    ranges[0].OffsetInDescriptorsFromTableStart = 0;

    ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    ranges[1].NumDescriptors = 1;
    ranges[1].BaseShaderRegister = 0;
    ranges[1].OffsetInDescriptorsFromTableStart = 1;

    ranges[2].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    ranges[2].NumDescriptors = 1;
    ranges[2].BaseShaderRegister = 1;
    ranges[2].OffsetInDescriptorsFromTableStart = 2;

    ranges[3].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    ranges[3].NumDescriptors = 1;
    ranges[3].BaseShaderRegister = 0;
    ranges[3].OffsetInDescriptorsFromTableStart = 3;

    D3D12_ROOT_PARAMETER param = {};
    param.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    param.DescriptorTable.NumDescriptorRanges = 4;
    param.DescriptorTable.pDescriptorRanges = ranges;
    param.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    D3D12_ROOT_SIGNATURE_DESC desc = {};
    desc.NumParameters = 1;
    desc.pParameters = &param;
    desc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

    ComPtr<ID3DBlob> sig;
    ComPtr<ID3DBlob> err;
    auto serialize = d3d12_dyn::SerializeRootSignature();
    if (!serialize) {
        r.error_msg = "D3D12SerializeRootSignature not available";
        return {};
    }
    HRESULT hr = serialize(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &sig, &err);
    if (FAILED(hr)) {
        r.error_msg = "D3D12SerializeRootSignature failed";
        return {};
    }
    ComPtr<ID3D12RootSignature> root;
    hr = device->CreateRootSignature(0, sig->GetBufferPointer(), sig->GetBufferSize(),
                                     IID_PPV_ARGS(&root));
    if (FAILED(hr)) {
        throw_if_failed(hr, "CreateRootSignature", r);
        return {};
    }
    return root;
}

ComPtr<ID3D12Resource> create_buffer(ID3D12Device* device, UINT64 bytes, D3D12_HEAP_TYPE heap,
                                     D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES state,
                                     GpugemvResult& r, const char* name) {
    D3D12_HEAP_PROPERTIES hp = {};
    hp.Type = heap;
    D3D12_RESOURCE_DESC rd = {};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width = bytes;
    rd.Height = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    rd.Flags = flags;
    ComPtr<ID3D12Resource> res;
    HRESULT hr = device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, state, nullptr,
                                                 IID_PPV_ARGS(&res));
    if (FAILED(hr)) {
        char b[128];
        std::snprintf(b, sizeof(b), "CreateCommittedResource %s", name);
        throw_if_failed(hr, b, r);
        return {};
    }
    return res;
}

double packed_gbs_from_sec(std::size_t bytes, double sec) {
    if (sec <= 0.0)
        return 0.0;
    return static_cast<double>(bytes) / 1e9 / sec;
}

struct FenceEvent {
    HANDLE h = nullptr;
    ~FenceEvent() {
        if (h)
            CloseHandle(h);
    }
};

void fill_checksum(GpugemvResult& r, const float* host_y, const float* gpu_y, int n) {
    r.y_checksum = gpugemv_checksum_floats(gpu_y, static_cast<std::size_t>(n));
    r.max_abs_err = gpugemv_max_abs_err(host_y, gpu_y, static_cast<std::size_t>(n));
    r.checksum_ok =
        (r.y_checksum == r.expected_y_checksum) || (r.max_abs_err <= kGpugemvMaxAbsErrTol);
    if (r.max_abs_err <= kGpugemvMaxAbsErrTol)
        r.checksum_ok = true;
    if (!r.checksum_ok && r.error_msg.empty())
        r.error_msg = "GEMV residual or checksum mismatch";
    if (r.checksum_ok)
        r.error_msg.clear();
}

} // namespace

void measure_gpugemv_each(int n, int k, int iterations, GpugemvKernel kernel,
                          std::vector<GpugemvResult>* out) {
    if (!out)
        return;
    auto push_err = [&](const char* msg) {
        GpugemvResult r;
        r.n = n;
        r.k = k;
        r.iterations = iterations < 1 ? 1 : iterations;
        r.run_index = 1;
        r.kernel = kernel;
        r.packed_bytes = gpugemv_packed_bytes(n, k);
        r.error_msg = msg ? msg : "gpugemv failed";
        out->push_back(std::move(r));
    };

    GpugemvResult seed;
    seed.n = n;
    seed.k = k;
    seed.iterations = iterations < 1 ? 1 : iterations;
    seed.kernel = kernel;
    seed.packed_bytes = gpugemv_packed_bytes(n, k);
    if (seed.packed_bytes == 0) {
        push_err("invalid n/k (need k%256==0, n>0,k>0)");
        return;
    }

    const GpugemvDispatch plan = gpugemv_plan_dispatch(n, kernel);
    if (plan.groups_x < 1u || plan.groups_x > 65535u) {
        push_err("Dispatch groups exceed 65535; reduce N");
        return;
    }

    const int nb = k / kGpugemvQK;
    const std::size_t n_blocks = static_cast<std::size_t>(n) * static_cast<std::size_t>(nb);
    std::vector<GpugemvQ4KBlock> host_w(n_blocks);
    std::vector<float> host_x(static_cast<std::size_t>(k));
    std::vector<float> host_y(static_cast<std::size_t>(n));
    gpugemv_fill_weights(host_w.data(), n, k);
    gpugemv_fill_x(host_x.data(), k);
    gpugemv_cpu_ref(host_w.data(), host_x.data(), host_y.data(), n, k);
    seed.expected_y_checksum = gpugemv_checksum_floats(host_y.data(), static_cast<std::size_t>(n));

    ComPtr<IDXGIFactory4> factory;
    HRESULT hr = CreateDXGIFactory1(IID_PPV_ARGS(&factory));
    if (FAILED(hr)) {
        throw_if_failed(hr, "CreateDXGIFactory1", seed);
        out->push_back(std::move(seed));
        return;
    }

    auto create_device = d3d12_dyn::CreateDevice();
    if (!create_device) {
        push_err("D3D12CreateDevice not available");
        return;
    }

    ComPtr<IDXGIAdapter1> adapter;
    ComPtr<ID3D12Device> device;
    for (UINT i = 0; factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i) {
        DXGI_ADAPTER_DESC1 ad = {};
        adapter->GetDesc1(&ad);
        if (ad.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)
            continue;
        hr = create_device(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device));
        if (SUCCEEDED(hr))
            break;
        device.Reset();
        adapter.Reset();
    }
    if (!device) {
        hr = create_device(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device));
        if (FAILED(hr)) {
            throw_if_failed(hr, "D3D12CreateDevice", seed);
            out->push_back(std::move(seed));
            return;
        }
    }

    D3D12_FEATURE_DATA_D3D12_OPTIONS1 opt1 = {};
    device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, &opt1, sizeof(opt1));
    (void)opt1; // PR 1 ships LDS-red only; wave_ops stays 0.

    D3D12_COMMAND_QUEUE_DESC qd = {};
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    ComPtr<ID3D12CommandQueue> queue;
    hr = device->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue));
    if (FAILED(hr)) {
        throw_if_failed(hr, "CreateCommandQueue", seed);
        out->push_back(std::move(seed));
        return;
    }

    ComPtr<ID3D12CommandAllocator> alloc;
    hr = device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc));
    if (FAILED(hr)) {
        throw_if_failed(hr, "CreateCommandAllocator", seed);
        out->push_back(std::move(seed));
        return;
    }

    ComPtr<ID3D12GraphicsCommandList> cl;
    hr = device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.Get(), nullptr,
                                   IID_PPV_ARGS(&cl));
    if (FAILED(hr)) {
        throw_if_failed(hr, "CreateCommandList", seed);
        out->push_back(std::move(seed));
        return;
    }

    ComPtr<ID3D12RootSignature> root = create_root_sig(device.Get(), seed);
    if (!root) {
        out->push_back(std::move(seed));
        return;
    }

    D3D12_COMPUTE_PIPELINE_STATE_DESC pso_desc = {};
    pso_desc.pRootSignature = root.Get();
    if (kernel == GpugemvKernel::Naive) {
        pso_desc.CS.pShaderBytecode = kGpugemvQ4kDxil;
        pso_desc.CS.BytecodeLength = kGpugemvQ4kDxilSize;
    } else {
        pso_desc.CS.pShaderBytecode = kGpugemvQ4kWave32Dxil;
        pso_desc.CS.BytecodeLength = kGpugemvQ4kWave32DxilSize;
    }
    ComPtr<ID3D12PipelineState> pso;
    hr = device->CreateComputePipelineState(&pso_desc, IID_PPV_ARGS(&pso));
    if (FAILED(hr)) {
        throw_if_failed(hr, "CreateComputePipelineState", seed);
        out->push_back(std::move(seed));
        return;
    }

    const UINT64 w_bytes = static_cast<UINT64>(seed.packed_bytes);
    const UINT64 x_bytes = static_cast<UINT64>(k) * sizeof(float);
    const UINT64 y_bytes = static_cast<UINT64>(n) * sizeof(float);

    auto w_default =
        create_buffer(device.Get(), w_bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_NONE,
                      D3D12_RESOURCE_STATE_COPY_DEST, seed, "w_default");
    if (!w_default) {
        out->push_back(std::move(seed));
        return;
    }
    auto w_upload =
        create_buffer(device.Get(), w_bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE,
                      D3D12_RESOURCE_STATE_GENERIC_READ, seed, "w_upload");
    if (!w_upload) {
        out->push_back(std::move(seed));
        return;
    }
    auto x_default =
        create_buffer(device.Get(), x_bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_NONE,
                      D3D12_RESOURCE_STATE_COPY_DEST, seed, "x_default");
    if (!x_default) {
        out->push_back(std::move(seed));
        return;
    }
    auto x_upload =
        create_buffer(device.Get(), x_bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE,
                      D3D12_RESOURCE_STATE_GENERIC_READ, seed, "x_upload");
    if (!x_upload) {
        out->push_back(std::move(seed));
        return;
    }
    auto y_default = create_buffer(device.Get(), y_bytes, D3D12_HEAP_TYPE_DEFAULT,
                                   D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                   D3D12_RESOURCE_STATE_UNORDERED_ACCESS, seed, "y_default");
    if (!y_default) {
        out->push_back(std::move(seed));
        return;
    }
    auto y_readback =
        create_buffer(device.Get(), y_bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE,
                      D3D12_RESOURCE_STATE_COPY_DEST, seed, "y_readback");
    if (!y_readback) {
        out->push_back(std::move(seed));
        return;
    }
    auto cb_upload =
        create_buffer(device.Get(), 256, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE,
                      D3D12_RESOURCE_STATE_GENERIC_READ, seed, "cb");
    if (!cb_upload) {
        out->push_back(std::move(seed));
        return;
    }

    {
        void* mapped = nullptr;
        hr = w_upload->Map(0, nullptr, &mapped);
        if (FAILED(hr)) {
            throw_if_failed(hr, "Map w", seed);
            out->push_back(std::move(seed));
            return;
        }
        std::memcpy(mapped, host_w.data(), static_cast<std::size_t>(w_bytes));
        w_upload->Unmap(0, nullptr);
    }
    {
        void* mapped = nullptr;
        hr = x_upload->Map(0, nullptr, &mapped);
        if (FAILED(hr)) {
            throw_if_failed(hr, "Map x", seed);
            out->push_back(std::move(seed));
            return;
        }
        std::memcpy(mapped, host_x.data(), static_cast<std::size_t>(x_bytes));
        x_upload->Unmap(0, nullptr);
    }
    {
        struct Params {
            std::uint32_t n;
            std::uint32_t k;
            std::uint32_t nb;
            std::uint32_t pad;
        } params{};
        params.n = static_cast<std::uint32_t>(n);
        params.k = static_cast<std::uint32_t>(k);
        params.nb = static_cast<std::uint32_t>(nb);
        void* mapped = nullptr;
        hr = cb_upload->Map(0, nullptr, &mapped);
        if (FAILED(hr)) {
            throw_if_failed(hr, "Map cb", seed);
            out->push_back(std::move(seed));
            return;
        }
        std::memcpy(mapped, &params, sizeof(params));
        cb_upload->Unmap(0, nullptr);
    }

    D3D12_DESCRIPTOR_HEAP_DESC heap_desc = {};
    heap_desc.NumDescriptors = 4;
    heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    ComPtr<ID3D12DescriptorHeap> heap;
    hr = device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&heap));
    if (FAILED(hr)) {
        throw_if_failed(hr, "CreateDescriptorHeap", seed);
        out->push_back(std::move(seed));
        return;
    }
    const UINT incr =
        device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    D3D12_CPU_DESCRIPTOR_HANDLE cpu = heap->GetCPUDescriptorHandleForHeapStart();

    D3D12_CONSTANT_BUFFER_VIEW_DESC cbv = {};
    cbv.BufferLocation = cb_upload->GetGPUVirtualAddress();
    cbv.SizeInBytes = 256;
    device->CreateConstantBufferView(&cbv, cpu);

    const bool raw_srv = (kernel != GpugemvKernel::Naive);

    D3D12_CPU_DESCRIPTOR_HANDLE cpu_w = cpu;
    cpu_w.ptr += incr;
    D3D12_SHADER_RESOURCE_VIEW_DESC srv_w = {};
    srv_w.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    srv_w.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv_w.Buffer.FirstElement = 0;
    if (raw_srv) {
        srv_w.Format = DXGI_FORMAT_R32_TYPELESS;
        srv_w.Buffer.NumElements = static_cast<UINT>(w_bytes / 4);
        srv_w.Buffer.StructureByteStride = 0;
        srv_w.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
    } else {
        srv_w.Format = DXGI_FORMAT_UNKNOWN;
        srv_w.Buffer.NumElements = static_cast<UINT>(n_blocks);
        srv_w.Buffer.StructureByteStride = static_cast<UINT>(kGpugemvBlockBytes);
    }
    device->CreateShaderResourceView(w_default.Get(), &srv_w, cpu_w);

    D3D12_CPU_DESCRIPTOR_HANDLE cpu_x = cpu;
    cpu_x.ptr += 2 * static_cast<SIZE_T>(incr);
    D3D12_SHADER_RESOURCE_VIEW_DESC srv_x = {};
    srv_x.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    srv_x.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv_x.Buffer.FirstElement = 0;
    if (raw_srv) {
        srv_x.Format = DXGI_FORMAT_R32_TYPELESS;
        srv_x.Buffer.NumElements = static_cast<UINT>(x_bytes / 4);
        srv_x.Buffer.StructureByteStride = 0;
        srv_x.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
    } else {
        srv_x.Format = DXGI_FORMAT_UNKNOWN;
        srv_x.Buffer.NumElements = static_cast<UINT>(k);
        srv_x.Buffer.StructureByteStride = sizeof(float);
    }
    device->CreateShaderResourceView(x_default.Get(), &srv_x, cpu_x);

    D3D12_CPU_DESCRIPTOR_HANDLE cpu_y = cpu;
    cpu_y.ptr += 3 * static_cast<SIZE_T>(incr);
    D3D12_UNORDERED_ACCESS_VIEW_DESC uav = {};
    uav.Format = DXGI_FORMAT_UNKNOWN;
    uav.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    uav.Buffer.FirstElement = 0;
    uav.Buffer.NumElements = static_cast<UINT>(n);
    uav.Buffer.StructureByteStride = sizeof(float);
    device->CreateUnorderedAccessView(y_default.Get(), nullptr, &uav, cpu_y);

    ComPtr<ID3D12Fence> fence;
    hr = device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
    if (FAILED(hr)) {
        throw_if_failed(hr, "CreateFence", seed);
        out->push_back(std::move(seed));
        return;
    }
    FenceEvent fence_event;
    fence_event.h = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!fence_event.h) {
        push_err("CreateEventW failed");
        return;
    }
    UINT64 fence_value = 0;
    auto wait_gpu = [&]() -> bool {
        const UINT64 v = ++fence_value;
        if (FAILED(queue->Signal(fence.Get(), v))) {
            seed.error_msg = "Signal failed";
            return false;
        }
        if (fence->GetCompletedValue() < v) {
            fence->SetEventOnCompletion(v, fence_event.h);
            WaitForSingleObject(fence_event.h, INFINITE);
        }
        return true;
    };

    {
        cl->CopyResource(w_default.Get(), w_upload.Get());
        cl->CopyResource(x_default.Get(), x_upload.Get());
        D3D12_RESOURCE_BARRIER bars[2] = {};
        bars[0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        bars[0].Transition.pResource = w_default.Get();
        bars[0].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
        bars[0].Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        bars[0].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        bars[1] = bars[0];
        bars[1].Transition.pResource = x_default.Get();
        cl->ResourceBarrier(2, bars);
        cl->Close();
        ID3D12CommandList* lists[] = {cl.Get()};
        queue->ExecuteCommandLists(1, lists);
        if (!wait_gpu()) {
            out->push_back(std::move(seed));
            return;
        }
    }

    bool use_ts = true;
    ComPtr<ID3D12QueryHeap> qh;
    D3D12_QUERY_HEAP_DESC qhd = {};
    qhd.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
    qhd.Count = 2;
    hr = device->CreateQueryHeap(&qhd, IID_PPV_ARGS(&qh));
    UINT64 ts_freq = 0;
    if (FAILED(hr) || FAILED(queue->GetTimestampFrequency(&ts_freq)) || ts_freq == 0) {
        use_ts = false;
        qh.Reset();
    }
    ComPtr<ID3D12Resource> ts_readback;
    if (use_ts) {
        ts_readback =
            create_buffer(device.Get(), 256, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE,
                          D3D12_RESOURCE_STATE_COPY_DEST, seed, "ts_readback");
        if (!ts_readback)
            use_ts = false;
    }

    auto record_list_a = [&](bool timestamps) -> bool {
        alloc->Reset();
        cl->Reset(alloc.Get(), pso.Get());
        cl->SetComputeRootSignature(root.Get());
        ID3D12DescriptorHeap* heaps[] = {heap.Get()};
        cl->SetDescriptorHeaps(1, heaps);
        cl->SetComputeRootDescriptorTable(0, heap->GetGPUDescriptorHandleForHeapStart());
        cl->SetPipelineState(pso.Get());
        if (timestamps && qh)
            cl->EndQuery(qh.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0);
        cl->Dispatch(plan.groups_x, 1, 1);
        D3D12_RESOURCE_BARRIER uavb = {};
        uavb.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        uavb.UAV.pResource = y_default.Get();
        cl->ResourceBarrier(1, &uavb);
        if (timestamps && qh)
            cl->EndQuery(qh.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 1);
        cl->Close();
        return true;
    };

    auto execute_wait = [&](double* cpu_sec) -> bool {
        const auto t0 = std::chrono::steady_clock::now();
        ID3D12CommandList* lists[] = {cl.Get()};
        queue->ExecuteCommandLists(1, lists);
        if (!wait_gpu())
            return false;
        if (cpu_sec)
            *cpu_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        return true;
    };

    // Warmup: one untimed list A.
    if (!record_list_a(/*timestamps=*/false) || !execute_wait(nullptr)) {
        out->push_back(std::move(seed));
        return;
    }

    std::vector<float> gpu_y(static_cast<std::size_t>(n));
    for (int it = 0; it < seed.iterations; ++it) {
        GpugemvResult row = seed;
        row.run_index = it + 1;
        row.wave_ops = false;
        row.gpu_timestamp = false;

        auto run_a = [&](double* cpu_a, double* gpu_sec) -> bool {
            if (!record_list_a(use_ts) || !execute_wait(cpu_a))
                return false;
            *gpu_sec = -1.0;
            if (!use_ts || !qh || !ts_readback)
                return true;
            alloc->Reset();
            cl->Reset(alloc.Get(), nullptr);
            cl->ResolveQueryData(qh.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, ts_readback.Get(), 0);
            cl->Close();
            ID3D12CommandList* lists[] = {cl.Get()};
            queue->ExecuteCommandLists(1, lists);
            if (!wait_gpu())
                return false;
            void* mapped = nullptr;
            D3D12_RANGE range = {0, sizeof(std::uint64_t) * 2};
            if (FAILED(ts_readback->Map(0, &range, &mapped)) || !mapped)
                return true;
            const auto* ts = static_cast<const std::uint64_t*>(mapped);
            const std::uint64_t t0 = ts[0];
            const std::uint64_t t1 = ts[1];
            ts_readback->Unmap(0, nullptr);
            if (t1 > t0 && ts_freq > 0)
                *gpu_sec = static_cast<double>(t1 - t0) / static_cast<double>(ts_freq);
            return true;
        };

        double cpu_a = 0.0;
        double gpu_sec = -1.0;
        if (!run_a(&cpu_a, &gpu_sec)) {
            out->push_back(std::move(seed));
            return;
        }
        if (use_ts && gpu_sec <= 0.0) {
            if (!run_a(&cpu_a, &gpu_sec)) {
                out->push_back(std::move(seed));
                return;
            }
            if (gpu_sec <= 0.0) {
                use_ts = false;
                row.error_msg = "timestamp query discarded (zero or inverted dt)";
            }
        }

        double cpu_b = 0.0;
        {
            alloc->Reset();
            cl->Reset(alloc.Get(), nullptr);
            D3D12_RESOURCE_BARRIER b = {};
            b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            b.Transition.pResource = y_default.Get();
            b.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            b.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
            b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            cl->ResourceBarrier(1, &b);
            cl->CopyResource(y_readback.Get(), y_default.Get());
            b.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
            b.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            cl->ResourceBarrier(1, &b);
            cl->Close();
            if (!execute_wait(&cpu_b)) {
                out->push_back(std::move(seed));
                return;
            }
        }

        row.packed_gbs_cpu = packed_gbs_from_sec(row.packed_bytes, cpu_a);
        row.packed_gbs_h61 = packed_gbs_from_sec(row.packed_bytes, cpu_a + cpu_b);
        if (use_ts && gpu_sec > 0.0) {
            row.gpu_timestamp = true;
            row.packed_gbs = packed_gbs_from_sec(row.packed_bytes, gpu_sec);
        } else {
            row.gpu_timestamp = false;
            row.packed_gbs = row.packed_gbs_cpu;
        }

        {
            void* mapped = nullptr;
            D3D12_RANGE range = {0, static_cast<SIZE_T>(y_bytes)};
            hr = y_readback->Map(0, &range, &mapped);
            if (FAILED(hr)) {
                throw_if_failed(hr, "Map y_readback", row);
                out->push_back(std::move(row));
                return;
            }
            std::memcpy(gpu_y.data(), mapped, static_cast<std::size_t>(y_bytes));
            y_readback->Unmap(0, nullptr);
        }
        row.d3d12_ran = true;
        fill_checksum(row, host_y.data(), gpu_y.data(), n);
        out->push_back(std::move(row));
    }
}

GpugemvResult measure_gpugemv(int n, int k, int iterations, GpugemvKernel kernel) {
    std::vector<GpugemvResult> rows;
    measure_gpugemv_each(n, k, iterations, kernel, &rows);
    return gpugemv_median_summary(rows);
}

#endif // _WIN32

} // namespace xllama
