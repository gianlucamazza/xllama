// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include "xllama/gpustep.h"

#include "xllama/gpugemv.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#if defined(_WIN32)
    #include "d3d12_compute.h"
    #include "gpugemv_q4k_rows_dxil.h"
#endif

namespace xllama {

// --- Shapes ---

const char* gpustep_model_name(GpustepModelId id) {
    switch (id) {
    case GpustepModelId::Coder3B:
        return "qwen25-coder-3b";
    case GpustepModelId::Lfm12B:
        return "lfm25-1.2b-instruct";
    case GpustepModelId::Lfm350M:
        return "lfm25-350m";
    }
    return "?";
}

bool gpustep_q4km_more_bits(int layer, int n_layer) {
    return layer < n_layer / 8 || layer >= 7 * n_layer / 8 || (layer - n_layer / 8) % 3 == 2;
}

namespace {

GpustepMatmul mm(int layer, const char* name, int n, int k, GpustepQuant q) {
    GpustepMatmul m;
    m.name = layer < 0 ? std::string(name) : "blk." + std::to_string(layer) + "." + name;
    m.n = n;
    m.k = k;
    m.quant = q;
    return m;
}

void ffn_splits(std::vector<GpustepSplit>& out, int layer, int n_embd, int n_ff,
                GpustepQuant down_q) {
    out.push_back({{mm(layer, "ffn_gate", n_ff, n_embd, GpustepQuant::Q4K),
                    mm(layer, "ffn_up", n_ff, n_embd, GpustepQuant::Q4K)}});
    out.push_back({{mm(layer, "ffn_down", n_embd, n_ff, down_q)}});
}

// Qwen2.5-3B (HF config): 36 layers, n_embd 2048, n_ff 11008, 16 heads,
// 2 KV heads of 128, vocab 151936, tied embeddings. Q4_K_M promotes attn_v and
// ffn_down to Q6_K on use_more_bits layers; the tied lm_head is Q6_K.
GpustepModel coder3b() {
    GpustepModel m;
    m.id = GpustepModelId::Coder3B;
    m.name = gpustep_model_name(m.id);
    const int n_layer = 36, n_embd = 2048, n_ff = 11008, n_kv = 2 * 128;
    for (int il = 0; il < n_layer; ++il) {
        const GpustepQuant more =
            gpustep_q4km_more_bits(il, n_layer) ? GpustepQuant::Q6K : GpustepQuant::Q4K;
        m.splits.push_back({{mm(il, "attn_q", n_embd, n_embd, GpustepQuant::Q4K),
                             mm(il, "attn_k", n_kv, n_embd, GpustepQuant::Q4K),
                             mm(il, "attn_v", n_kv, n_embd, more)}});
        m.splits.push_back({{mm(il, "attn_output", n_embd, n_embd, GpustepQuant::Q4K)}});
        ffn_splits(m.splits, il, n_embd, n_ff, more);
    }
    m.splits.push_back({{mm(-1, "output", 151936, n_embd, GpustepQuant::Q6K)}});
    return m;
}

// LFM2.5 GGUF headers (QAD Q4_0, same density as Q4_K): 16 layers, attention
// at 2,5,8,10,12,14 (8 KV heads of 64), short-conv elsewhere (in_proj 3*n_embd,
// out_proj), vocab 65536, tied Q6_K token_embd as lm_head.
GpustepModel lfm(GpustepModelId id, int n_embd, int n_ff) {
    GpustepModel m;
    m.id = id;
    m.name = gpustep_model_name(id);
    const int n_kv = 8 * 64;
    for (int il = 0; il < 16; ++il) {
        const bool attn = il == 2 || il == 5 || il == 8 || il == 10 || il == 12 || il == 14;
        if (attn) {
            m.splits.push_back({{mm(il, "attn_q", n_embd, n_embd, GpustepQuant::Q4K),
                                 mm(il, "attn_k", n_kv, n_embd, GpustepQuant::Q4K),
                                 mm(il, "attn_v", n_kv, n_embd, GpustepQuant::Q4K)}});
            m.splits.push_back({{mm(il, "attn_output", n_embd, n_embd, GpustepQuant::Q4K)}});
        } else {
            m.splits.push_back(
                {{mm(il, "shortconv.in_proj", 3 * n_embd, n_embd, GpustepQuant::Q4K)}});
            m.splits.push_back({{mm(il, "shortconv.out_proj", n_embd, n_embd, GpustepQuant::Q4K)}});
        }
        ffn_splits(m.splits, il, n_embd, n_ff, GpustepQuant::Q4K);
    }
    m.splits.push_back({{mm(-1, "token_embd(lm_head)", 65536, n_embd, GpustepQuant::Q6K)}});
    return m;
}

} // namespace

GpustepModel gpustep_model(GpustepModelId id) {
    switch (id) {
    case GpustepModelId::Coder3B:
        return coder3b();
    case GpustepModelId::Lfm12B:
        return lfm(id, 2048, 8192);
    case GpustepModelId::Lfm350M:
        return lfm(id, 1024, 4608);
    }
    return {};
}

std::size_t gpustep_matmul_bytes(const GpustepMatmul& m) {
    if (m.n <= 0 || m.k <= 0 || m.k % kGpugemvQK != 0)
        return 0;
    const std::size_t block = m.quant == GpustepQuant::Q6K ? 210 : kGpugemvBlockBytes;
    return static_cast<std::size_t>(m.n) * static_cast<std::size_t>(m.k / kGpugemvQK) * block;
}

std::size_t gpustep_split_bytes(const GpustepSplit& s) {
    std::size_t b = 0;
    for (const auto& m : s.mm)
        b += gpustep_matmul_bytes(m);
    return b;
}

std::size_t gpustep_model_bytes(const GpustepModel& m) {
    std::size_t b = 0;
    for (const auto& s : m.splits)
        b += gpustep_split_bytes(s);
    return b;
}

std::size_t gpustep_model_matmuls(const GpustepModel& m) {
    std::size_t n = 0;
    for (const auto& s : m.splits)
        n += s.mm.size();
    return n;
}

int gpustep_sim_rows(const GpustepMatmul& m) {
    if (m.quant == GpustepQuant::Q4K)
        return m.n;
    const long long rows = (static_cast<long long>(m.n) * 210 + 143) / 144;
    return static_cast<int>((rows + 3) / 4 * 4);
}

double gpustep_projected_step_ms(const GpustepModel& m, double t_rt_us, double bw_gbs) {
    if (bw_gbs <= 0.0)
        return 0.0;
    double ms = 0.0;
    for (const auto& s : m.splits)
        ms += t_rt_us / 1000.0 + static_cast<double>(gpustep_split_bytes(s)) / (bw_gbs * 1e6);
    return ms;
}

double gpustep_projected_toks(double t_step_ms, double t_cpu_rest_ms) {
    const double t = t_step_ms + t_cpu_rest_ms;
    return t > 0.0 ? 1000.0 / t : 0.0;
}

// --- CSV ---

const char* gpustep_csv_header() {
    return "kind,variant,process,iterations,median,p90,unit,bytes,ok,d3d12_ran,uma,cc_uma,host,"
           "date,error\n";
}

namespace {

std::string csv_safe(std::string s) {
    for (char& c : s)
        if (c == ',' || c == '\n' || c == '\r')
            c = ' ';
    return s.empty() ? "-" : s;
}

} // namespace

std::string format_gpustep_row(const GpustepRow& r, const char* host_label) {
    char date_buf[32];
    std::time_t now = std::time(nullptr);
    std::strftime(date_buf, sizeof(date_buf), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&now));
    char buf[768];
    std::snprintf(buf, sizeof(buf), "%s,%s,%s,%d,%.3f,%.3f,%s,%zu,%d,%d,%d,%d,%s,%s,%s\n",
                  csv_safe(r.kind).c_str(), csv_safe(r.variant).c_str(),
                  csv_safe(r.process).c_str(), r.iterations, r.median, r.p90,
                  csv_safe(r.unit).c_str(), r.bytes, r.ok ? 1 : 0, r.d3d12_ran ? 1 : 0,
                  r.uma ? 1 : 0, r.cc_uma ? 1 : 0, host_label ? host_label : "unknown", date_buf,
                  csv_safe(r.error).c_str());
    return buf;
}

bool parse_gpustep_row(const std::string& line, GpustepRow* out) {
    if (!out || line.empty() || line.rfind("kind,", 0) == 0)
        return false;
    std::vector<std::string> f;
    std::stringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ','))
        f.push_back(cell);
    if (f.size() != 15)
        return false;
    for (auto& c : f)
        while (!c.empty() && (c.back() == '\r' || c.back() == '\n'))
            c.pop_back();
    GpustepRow r;
    r.kind = f[0];
    r.variant = f[1];
    r.process = f[2];
    char* end = nullptr;
    r.iterations = static_cast<int>(std::strtol(f[3].c_str(), &end, 10));
    r.median = std::strtod(f[4].c_str(), &end);
    r.p90 = std::strtod(f[5].c_str(), &end);
    r.unit = f[6];
    r.bytes = static_cast<std::size_t>(std::strtoull(f[7].c_str(), &end, 10));
    r.ok = f[8] == "1";
    r.d3d12_ran = f[9] == "1";
    r.uma = f[10] == "1";
    r.cc_uma = f[11] == "1";
    r.error = f[14] == "-" ? std::string() : f[14];
    *out = std::move(r);
    return true;
}

// --- Verdict ---

const char* gpustep_ladder_name(GpustepLadder l) {
    switch (l) {
    case GpustepLadder::D2MatmulOnly:
        return "D2-matmul-only";
    case GpustepLadder::D2Fused:
        return "D2-fused";
    case GpustepLadder::Park3B:
        return "park-3b";
    case GpustepLadder::ParkGui:
        return "park-gui";
    case GpustepLadder::NotAVerdict:
        return "NotAVerdict";
    }
    return "NotAVerdict";
}

namespace {

bool row_good(const GpustepRow& r) {
    return r.ok && r.d3d12_ran && r.error.empty();
}

bool ends_with(const std::string& s, const char* suffix) {
    const std::size_t n = std::strlen(suffix);
    return s.size() >= n && s.compare(s.size() - n, n, suffix) == 0;
}

// G1 for one process: every copy-path rt row and the DEFAULT-heap row passed.
// Zero-copy rows need a CUSTOM heap the adapter may refuse; they inform D1c/D1a
// but never fail G1.
bool process_g1(const std::vector<GpustepRow>& rows, const char* process) {
    bool have_rt = false, have_heap = false, all_ok = true;
    for (const auto& r : rows) {
        if (r.process != process)
            continue;
        if (r.kind == "rt" && ends_with(r.variant, "-copy")) {
            have_rt = true;
            all_ok = all_ok && row_good(r);
        } else if (r.kind == "heap" && r.variant == "default") {
            have_heap = true;
            all_ok = all_ok && row_good(r);
        }
    }
    return have_rt && have_heap && all_ok;
}

} // namespace

GpustepVerdict gpustep_evaluate(const std::vector<GpustepRow>& rows) {
    GpustepVerdict v;
    v.g1_headless = process_g1(rows, "headless");
    double heap_default = 0.0, heap_host = 0.0;
    const std::string coder = gpustep_model_name(GpustepModelId::Coder3B);
    for (const auto& r : rows) {
        if (r.process == "inproc")
            v.have_inproc = true;
        if (r.process != "headless" || !row_good(r))
            continue;
        if (r.kind == "rt" && (v.rt_best_us <= 0.0 || r.median < v.rt_best_us))
            v.rt_best_us = r.median;
        else if (r.kind == "step" && r.variant == coder + "/sync")
            v.step_sync_ms = r.median;
        else if (r.kind == "step" && r.variant == coder + "/nosync")
            v.step_nosync_ms = r.median;
        else if (r.kind == "heap" && r.variant == "default")
            heap_default = r.median;
        else if (r.kind == "heap")
            heap_host = std::max(heap_host, r.median);
    }
    v.g1_inproc = v.have_inproc && process_g1(rows, "inproc");
    v.d1a = v.rt_best_us > 0.0 && v.rt_best_us <= kGpustepRtGateUs;
    v.d1b = v.step_sync_ms > 0.0 && v.step_sync_ms <= kGpustepStepGateMs;
    v.d1d = v.g1_inproc;
    if (heap_default <= 0.0)
        v.d1c = "n/a";
    else
        v.d1c =
            heap_host >= kGpustepHostHeapMinRatio * heap_default ? "host-visible" : "default+copy";

    if (!v.g1_headless || v.rt_best_us <= 0.0 || v.step_sync_ms <= 0.0 || v.step_nosync_ms <= 0.0 ||
        !v.have_inproc)
        v.ladder = GpustepLadder::NotAVerdict;
    else if (!v.d1d)
        v.ladder = GpustepLadder::ParkGui;
    else if (v.d1a && v.d1b)
        v.ladder = GpustepLadder::D2MatmulOnly;
    else if (v.step_nosync_ms <= kGpustepStepGateMs)
        v.ladder = GpustepLadder::D2Fused;
    else
        v.ladder = GpustepLadder::Park3B;
    return v;
}

namespace {

void median_p90(std::vector<double> v, double* median, double* p90) {
    if (v.empty()) {
        *median = *p90 = 0.0;
        return;
    }
    std::sort(v.begin(), v.end());
    *median = v[v.size() / 2];
    *p90 = v[std::min(v.size() - 1, v.size() * 9 / 10)];
}

} // namespace

#if !defined(_WIN32)

void measure_gpustep(const char* process_label, std::vector<GpustepRow>* out) {
    if (!out)
        return;
    GpustepRow r;
    r.kind = "caps";
    r.variant = "-";
    r.process = process_label ? process_label : "host";
    r.unit = "-";
    r.error = "d3d12 unavailable on this platform";
    out->push_back(std::move(r));
    (void)&median_p90;
}

#else

namespace {

using d3d12c::ComPtr;
using Clock = std::chrono::steady_clock;

double elapsed_us(Clock::time_point t0) {
    return std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
}

struct Params {
    std::uint32_t n, k, nb, pad;
};

// Shared per-process D3D12 state for every D1 section.
struct Ctx {
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12RootSignature> root;
    ComPtr<ID3D12PipelineState> pso;
    d3d12c::QueueFence fence;
    UINT64 ts_freq = 0;
    UINT incr = 0;
    bool uma = false;
    bool cc_uma = false;
    std::string process;
};

GpustepRow make_row(const Ctx& c, const char* kind, const std::string& variant, const char* unit) {
    GpustepRow r;
    r.kind = kind;
    r.variant = variant;
    r.process = c.process;
    r.unit = unit;
    r.uma = c.uma;
    r.cc_uma = c.cc_uma;
    return r;
}

bool init_ctx(Ctx& c, std::string* err) {
    c.device = d3d12c::create_device(err);
    if (!c.device)
        return false;
    D3D12_FEATURE_DATA_ARCHITECTURE1 arch = {};
    if (SUCCEEDED(
            c.device->CheckFeatureSupport(D3D12_FEATURE_ARCHITECTURE1, &arch, sizeof(arch)))) {
        c.uma = arch.UMA != FALSE;
        c.cc_uma = arch.CacheCoherentUMA != FALSE;
    }
    D3D12_COMMAND_QUEUE_DESC qd = {};
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    HRESULT hr = c.device->CreateCommandQueue(&qd, IID_PPV_ARGS(&c.queue));
    if (FAILED(hr)) {
        *err = d3d12c::hr_message("CreateCommandQueue", hr);
        return false;
    }
    if (FAILED(c.queue->GetTimestampFrequency(&c.ts_freq)))
        c.ts_freq = 0;
    if (!c.fence.init(c.device.Get(), err))
        return false;
    c.root = d3d12c::create_gemv_root_sig(c.device.Get(), err);
    if (!c.root)
        return false;
    D3D12_COMPUTE_PIPELINE_STATE_DESC pd = {};
    pd.pRootSignature = c.root.Get();
    pd.CS.pShaderBytecode = kGpugemvQ4kRowsDxil;
    pd.CS.BytecodeLength = kGpugemvQ4kRowsDxilSize;
    hr = c.device->CreateComputePipelineState(&pd, IID_PPV_ARGS(&c.pso));
    if (FAILED(hr)) {
        *err = d3d12c::hr_message("CreateComputePipelineState", hr);
        return false;
    }
    c.incr = c.device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    return true;
}

// One command allocator + list; every list in a section shares the allocator
// and is recorded once, then re-executed (never reset while in flight).
struct Recorder {
    ComPtr<ID3D12CommandAllocator> alloc;
    bool init(Ctx& c, std::string* err) {
        HRESULT hr =
            c.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc));
        if (FAILED(hr))
            *err = d3d12c::hr_message("CreateCommandAllocator", hr);
        return SUCCEEDED(hr);
    }
    ComPtr<ID3D12GraphicsCommandList> begin(Ctx& c, std::string* err) {
        ComPtr<ID3D12GraphicsCommandList> cl;
        HRESULT hr = c.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.Get(),
                                                 nullptr, IID_PPV_ARGS(&cl));
        if (FAILED(hr))
            *err = d3d12c::hr_message("CreateCommandList", hr);
        return cl;
    }
};

bool run_list(Ctx& c, ID3D12GraphicsCommandList* cl, bool spin) {
    ID3D12CommandList* lists[] = {cl};
    c.queue->ExecuteCommandLists(1, lists);
    return c.fence.signal_and_wait(c.queue.Get(), spin);
}

void barrier(ID3D12GraphicsCommandList* cl, ID3D12Resource* res, D3D12_RESOURCE_STATES before,
             D3D12_RESOURCE_STATES after) {
    D3D12_RESOURCE_BARRIER b = {};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = res;
    b.Transition.StateBefore = before;
    b.Transition.StateAfter = after;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    cl->ResourceBarrier(1, &b);
}

void uav_barrier(ID3D12GraphicsCommandList* cl, ID3D12Resource* res) {
    D3D12_RESOURCE_BARRIER b = {};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    b.UAV.pResource = res;
    cl->ResourceBarrier(1, &b);
}

D3D12_HEAP_PROPERTIES custom_writeback_props() {
    D3D12_HEAP_PROPERTIES hp = {};
    hp.Type = D3D12_HEAP_TYPE_CUSTOM;
    hp.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_WRITE_BACK;
    hp.MemoryPoolPreference = D3D12_MEMORY_POOL_L0;
    return hp;
}

void* map_all(ID3D12Resource* r) {
    void* p = nullptr;
    return SUCCEEDED(r->Map(0, nullptr, &p)) ? p : nullptr;
}

// Descriptor tables (CBV, W, X, Y) for many GEMVs; constants in 256 B slots.
struct Tables {
    ComPtr<ID3D12DescriptorHeap> heap;
    ComPtr<ID3D12Resource> cb;
    std::uint8_t* cb_ptr = nullptr;
    UINT count = 0;
    bool init(Ctx& c, UINT n_tables, std::string* err) {
        D3D12_DESCRIPTOR_HEAP_DESC hd = {};
        hd.NumDescriptors = 4 * n_tables;
        hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        HRESULT hr = c.device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&heap));
        if (FAILED(hr)) {
            *err = d3d12c::hr_message("CreateDescriptorHeap", hr);
            return false;
        }
        cb = d3d12c::create_buffer(c.device.Get(), 256ull * n_tables, D3D12_HEAP_TYPE_UPLOAD,
                                   D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ,
                                   "cb", err);
        if (!cb)
            return false;
        cb_ptr = static_cast<std::uint8_t*>(map_all(cb.Get()));
        count = n_tables;
        return cb_ptr != nullptr;
    }
    // y_first/y_n in float elements.
    void set(Ctx& c, UINT i, int n, int k, ID3D12Resource* w, UINT64 w_bytes, ID3D12Resource* x,
             UINT64 x_bytes, ID3D12Resource* y, UINT y_first, UINT y_n) {
        const Params p{static_cast<std::uint32_t>(n), static_cast<std::uint32_t>(k),
                       static_cast<std::uint32_t>(k / kGpugemvQK), 0};
        std::memcpy(cb_ptr + 256ull * i, &p, sizeof(p));
        D3D12_CPU_DESCRIPTOR_HANDLE h = heap->GetCPUDescriptorHandleForHeapStart();
        h.ptr += static_cast<SIZE_T>(4 * i) * c.incr;
        D3D12_CONSTANT_BUFFER_VIEW_DESC cbv = {};
        cbv.BufferLocation = cb->GetGPUVirtualAddress() + 256ull * i;
        cbv.SizeInBytes = 256;
        c.device->CreateConstantBufferView(&cbv, h);
        auto raw_srv = [&](ID3D12Resource* res, UINT64 bytes, D3D12_CPU_DESCRIPTOR_HANDLE at) {
            D3D12_SHADER_RESOURCE_VIEW_DESC s = {};
            s.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
            s.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            s.Format = DXGI_FORMAT_R32_TYPELESS;
            s.Buffer.NumElements = static_cast<UINT>(bytes / 4);
            s.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
            c.device->CreateShaderResourceView(res, &s, at);
        };
        h.ptr += c.incr;
        raw_srv(w, w_bytes, h);
        h.ptr += c.incr;
        raw_srv(x, x_bytes, h);
        h.ptr += c.incr;
        D3D12_UNORDERED_ACCESS_VIEW_DESC u = {};
        u.Format = DXGI_FORMAT_UNKNOWN;
        u.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        u.Buffer.FirstElement = y_first;
        u.Buffer.NumElements = y_n;
        u.Buffer.StructureByteStride = sizeof(float);
        c.device->CreateUnorderedAccessView(y, nullptr, &u, h);
    }
    void bind(Ctx& c, ID3D12GraphicsCommandList* cl) {
        cl->SetComputeRootSignature(c.root.Get());
        ID3D12DescriptorHeap* heaps[] = {heap.Get()};
        cl->SetDescriptorHeaps(1, heaps);
        cl->SetPipelineState(c.pso.Get());
    }
    void dispatch(Ctx& c, ID3D12GraphicsCommandList* cl, UINT i, int n) {
        D3D12_GPU_DESCRIPTOR_HANDLE g = heap->GetGPUDescriptorHandleForHeapStart();
        g.ptr += static_cast<UINT64>(4 * i) * c.incr;
        cl->SetComputeRootDescriptorTable(0, g);
        cl->Dispatch(static_cast<UINT>((n + kGpugemvRowsPerGroup - 1) / kGpugemvRowsPerGroup), 1,
                     1);
    }
};

// Upload `bytes` from host memory into a DEFAULT buffer (COPY_DEST → `after`).
bool upload_default(Ctx& c, ID3D12Resource* dst, const void* src, UINT64 bytes,
                    D3D12_RESOURCE_STATES after, std::string* err) {
    auto staging = d3d12c::create_buffer(c.device.Get(), bytes, D3D12_HEAP_TYPE_UPLOAD,
                                         D3D12_RESOURCE_FLAG_NONE,
                                         D3D12_RESOURCE_STATE_GENERIC_READ, "staging", err);
    if (!staging)
        return false;
    void* p = map_all(staging.Get());
    if (!p) {
        *err = "Map staging failed";
        return false;
    }
    std::memcpy(p, src, static_cast<std::size_t>(bytes));
    staging->Unmap(0, nullptr);
    Recorder rec;
    if (!rec.init(c, err))
        return false;
    auto cl = rec.begin(c, err);
    if (!cl)
        return false;
    cl->CopyBufferRegion(dst, 0, staging.Get(), 0, bytes);
    barrier(cl.Get(), dst, D3D12_RESOURCE_STATE_COPY_DEST, after);
    cl->Close();
    return run_list(c, cl.Get(), false);
}

// --- heap: N=K=8192 GEMV with W in DEFAULT / UPLOAD / CUSTOM(WRITE_BACK, L0) ---
void section_heap(Ctx& c, std::vector<GpustepRow>* out) {
    const int n = kGpugemvDefaultN, k = kGpugemvDefaultK;
    const UINT64 w_bytes = gpugemv_packed_bytes(n, k);
    std::vector<GpugemvQ4KBlock> host_w(static_cast<std::size_t>(n) * (k / kGpugemvQK));
    std::vector<float> host_x(static_cast<std::size_t>(k)), host_y(static_cast<std::size_t>(n));
    gpugemv_fill_weights(host_w.data(), n, k);
    gpugemv_fill_x(host_x.data(), k);
    gpugemv_cpu_ref(host_w.data(), host_x.data(), host_y.data(), n, k);

    std::string err;
    const UINT64 x_bytes = static_cast<UINT64>(k) * 4, y_bytes = static_cast<UINT64>(n) * 4;
    auto x =
        d3d12c::create_buffer(c.device.Get(), x_bytes, D3D12_HEAP_TYPE_DEFAULT,
                              D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST, "x", &err);
    auto y = d3d12c::create_buffer(c.device.Get(), y_bytes, D3D12_HEAP_TYPE_DEFAULT,
                                   D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                   D3D12_RESOURCE_STATE_UNORDERED_ACCESS, "y", &err);
    auto y_rb = d3d12c::create_buffer(c.device.Get(), y_bytes, D3D12_HEAP_TYPE_READBACK,
                                      D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST,
                                      "y_rb", &err);
    auto ts_rb = d3d12c::create_buffer(c.device.Get(), 256, D3D12_HEAP_TYPE_READBACK,
                                       D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST,
                                       "ts_rb", &err);
    ComPtr<ID3D12QueryHeap> qh;
    D3D12_QUERY_HEAP_DESC qhd = {};
    qhd.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
    qhd.Count = 2;
    if (x && y && y_rb && ts_rb && c.ts_freq > 0)
        c.device->CreateQueryHeap(&qhd, IID_PPV_ARGS(&qh));
    if (!x || !y || !y_rb || !ts_rb || !qh ||
        !upload_default(c, x.Get(), host_x.data(), x_bytes,
                        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, &err)) {
        GpustepRow r = make_row(c, "heap", "default", "gbs");
        r.error = err.empty() ? "heap setup failed" : err;
        out->push_back(std::move(r));
        return;
    }

    const char* kinds[] = {"default", "upload", "custom"};
    for (const char* kind : kinds) {
        GpustepRow r = make_row(c, "heap", kind, "gbs");
        r.bytes = static_cast<std::size_t>(w_bytes);
        err.clear();
        ComPtr<ID3D12Resource> w;
        if (std::strcmp(kind, "default") == 0) {
            w = d3d12c::create_buffer(c.device.Get(), w_bytes, D3D12_HEAP_TYPE_DEFAULT,
                                      D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST,
                                      "w_default", &err);
            if (w && !upload_default(c, w.Get(), host_w.data(), w_bytes,
                                     D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, &err))
                w.Reset();
        } else {
            if (std::strcmp(kind, "upload") == 0)
                w = d3d12c::create_buffer(c.device.Get(), w_bytes, D3D12_HEAP_TYPE_UPLOAD,
                                          D3D12_RESOURCE_FLAG_NONE,
                                          D3D12_RESOURCE_STATE_GENERIC_READ, "w_upload", &err);
            else
                w = d3d12c::create_buffer_props(c.device.Get(), w_bytes, custom_writeback_props(),
                                                D3D12_RESOURCE_FLAG_NONE,
                                                D3D12_RESOURCE_STATE_COMMON, "w_custom", &err);
            void* p = w ? map_all(w.Get()) : nullptr;
            if (p) {
                std::memcpy(p, host_w.data(), static_cast<std::size_t>(w_bytes));
                w->Unmap(0, nullptr);
            } else if (w) {
                err = "Map w failed";
                w.Reset();
            }
        }
        if (!w) {
            r.error = err;
            out->push_back(std::move(r));
            continue;
        }

        Tables t;
        Recorder rec;
        ComPtr<ID3D12GraphicsCommandList> cl;
        if (!t.init(c, 1, &err) || !rec.init(c, &err) || !(cl = rec.begin(c, &err))) {
            r.error = err;
            out->push_back(std::move(r));
            continue;
        }
        t.set(c, 0, n, k, w.Get(), w_bytes, x.Get(), x_bytes, y.Get(), 0, static_cast<UINT>(n));
        t.bind(c, cl.Get());
        cl->EndQuery(qh.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0);
        t.dispatch(c, cl.Get(), 0, n);
        uav_barrier(cl.Get(), y.Get());
        cl->EndQuery(qh.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 1);
        cl->ResolveQueryData(qh.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, ts_rb.Get(), 0);
        barrier(cl.Get(), y.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COPY_SOURCE);
        cl->CopyBufferRegion(y_rb.Get(), 0, y.Get(), 0, y_bytes);
        barrier(cl.Get(), y.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        cl->Close();

        std::vector<double> gbs;
        bool ran = run_list(c, cl.Get(), false); // warm-up
        const int iters = 5;
        for (int it = 0; ran && it < iters; ++it) {
            ran = run_list(c, cl.Get(), false);
            const auto* ts = static_cast<const std::uint64_t*>(map_all(ts_rb.Get()));
            if (ts && ts[1] > ts[0])
                gbs.push_back(
                    static_cast<double>(w_bytes) / 1e9 /
                    (static_cast<double>(ts[1] - ts[0]) / static_cast<double>(c.ts_freq)));
            ts_rb->Unmap(0, nullptr);
        }
        r.iterations = iters;
        r.d3d12_ran = ran;
        median_p90(gbs, &r.median, &r.p90);
        const float* gy = static_cast<const float*>(map_all(y_rb.Get()));
        if (gy) {
            r.ok = ran && gpugemv_max_abs_err(host_y.data(), gy, static_cast<std::size_t>(n)) <=
                              kGpugemvMaxAbsErrTol;
            y_rb->Unmap(0, nullptr);
        }
        if (!ran)
            r.error = "execute/fence failed";
        else if (!r.ok)
            r.error = "GEMV residual mismatch";
        out->push_back(std::move(r));
    }
}

// --- rt: tiny GEMV (N=256, K=2048): host X -> GPU -> host Y, timed on the CPU ---
void section_rt(Ctx& c, std::vector<GpustepRow>* out) {
    const int n = 256, k = 2048;
    const UINT64 w_bytes = gpugemv_packed_bytes(n, k);
    const UINT64 x_bytes = static_cast<UINT64>(k) * 4, y_bytes = static_cast<UINT64>(n) * 4;
    std::vector<GpugemvQ4KBlock> host_w(static_cast<std::size_t>(n) * (k / kGpugemvQK));
    std::vector<float> host_x(static_cast<std::size_t>(k)), host_y(static_cast<std::size_t>(n)),
        got(static_cast<std::size_t>(n));
    gpugemv_fill_weights(host_w.data(), n, k);
    gpugemv_fill_x(host_x.data(), k);
    gpugemv_cpu_ref(host_w.data(), host_x.data(), host_y.data(), n, k);

    struct Variant {
        const char* name;
        bool spin;
        bool zerocopy;
    };
    const Variant variants[] = {
        {"event-copy", false, false}, {"spin-copy", true, false}, {"spin-zerocopy", true, true}};
    for (const Variant& v : variants) {
        GpustepRow r = make_row(c, "rt", v.name, "us");
        r.bytes = static_cast<std::size_t>(x_bytes + y_bytes);
        std::string err;
        auto w = d3d12c::create_buffer(c.device.Get(), w_bytes, D3D12_HEAP_TYPE_DEFAULT,
                                       D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST,
                                       "w", &err);
        auto x_up = d3d12c::create_buffer(c.device.Get(), x_bytes, D3D12_HEAP_TYPE_UPLOAD,
                                          D3D12_RESOURCE_FLAG_NONE,
                                          D3D12_RESOURCE_STATE_GENERIC_READ, "x_up", &err);
        ComPtr<ID3D12Resource> x_def, y, y_rb;
        if (v.zerocopy) {
            // Y written by the GPU into CPU-visible memory: no copy, no readback heap.
            y = d3d12c::create_buffer_props(c.device.Get(), y_bytes, custom_writeback_props(),
                                            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, "y_custom",
                                            &err);
        } else {
            x_def = d3d12c::create_buffer(c.device.Get(), x_bytes, D3D12_HEAP_TYPE_DEFAULT,
                                          D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST,
                                          "x_def", &err);
            y = d3d12c::create_buffer(c.device.Get(), y_bytes, D3D12_HEAP_TYPE_DEFAULT,
                                      D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                      D3D12_RESOURCE_STATE_UNORDERED_ACCESS, "y", &err);
            y_rb = d3d12c::create_buffer(c.device.Get(), y_bytes, D3D12_HEAP_TYPE_READBACK,
                                         D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST,
                                         "y_rb", &err);
        }
        Tables t;
        Recorder rec;
        ComPtr<ID3D12GraphicsCommandList> cl;
        const bool setup = w && x_up && y && (v.zerocopy || (x_def && y_rb)) &&
                           upload_default(c, w.Get(), host_w.data(), w_bytes,
                                          D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, &err) &&
                           t.init(c, 1, &err) && rec.init(c, &err) && (cl = rec.begin(c, &err));
        void* x_ptr = setup ? map_all(x_up.Get()) : nullptr;
        void* y_ptr = setup ? map_all(v.zerocopy ? y.Get() : y_rb.Get()) : nullptr;
        if (!setup || !x_ptr || !y_ptr) {
            r.error = err.empty() ? "rt setup/map failed" : err;
            out->push_back(std::move(r));
            continue;
        }
        t.set(c, 0, n, k, w.Get(), w_bytes, v.zerocopy ? x_up.Get() : x_def.Get(), x_bytes, y.Get(),
              0, static_cast<UINT>(n));
        if (!v.zerocopy) {
            cl->CopyBufferRegion(x_def.Get(), 0, x_up.Get(), 0, x_bytes);
            barrier(cl.Get(), x_def.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        }
        t.bind(c, cl.Get());
        t.dispatch(c, cl.Get(), 0, n);
        uav_barrier(cl.Get(), y.Get());
        if (!v.zerocopy) {
            barrier(cl.Get(), y.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_COPY_SOURCE);
            cl->CopyBufferRegion(y_rb.Get(), 0, y.Get(), 0, y_bytes);
            barrier(cl.Get(), y.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            barrier(cl.Get(), x_def.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                    D3D12_RESOURCE_STATE_COPY_DEST);
        }
        cl->Close();

        const int warm = 50, iters = 500;
        std::vector<double> us;
        bool ran = true;
        for (int it = 0; ran && it < warm + iters; ++it) {
            const auto t0 = Clock::now();
            std::memcpy(x_ptr, host_x.data(), static_cast<std::size_t>(x_bytes));
            ran = run_list(c, cl.Get(), v.spin);
            std::memcpy(got.data(), y_ptr, static_cast<std::size_t>(y_bytes));
            if (it >= warm)
                us.push_back(elapsed_us(t0));
        }
        r.iterations = iters;
        r.d3d12_ran = ran;
        median_p90(us, &r.median, &r.p90);
        r.ok = ran && gpugemv_max_abs_err(host_y.data(), got.data(), static_cast<std::size_t>(n)) <=
                          kGpugemvMaxAbsErrTol;
        if (!ran)
            r.error = "execute/fence failed";
        else if (!r.ok)
            r.error = "GEMV residual mismatch";
        out->push_back(std::move(r));
    }
}

// --- step: one simulated decode token of a model shape ---
void section_step(Ctx& c, GpustepModelId id, std::vector<GpustepRow>* out) {
    const GpustepModel model = gpustep_model(id);
    std::string err;
    const UINT n_mm = static_cast<UINT>(gpustep_model_matmuls(model));
    int max_k = 0;
    UINT max_split_rows = 0;
    for (const auto& s : model.splits) {
        UINT rows = 0;
        for (const auto& m : s.mm) {
            max_k = std::max(max_k, m.k);
            rows += static_cast<UINT>(gpustep_sim_rows(m));
        }
        max_split_rows = std::max(max_split_rows, rows);
    }
    const UINT64 x_bytes = static_cast<UINT64>(max_k) * 4;
    const UINT64 y_bytes = static_cast<UINT64>(max_split_rows) * 4;

    auto fail = [&](const std::string& e) {
        const char* modes[] = {"sync", "nosync"};
        for (const char* mode : modes) {
            GpustepRow r = make_row(c, "step", model.name + "/" + mode, "ms");
            r.error = e.empty() ? "step setup failed" : e;
            out->push_back(std::move(r));
        }
    };

    // Weights: one DEFAULT buffer per matmul, filled from a 64 MiB pattern tile.
    const UINT64 tile_bytes = 64ull << 20;
    const std::size_t tile_blocks = static_cast<std::size_t>(tile_bytes / kGpugemvBlockBytes);
    auto staging = d3d12c::create_buffer(c.device.Get(), tile_bytes, D3D12_HEAP_TYPE_UPLOAD,
                                         D3D12_RESOURCE_FLAG_NONE,
                                         D3D12_RESOURCE_STATE_GENERIC_READ, "tile", &err);
    if (!staging)
        return fail(err);
    {
        void* p = map_all(staging.Get());
        if (!p)
            return fail("Map tile failed");
        std::vector<GpugemvQ4KBlock> blocks(tile_blocks);
        // fill_weights wants n rows of k/256 blocks; 256-wide rows tile cleanly.
        gpugemv_fill_weights(blocks.data(), static_cast<int>(tile_blocks), kGpugemvQK);
        std::memcpy(p, blocks.data(), blocks.size() * sizeof(GpugemvQ4KBlock));
        staging->Unmap(0, nullptr);
    }
    const UINT64 tile_used = static_cast<UINT64>(tile_blocks) * kGpugemvBlockBytes;

    std::vector<ComPtr<ID3D12Resource>> w(n_mm);
    std::vector<UINT64> w_bytes(n_mm);
    Recorder fill_rec;
    if (!fill_rec.init(c, &err))
        return fail(err);
    {
        auto cl = fill_rec.begin(c, &err);
        if (!cl)
            return fail(err);
        UINT i = 0;
        for (const auto& s : model.splits) {
            for (const auto& m : s.mm) {
                w_bytes[i] = gpugemv_packed_bytes(gpustep_sim_rows(m), m.k);
                w[i] = d3d12c::create_buffer(c.device.Get(), w_bytes[i], D3D12_HEAP_TYPE_DEFAULT,
                                             D3D12_RESOURCE_FLAG_NONE,
                                             D3D12_RESOURCE_STATE_COPY_DEST, m.name.c_str(), &err);
                if (!w[i])
                    return fail(err);
                for (UINT64 off = 0; off < w_bytes[i]; off += tile_used)
                    cl->CopyBufferRegion(w[i].Get(), off, staging.Get(), 0,
                                         std::min(tile_used, w_bytes[i] - off));
                barrier(cl.Get(), w[i].Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                ++i;
            }
        }
        cl->Close();
        if (!run_list(c, cl.Get(), false))
            return fail("weight fill failed");
    }
    staging.Reset();

    auto x_up = d3d12c::create_buffer(c.device.Get(), x_bytes, D3D12_HEAP_TYPE_UPLOAD,
                                      D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ,
                                      "x_up", &err);
    auto x_def = d3d12c::create_buffer(c.device.Get(), x_bytes, D3D12_HEAP_TYPE_DEFAULT,
                                       D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST,
                                       "x_def", &err);
    auto y = d3d12c::create_buffer(c.device.Get(), y_bytes, D3D12_HEAP_TYPE_DEFAULT,
                                   D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                   D3D12_RESOURCE_STATE_UNORDERED_ACCESS, "y", &err);
    auto y_rb = d3d12c::create_buffer(c.device.Get(), y_bytes, D3D12_HEAP_TYPE_READBACK,
                                      D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST,
                                      "y_rb", &err);
    Tables t;
    Recorder rec;
    if (!x_up || !x_def || !y || !y_rb || !t.init(c, n_mm, &err) || !rec.init(c, &err))
        return fail(err);
    void* x_ptr = map_all(x_up.Get());
    void* y_ptr = map_all(y_rb.Get());
    if (!x_ptr || !y_ptr)
        return fail("Map x/y failed");
    std::vector<float> host_x(static_cast<std::size_t>(max_k));
    gpugemv_fill_x(host_x.data(), max_k);
    std::vector<float> host_y(static_cast<std::size_t>(max_split_rows));

    // One pre-recorded list per split: copy X in, dispatch its matmuls, copy Y out.
    std::vector<ComPtr<ID3D12GraphicsCommandList>> lists;
    std::vector<UINT64> split_x(model.splits.size()), split_y(model.splits.size());
    UINT ti = 0;
    for (std::size_t si = 0; si < model.splits.size(); ++si) {
        const auto& s = model.splits[si];
        auto cl = rec.begin(c, &err);
        if (!cl)
            return fail(err);
        const int k = s.mm.front().k;
        split_x[si] = static_cast<UINT64>(k) * 4;
        cl->CopyBufferRegion(x_def.Get(), 0, x_up.Get(), 0, split_x[si]);
        barrier(cl.Get(), x_def.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        t.bind(c, cl.Get());
        UINT y_off = 0;
        for (const auto& m : s.mm) {
            const int rows = gpustep_sim_rows(m);
            t.set(c, ti, rows, m.k, w[ti].Get(), w_bytes[ti], x_def.Get(), x_bytes, y.Get(), y_off,
                  static_cast<UINT>(rows));
            t.dispatch(c, cl.Get(), ti, rows);
            y_off += static_cast<UINT>(rows);
            ++ti;
        }
        split_y[si] = static_cast<UINT64>(y_off) * 4;
        uav_barrier(cl.Get(), y.Get());
        barrier(cl.Get(), y.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COPY_SOURCE);
        cl->CopyBufferRegion(y_rb.Get(), 0, y.Get(), 0, split_y[si]);
        barrier(cl.Get(), y.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        barrier(cl.Get(), x_def.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_COPY_DEST);
        cl->Close();
        lists.push_back(cl);
    }
    std::vector<ID3D12CommandList*> raw(lists.size());
    for (std::size_t i = 0; i < lists.size(); ++i)
        raw[i] = lists[i].Get();

    const int warm = 20, iters = 50;
    const std::size_t bytes = gpustep_model_bytes(model);
    // sync: CPU writes X before and reads Y after every split, like the scheduler.
    {
        GpustepRow r = make_row(c, "step", model.name + "/sync", "ms");
        r.bytes = bytes;
        std::vector<double> ms;
        bool ran = true;
        for (int it = 0; ran && it < warm + iters; ++it) {
            const auto t0 = Clock::now();
            for (std::size_t si = 0; ran && si < lists.size(); ++si) {
                std::memcpy(x_ptr, host_x.data(), static_cast<std::size_t>(split_x[si]));
                ran = run_list(c, lists[si].Get(), true);
                std::memcpy(host_y.data(), y_ptr, static_cast<std::size_t>(split_y[si]));
            }
            if (it >= warm)
                ms.push_back(elapsed_us(t0) / 1000.0);
        }
        r.iterations = iters;
        r.d3d12_ran = ran;
        median_p90(ms, &r.median, &r.p90);
        r.ok = ran && std::isfinite(host_y[0]);
        if (!r.ok)
            r.error = ran ? "non-finite output" : "execute/fence failed";
        out->push_back(std::move(r));
    }
    // nosync: every split in one submit, one wait — the pure-GPU lower bound.
    {
        GpustepRow r = make_row(c, "step", model.name + "/nosync", "ms");
        r.bytes = bytes;
        std::vector<double> ms;
        bool ran = true;
        for (int it = 0; ran && it < warm + iters; ++it) {
            const auto t0 = Clock::now();
            c.queue->ExecuteCommandLists(static_cast<UINT>(raw.size()), raw.data());
            ran = c.fence.signal_and_wait(c.queue.Get(), true);
            if (it >= warm)
                ms.push_back(elapsed_us(t0) / 1000.0);
        }
        r.iterations = iters;
        r.d3d12_ran = ran;
        median_p90(ms, &r.median, &r.p90);
        r.ok = ran;
        if (!ran)
            r.error = "execute/fence failed";
        out->push_back(std::move(r));
    }
}

} // namespace

void measure_gpustep(const char* process_label, std::vector<GpustepRow>* out) {
    if (!out)
        return;
    Ctx c;
    c.process = process_label ? process_label : "headless";
    GpustepRow caps = make_row(c, "caps", "-", "-");
    std::string err;
    if (!init_ctx(c, &err)) {
        caps.error = err;
        out->push_back(std::move(caps));
        return;
    }
    caps.uma = c.uma;
    caps.cc_uma = c.cc_uma;
    caps.ok = true;
    caps.d3d12_ran = true;
    out->push_back(std::move(caps));

    section_heap(c, out);
    section_rt(c, out);
    const GpustepModelId models[] = {GpustepModelId::Coder3B, GpustepModelId::Lfm12B,
                                     GpustepModelId::Lfm350M};
    for (GpustepModelId id : models)
        section_step(c, id, out);
}

#endif // _WIN32

} // namespace xllama
