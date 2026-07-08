// Copyright (c) 2024 Venere Labs
// SPDX-License-Identifier: MIT
// Image-generation spike: plain ONNX Runtime DirectML (NOT ORT GenAI).
//
// Runs a compute-bound conv stack (imgspike.onnx, ~309 GFLOP/forward, fp16,
// 512x512 — a faithful proxy for one diffusion UNet step) through the DirectML
// EP and a CPU EP control, measuring forward-pass latency. Purpose: test the
// hypothesis that the RDNA 2 GPU wins on compute-bound fp16 batch workloads
// (image gen) — the opposite of the M=1, dispatch-bound text decode where it
// loses. Runs in the headless path (image.flag), so no XAML compositor D3D12
// device exists to collide with ORT's DML device (cf. the 887A0036 finding).

#include "inference-bridge.h"

#ifdef XLLAMA_UWP

// clang-format off
    #include <windows.h>
    #include <eh.h>
    #include <onnxruntime_cxx_api.h>
    #include <dml_provider_factory.h>
// clang-format on

    #include "xllama/path_utils.h"
    #include "xllama/platform.h"
    #include "xllama/utf8_utils.h"

    #include <array>
    #include <chrono>
    #include <cstdio>
    #include <ctime>
    #include <string>
    #include <vector>

namespace xllama::bridge {

namespace {

// imgspike.onnx: 1x3x512x512 fp16, 17 Conv(3x3,64ch)+Relu layers.
constexpr int kH = 512;
constexpr int kW = 512;
constexpr double kGFlop = 309.2; // ~ from the model's conv FLOP count

struct SpikeRun {
    const char* ep = "unknown";
    bool ok = false;
    double best_ms = 0.0;
    std::string err;
};

// Run the model once (warmup) + `iters` timed forwards; return best latency.
SpikeRun run_ep(Ort::Env& env, const std::wstring& model_path, bool use_dml, int iters) {
    SpikeRun r;
    r.ep = use_dml ? "directml" : "cpu";
    try {
        Ort::SessionOptions so;
        // DirectML requires sequential execution and no memory pattern.
        so.SetExecutionMode(ORT_SEQUENTIAL);
        so.DisableMemPattern();
        if (use_dml) {
            Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_DML(so, /*device_id=*/0));
        }
        Ort::Session session(env, model_path.c_str(), so);

        const size_t n = static_cast<size_t>(1) * 3 * kH * kW;
        std::vector<Ort::Float16_t> input(n); // fp16 zeros — values irrelevant for timing
        std::array<int64_t, 4> shape{1, 3, kH, kW};
        Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        Ort::Value in = Ort::Value::CreateTensor<Ort::Float16_t>(mem, input.data(), n, shape.data(),
                                                                 shape.size());
        const char* in_names[] = {"input"};
        const char* out_names[] = {"output"};

        // Warmup (DML compiles shaders / builds the graph on the first run).
        session.Run(Ort::RunOptions{nullptr}, in_names, &in, 1, out_names, 1);

        double best = 1e30;
        for (int i = 0; i < iters; ++i) {
            auto t0 = std::chrono::steady_clock::now();
            auto out = session.Run(Ort::RunOptions{nullptr}, in_names, &in, 1, out_names, 1);
            auto t1 = std::chrono::steady_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            if (ms < best)
                best = ms;
        }
        r.best_ms = best;
        r.ok = true;
    } catch (const Ort::Exception& e) {
        r.err = e.what();
    } catch (const std::exception& e) {
        r.err = e.what();
    }
    return r;
}

} // namespace

void run_image_spike() {
    _set_se_translator([](unsigned int code, EXCEPTION_POINTERS*) {
        char b[48];
        snprintf(b, sizeof(b), "SEH 0x%08X", code);
        throw std::runtime_error(b);
    });

    log_output("[xllama] image-spike: plain ORT DirectML conv model (~309 GFLOP/forward)\n");

    std::wstring model_path = utf8_to_wstring(resolve_local_path("imgspike.onnx"));

    SpikeRun dml, cpu;
    try {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "imgspike");
        dml = run_ep(env, model_path, /*use_dml=*/true, /*iters=*/5);
        cpu = run_ep(env, model_path, /*use_dml=*/false, /*iters=*/3);
    } catch (const std::exception& e) {
        log_output(std::string("[xllama] image-spike: env/setup error: ") + e.what() + "\n");
        return;
    }

    auto gflops = [](const SpikeRun& s) {
        return s.ok && s.best_ms > 0 ? kGFlop / (s.best_ms / 1000.0) : 0.0;
    };
    double dml_gfs = gflops(dml), cpu_gfs = gflops(cpu);
    double speedup = (dml.ok && cpu.ok && dml.best_ms > 0) ? cpu.best_ms / dml.best_ms : 0.0;

    char lb[320];
    snprintf(lb, sizeof(lb),
             "[xllama] image-spike: DML %.1f ms (%.0f GFLOP/s, ok=%d) | CPU %.1f ms (%.0f "
             "GFLOP/s, ok=%d) | GPU speedup %.1fx\n",
             dml.best_ms, dml_gfs, dml.ok, cpu.best_ms, cpu_gfs, cpu.ok, speedup);
    log_output(lb);
    if (!dml.ok)
        log_output("[xllama] image-spike: DML error: " + dml.err + "\n");
    if (!cpu.ok)
        log_output("[xllama] image-spike: CPU error: " + cpu.err + "\n");

    // Write result CSV (+ .done marker), fetchable via Device Portal.
    FILE* fp = _wfopen(utf8_to_wstring(resolve_local_path("imgspike-result.csv")).c_str(), L"w");
    if (fp) {
        time_t now = time(nullptr);
        char date_buf[32];
        strftime(date_buf, sizeof(date_buf), "%Y-%m-%dT%H:%M:%SZ", gmtime(&now));
        fputs("ep,ok,forward_ms,gflop_s,resolution,gflop_forward,speedup_vs_cpu,date\n", fp);
        fprintf(fp, "directml,%d,%.2f,%.1f,%dx%d,%.1f,%.2f,%s\n", dml.ok, dml.best_ms, dml_gfs, kW,
                kH, kGFlop, speedup, date_buf);
        fprintf(fp, "cpu,%d,%.2f,%.1f,%dx%d,%.1f,%.2f,%s\n", cpu.ok, cpu.best_ms, cpu_gfs, kW, kH,
                kGFlop, 1.0, date_buf);
        fclose(fp);
        FILE* done =
            _wfopen(utf8_to_wstring(resolve_local_path("imgspike-result.csv.done")).c_str(), L"w");
        if (done) {
            fputs("done\n", done);
            fclose(done);
        }
        log_output("[xllama] imgspike-result.csv written\n");
    }
}

} // namespace xllama::bridge

#endif // XLLAMA_UWP
