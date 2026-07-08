// Copyright (c) 2024 Venere Labs
// SPDX-License-Identifier: MIT
// Diffusion image generation: plain ONNX Runtime DirectML, three sessions
// (text_encoder + UNet + VAE decoder) driven by the host-validated CLIP tokenizer
// and Euler scheduler. Runs in the headless path (diffuse.flag) so no XAML
// compositor D3D12 device collides with ORT's DML device (the 887A0036 finding),
// mirroring the image spike (image.flag).
//
// The correctness-critical logic (tokenizer, scheduler, fp16 conversion, PNG) is
// unit-tested on the host against the diffusers/transformers reference
// (tests/test_diffusion.cpp). This file is the ORT DirectML orchestration around
// it — CI-compile-validated; runtime-validated on console per
// docs/console-validation-runbook.md §7.
//
// MODEL CONTRACT: an SD-Turbo-class ONNX model (epsilon prediction, Euler trailing
// schedule, guidance-free) under LocalState\models\<dir>\{text_encoder,unet,
// vae_decoder}\model.onnx, each self-contained (< 2 GB, external data merged —
// docs/uwp-constraints.md §8), plus the CLIP tokenizer's vocab.json + merges.txt
// in LocalState\clip\. Tensor dtypes are ADAPTIVE: float inputs are fed as fp16 or
// fp32 and input_ids as int32 or int64, matching what each session declares
// (covers both a pure-fp16 export and a keep_io_types fp32-boundary one); outputs
// are read as fp16 or fp32. Classifier-free guidance is NOT implemented — SD-Turbo
// runs guidance-free (scale 0); SD1.x/LCM-class models would need CFG (two text
// embeddings + guidance blend), out of scope for the console demo.
//
// Inputs (LocalState): prompt.txt, diffuse-model.txt (model dir name),
// diffuse-steps.txt (default 1), diffuse-seed.txt (default 42).
// Outputs: diffuse-out.png (+ .done) and diffuse-result.csv with per-stage ms.

#include "inference-bridge.h"

#ifdef XLLAMA_UWP

// clang-format off
    #include <windows.h>
    #include <eh.h>
    #include <onnxruntime_cxx_api.h>
    #include <dml_provider_factory.h>
// clang-format on

    #include "xllama/diffusion/clip_tokenizer.h"
    #include "xllama/diffusion/euler_scheduler.h"
    #include "xllama/diffusion/half.h"
    #include "xllama/diffusion/png_writer.h"
    #include "xllama/path_utils.h"
    #include "xllama/platform.h"
    #include "xllama/utf8_utils.h"

    #include <algorithm>
    #include <array>
    #include <chrono>
    #include <cstdint>
    #include <cstdio>
    #include <cstring>
    #include <ctime>
    #include <fstream>
    #include <iterator>
    #include <random>
    #include <string>
    #include <vector>

namespace xllama::bridge {

namespace {

using xllama::diffusion::from_half;
using xllama::diffusion::to_half;

constexpr int kLatentC = 4;
constexpr int kLatentHW = 64; // 64x64 latent -> 512x512 image (VAE upsamples x8)
constexpr int kImageHW = 512;
constexpr int kSeq = 77; // CLIP context length
constexpr double kVaeScale = 0.18215;

// A DML session with sequential execution + no mem pattern (DML requirements).
Ort::Session make_session(Ort::Env& env, const std::string& path) {
    Ort::SessionOptions so;
    so.SetExecutionMode(ORT_SEQUENTIAL);
    so.DisableMemPattern();
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_DML(so, /*device_id=*/0));
    return Ort::Session(env, utf8_to_wstring(path).c_str(), so);
}

Ort::MemoryInfo cpu_mem() {
    return Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
}

// Read a run output (index 0) into a float vector, converting from fp16 or fp32.
std::vector<float> read_output_f32(Ort::Value& v) {
    auto info = v.GetTensorTypeAndShapeInfo();
    const size_t n = info.GetElementCount();
    std::vector<float> out(n);
    if (info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
        const auto* p = reinterpret_cast<const uint16_t*>(v.GetTensorData<Ort::Float16_t>());
        out = from_half(p, n);
    } else {
        const float* p = v.GetTensorData<float>();
        std::copy(p, p + n, out.begin());
    }
    return out;
}

// Declared element type of a session input, looked up by name (input order is not
// guaranteed). UNDEFINED if the name is absent.
ONNXTensorElementDataType input_type_by_name(Ort::Session& s, const char* name) {
    Ort::AllocatorWithDefaultOptions alloc;
    const size_t n = s.GetInputCount();
    for (size_t i = 0; i < n; ++i) {
        auto nm = s.GetInputNameAllocated(i, alloc);
        if (std::strcmp(nm.get(), name) == 0)
            return s.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetElementType();
    }
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
}

// A float tensor fed as fp16 or fp32 depending on what the session declares.
// Owns the backing storage; keep alive until after Run. (Moving is safe: vector
// moves preserve the heap buffer the Ort::Value points into.)
struct FloatInput {
    std::vector<float> f32;
    std::vector<uint16_t> f16;
    Ort::Value value{nullptr};
};

FloatInput make_float_input(Ort::MemoryInfo& mem, std::vector<float> data, const int64_t* shape,
                            size_t rank, bool as_fp16) {
    FloatInput t;
    if (as_fp16) {
        t.f16 = to_half(data);
        t.value = Ort::Value::CreateTensor(mem, t.f16.data(), t.f16.size() * sizeof(uint16_t),
                                           shape, rank, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
    } else {
        t.f32 = std::move(data);
        t.value = Ort::Value::CreateTensor<float>(mem, t.f32.data(), t.f32.size(), shape, rank);
    }
    return t;
}

std::string read_text_file(const char* name, const std::string& fallback) {
    std::ifstream f(utf8_to_wstring(resolve_local_path(name)).c_str());
    if (!f.good())
        return fallback;
    std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' '))
        s.pop_back();
    return s.empty() ? fallback : s;
}

int read_int_file(const char* name, int fallback, int lo, int hi) {
    const std::string s = read_text_file(name, "");
    if (s.empty())
        return fallback;
    const int v = std::atoi(s.c_str());
    return (v >= lo && v <= hi) ? v : fallback;
}

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

} // namespace

void run_diffuse() {
    _set_se_translator([](unsigned int code, EXCEPTION_POINTERS*) {
        char b[48];
        snprintf(b, sizeof(b), "SEH 0x%08X", code);
        throw std::runtime_error(b);
    });

    const std::string dir = read_text_file("diffuse-model.txt", "sd-turbo-fp16");
    const std::string prompt =
        read_text_file("prompt.txt", "a red sports car on a mountain road at sunset");
    const int steps = read_int_file("diffuse-steps.txt", 1, 1, 50);
    const int seed = read_int_file("diffuse-seed.txt", 42, 0, 1'000'000'000);
    log_output("[xllama] diffuse: model='" + dir + "' steps=" + std::to_string(steps) +
               " seed=" + std::to_string(seed) + " prompt='" + prompt + "'\n");

    try {
        auto model = [&](const char* comp) {
            return resolve_local_path("models\\" + dir + "\\" + comp + "\\model.onnx");
        };
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "diffuse");
        Ort::MemoryInfo mem = cpu_mem();

        // ---- Tokenize (host-validated CLIP BPE) -----------------------------
        auto tok = diffusion::ClipTokenizer::from_files(resolve_local_path("clip\\vocab.json"),
                                                        resolve_local_path("clip\\merges.txt"));
        const std::vector<int> ids = tok.encode(prompt); // length 77

        // ---- Text encoder: input_ids[1,77] -> last_hidden_state[1,77,H] -----
        auto t_load0 = std::chrono::steady_clock::now();
        Ort::Session te = make_session(env, model("text_encoder"));
        std::array<int64_t, 2> te_shape{1, kSeq};
        // input_ids dtype is export-dependent (optimum: int32; others: int64).
        const auto ids_type = input_type_by_name(te, "input_ids");
        std::vector<int32_t> ids32(ids.begin(), ids.end());
        std::vector<int64_t> ids64(ids.begin(), ids.end());
        Ort::Value te_in =
            (ids_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)
                ? Ort::Value::CreateTensor<int64_t>(mem, ids64.data(), ids64.size(),
                                                    te_shape.data(), te_shape.size())
                : Ort::Value::CreateTensor<int32_t>(mem, ids32.data(), ids32.size(),
                                                    te_shape.data(), te_shape.size());
        const char* te_in_names[] = {"input_ids"};
        const char* te_out_names[] = {"last_hidden_state"};
        auto t_te0 = std::chrono::steady_clock::now();
        auto te_out = te.Run(Ort::RunOptions{nullptr}, te_in_names, &te_in, 1, te_out_names, 1);
        const double te_ms = ms_since(t_te0);
        std::vector<float> hidden = read_output_f32(te_out[0]);   // [1,77,H]
        const int64_t hidden_dim = (int64_t)hidden.size() / kSeq; // 1024 (SD2-class) or 768 (SD1.x)

        // ---- Scheduler + init latent ---------------------------------------
        diffusion::EulerDiscreteScheduler sched;
        sched.set_timesteps(steps);
        const size_t latent_n = (size_t)kLatentC * kLatentHW * kLatentHW;
        std::vector<float> latent(latent_n);
        std::mt19937 rng((unsigned)seed);
        std::normal_distribution<float> gauss(0.0f, 1.0f);
        for (auto& v : latent)
            v = gauss(rng) * (float)sched.init_noise_sigma();

        Ort::Session unet = make_session(env, model("unet"));
        Ort::Session vae = make_session(env, model("vae_decoder"));
        const bool unet_fp16 =
            input_type_by_name(unet, "sample") == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
        const bool vae_fp16 =
            input_type_by_name(vae, "latent_sample") == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
        // timestep dtype is export-dependent too: int64 (optimum default) or
        // float16/float32 (the ORT-team fp16 export declares timestep:f16 — the
        // values used here, e.g. 999, are integers < 2048, exact in fp16).
        const auto ts_type = input_type_by_name(unet, "timestep");
        log_output(std::string("[xllama] diffuse: unet input ") + (unet_fp16 ? "fp16" : "fp32") +
                   ", vae input " + (vae_fp16 ? "fp16" : "fp32") + ", hidden_dim " +
                   std::to_string(hidden_dim) + ", load " + std::to_string((int)ms_since(t_load0)) +
                   " ms\n");

        std::array<int64_t, 4> s_shape{1, kLatentC, kLatentHW, kLatentHW};
        std::array<int64_t, 1> t_shape{1};
        std::array<int64_t, 3> h_shape{1, kSeq, hidden_dim};
        double unet_ms = 0.0;
        for (size_t s = 0; s < sched.timesteps().size(); ++s) {
            // scale_model_input on a copy (UNet sees the scaled latent).
            std::vector<float> scaled = latent;
            sched.scale_model_input(scaled);
            FloatInput u_sample =
                make_float_input(mem, std::move(scaled), s_shape.data(), s_shape.size(), unet_fp16);
            FloatInput u_hs =
                make_float_input(mem, hidden, h_shape.data(), h_shape.size(), unet_fp16);
            const double ts_val = sched.timesteps()[s];
            int64_t ts_i64 = (int64_t)ts_val;
            float ts_f32 = (float)ts_val;
            uint16_t ts_f16 = diffusion::float_to_half(ts_f32);
            Ort::Value u_ts{nullptr};
            if (ts_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
                u_ts =
                    Ort::Value::CreateTensor(mem, &ts_f16, sizeof(ts_f16), t_shape.data(),
                                             t_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
            } else if (ts_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
                u_ts = Ort::Value::CreateTensor<float>(mem, &ts_f32, 1, t_shape.data(),
                                                       t_shape.size());
            } else {
                u_ts = Ort::Value::CreateTensor<int64_t>(mem, &ts_i64, 1, t_shape.data(),
                                                         t_shape.size());
            }

            Ort::Value u_ins[] = {std::move(u_sample.value), std::move(u_ts),
                                  std::move(u_hs.value)};
            const char* u_in_names[] = {"sample", "timestep", "encoder_hidden_states"};
            const char* u_out_names[] = {"out_sample"};
            auto t_u0 = std::chrono::steady_clock::now();
            auto u_out = unet.Run(Ort::RunOptions{nullptr}, u_in_names, u_ins, 3, u_out_names, 1);
            unet_ms += ms_since(t_u0);
            std::vector<float> noise = read_output_f32(u_out[0]); // epsilon [1,4,64,64]

            sched.step(noise, latent); // latent <- previous sample
        }

        // ---- VAE decode: latent/scale [1,4,64,64] -> sample[1,3,512,512] -----
        for (auto& v : latent)
            v = v / (float)kVaeScale;
        FloatInput v_in =
            make_float_input(mem, std::move(latent), s_shape.data(), s_shape.size(), vae_fp16);
        const char* v_in_names[] = {"latent_sample"};
        const char* v_out_names[] = {"sample"};
        auto t_v0 = std::chrono::steady_clock::now();
        auto v_out = vae.Run(Ort::RunOptions{nullptr}, v_in_names, &v_in.value, 1, v_out_names, 1);
        const double vae_ms = ms_since(t_v0);
        std::vector<float> img = read_output_f32(v_out[0]); // [1,3,512,512] in [-1,1]

        // ---- CHW [-1,1] -> HWC uint8 RGB -> PNG -----------------------------
        const int HW = kImageHW * kImageHW;
        std::vector<uint8_t> rgb((size_t)3 * HW);
        for (int c = 0; c < 3; ++c)
            for (int i = 0; i < HW; ++i) {
                float x = img[(size_t)c * HW + i] * 0.5f + 0.5f; // [-1,1] -> [0,1]
                x = std::max(0.0f, std::min(1.0f, x));
                rgb[(size_t)i * 3 + c] = (uint8_t)(x * 255.0f + 0.5f);
            }
        // Encode with the host-tested writer, then write via the wide-path API
        // used throughout the UWP code (LocalState paths are wide).
        const std::vector<uint8_t> png = diffusion::encode_png_rgb(kImageHW, kImageHW, rgb);
        FILE* pf = _wfopen(utf8_to_wstring(resolve_local_path("diffuse-out.png")).c_str(), L"wb");
        if (pf) {
            fwrite(png.data(), 1, png.size(), pf);
            fclose(pf);
        }

        const double total_ms = te_ms + unet_ms + vae_ms;
        char lb[256];
        snprintf(lb, sizeof(lb),
                 "[xllama] diffuse: te %.0f ms | unet %.0f ms (%zu step, %.0f ms/step) | vae "
                 "%.0f ms | total %.0f ms\n",
                 te_ms, unet_ms, sched.timesteps().size(),
                 unet_ms / (double)sched.timesteps().size(), vae_ms, total_ms);
        log_output(lb);

        // Result CSV (+ .done marker), fetchable via Device Portal.
        FILE* fp = _wfopen(utf8_to_wstring(resolve_local_path("diffuse-result.csv")).c_str(), L"w");
        if (fp) {
            time_t now = time(nullptr);
            char date_buf[32];
            strftime(date_buf, sizeof(date_buf), "%Y-%m-%dT%H:%M:%SZ", gmtime(&now));
            fputs("model,steps,seed,te_ms,unet_ms,unet_ms_per_step,vae_ms,total_ms,resolution,"
                  "date\n",
                  fp);
            fprintf(fp, "%s,%d,%d,%.1f,%.1f,%.1f,%.1f,%.1f,%dx%d,%s\n", dir.c_str(), steps, seed,
                    te_ms, unet_ms, unet_ms / (double)sched.timesteps().size(), vae_ms, total_ms,
                    kImageHW, kImageHW, date_buf);
            fclose(fp);
        }
        if (pf) {
            FILE* done =
                _wfopen(utf8_to_wstring(resolve_local_path("diffuse-out.png.done")).c_str(), L"w");
            if (done) {
                fputs("done\n", done);
                fclose(done);
            }
            log_output("[xllama] diffuse: wrote diffuse-out.png (" + std::to_string(kImageHW) +
                       "x" + std::to_string(kImageHW) + ")\n");
        } else {
            log_output("[xllama] diffuse: PNG write failed\n");
        }
    } catch (const Ort::Exception& e) {
        log_output(std::string("[xllama] diffuse: ORT error: ") + e.what() + "\n");
    } catch (const std::exception& e) {
        log_output(std::string("[xllama] diffuse: error: ") + e.what() + "\n");
    }
}

} // namespace xllama::bridge

#endif // XLLAMA_UWP
