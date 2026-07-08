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
// MODEL CONTRACT: expects an fp16 SD-Turbo-class ONNX model uploaded under
// LocalState\models\<dir>\{text_encoder,unet,vae_decoder}\model.onnx, each
// self-contained (< 2 GB, external data merged — docs/uwp-constraints.md §8), plus
// the CLIP tokenizer's vocab.json + merges.txt in LocalState\clip\. Float tensors
// are fp16; input_ids are int32; timestep is int64. Output tensors are read as
// either fp16 or fp32 (whichever the graph emits).

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
    #include <cstdint>
    #include <cstdio>
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

std::string read_prompt() {
    std::ifstream f(utf8_to_wstring(resolve_local_path("prompt.txt")).c_str());
    if (!f.good())
        return "a red sports car on a mountain road at sunset";
    std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' '))
        s.pop_back();
    return s;
}

std::string read_model_dir() {
    std::ifstream f(utf8_to_wstring(resolve_local_path("diffuse-model.txt")).c_str());
    if (!f.good())
        return "sd-turbo-fp16";
    std::string s;
    std::getline(f, s);
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' '))
        s.pop_back();
    return s.empty() ? "sd-turbo-fp16" : s;
}

} // namespace

void run_diffuse() {
    _set_se_translator([](unsigned int code, EXCEPTION_POINTERS*) {
        char b[48];
        snprintf(b, sizeof(b), "SEH 0x%08X", code);
        throw std::runtime_error(b);
    });

    const std::string dir = read_model_dir();
    const std::string prompt = read_prompt();
    log_output("[xllama] diffuse: model='" + dir + "' prompt='" + prompt + "'\n");

    try {
        auto model = [&](const char* comp) {
            return resolve_local_path("models\\" + dir + "\\" + comp + "\\model.onnx");
        };
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "diffuse");
        Ort::MemoryInfo mem = cpu_mem();

        // ---- Tokenize (host-validated CLIP BPE) -----------------------------
        auto tok = diffusion::ClipTokenizer::from_files(resolve_local_path("clip\\vocab.json"),
                                                        resolve_local_path("clip\\merges.txt"));
        std::vector<int> ids32 = tok.encode(prompt); // length 77, int32

        // ---- Text encoder: input_ids[1,77] i32 -> last_hidden_state[1,77,1024]
        Ort::Session te = make_session(env, model("text_encoder"));
        std::array<int64_t, 2> te_shape{1, kSeq};
        Ort::Value te_in = Ort::Value::CreateTensor<int32_t>(mem, ids32.data(), ids32.size(),
                                                             te_shape.data(), te_shape.size());
        const char* te_in_names[] = {"input_ids"};
        const char* te_out_names[] = {"last_hidden_state"};
        auto te_out = te.Run(Ort::RunOptions{nullptr}, te_in_names, &te_in, 1, te_out_names, 1);
        std::vector<float> hidden = read_output_f32(te_out[0]); // [1,77,1024]
        const int64_t hidden_dim =
            (int64_t)hidden.size() / kSeq; // 1024 (or 768 for SD1.x text encoders)
        std::vector<uint16_t> hidden_h = to_half(hidden);

        // ---- Scheduler + init latent ---------------------------------------
        diffusion::EulerDiscreteScheduler sched;
        sched.set_timesteps(1); // SD-Turbo: single step
        const size_t latent_n = (size_t)kLatentC * kLatentHW * kLatentHW;
        std::vector<float> latent(latent_n);
        std::mt19937 rng(42); // fixed seed -> deterministic image
        std::normal_distribution<float> gauss(0.0f, 1.0f);
        for (auto& v : latent)
            v = gauss(rng) * (float)sched.init_noise_sigma();

        Ort::Session unet = make_session(env, model("unet"));
        Ort::Session vae = make_session(env, model("vae_decoder"));

        for (size_t s = 0; s < sched.timesteps().size(); ++s) {
            // scale_model_input on a copy (UNet sees the scaled latent).
            std::vector<float> scaled = latent;
            sched.scale_model_input(scaled);
            std::vector<uint16_t> sample_h = to_half(scaled);
            std::vector<uint16_t> hs_h = hidden_h;
            int64_t ts = (int64_t)sched.timesteps()[s];

            std::array<int64_t, 4> s_shape{1, kLatentC, kLatentHW, kLatentHW};
            std::array<int64_t, 1> t_shape{1};
            std::array<int64_t, 3> h_shape{1, kSeq, hidden_dim};
            Ort::Value u_sample = Ort::Value::CreateTensor(
                mem, sample_h.data(), sample_h.size() * sizeof(uint16_t), s_shape.data(),
                s_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
            Ort::Value u_ts =
                Ort::Value::CreateTensor<int64_t>(mem, &ts, 1, t_shape.data(), t_shape.size());
            Ort::Value u_hs = Ort::Value::CreateTensor(
                mem, hs_h.data(), hs_h.size() * sizeof(uint16_t), h_shape.data(), h_shape.size(),
                ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
            Ort::Value u_ins[] = {std::move(u_sample), std::move(u_ts), std::move(u_hs)};
            const char* u_in_names[] = {"sample", "timestep", "encoder_hidden_states"};
            const char* u_out_names[] = {"out_sample"};
            auto u_out = unet.Run(Ort::RunOptions{nullptr}, u_in_names, u_ins, 3, u_out_names, 1);
            std::vector<float> noise = read_output_f32(u_out[0]); // epsilon [1,4,64,64]

            sched.step(noise, latent); // latent <- previous sample
        }

        // ---- VAE decode: latent/scale [1,4,64,64] -> sample[1,3,512,512] -----
        for (auto& v : latent)
            v = v / (float)kVaeScale;
        std::vector<uint16_t> latent_h = to_half(latent);
        std::array<int64_t, 4> v_shape{1, kLatentC, kLatentHW, kLatentHW};
        Ort::Value v_in = Ort::Value::CreateTensor(
            mem, latent_h.data(), latent_h.size() * sizeof(uint16_t), v_shape.data(),
            v_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
        const char* v_in_names[] = {"latent_sample"};
        const char* v_out_names[] = {"sample"};
        auto v_out = vae.Run(Ort::RunOptions{nullptr}, v_in_names, &v_in, 1, v_out_names, 1);
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
