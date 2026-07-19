// op-repro.cpp — single-op CPU-vs-DML repro runner (#111).
//
// Purpose: isolate suspected-broken DML kernels (first target: the
// (Skip)SimplifiedLayerNormalization -> DML MVN2 UseMean=false path behind
// #91) with the smallest possible on-device experiment: one ONNX model, one
// fixed input payload, the same forward pass on the CPU EP and on the DML EP,
// raw outputs pulled back to the host for diffing.
//
// Contract (all files in LocalState, driven by scripts/validate-op-repro.sh):
//   oprepro.flag       -> triggers this mode at launch (consumed by App.cpp)
//   repro.onnx         -> model with STATIC input shapes only
//   repro-input.bin    -> fp32 payload: model inputs concatenated in session
//                         input order; converted per-tensor to the input dtype
//                         (fp16 or fp32)
//   repro-out-cpu.bin  -> all outputs of the CPU run, fp32, concatenated
//   repro-out-dml.bin  -> same for the DML run
//   repro.done         -> "ok" or "error:<message>"

#include "inference-bridge.h"

#ifdef XLLAMA_UWP

// clang-format off
    #include <windows.h>
    #include <onnxruntime_cxx_api.h>
    #include <dml_provider_factory.h>
// clang-format on

    #include <cstdio>
    #include <string>
    #include <vector>

    #include "xllama/diffusion/half.h"
    #include "xllama/path_utils.h"
    #include "xllama/platform.h"
    #include "xllama/utf8_utils.h"

namespace xllama::bridge {

namespace {

std::vector<float> read_f32_file(const std::string& path) {
    std::vector<float> out;
    FILE* fp = _wfopen(utf8_to_wstring(path).c_str(), L"rb");
    if (!fp)
        return out;
    fseek(fp, 0, SEEK_END);
    long bytes = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    out.resize(static_cast<size_t>(bytes) / sizeof(float));
    if (fread(out.data(), 1, static_cast<size_t>(bytes), fp) != static_cast<size_t>(bytes))
        out.clear();
    fclose(fp);
    return out;
}

void write_f32_file(const std::string& path, const std::vector<float>& data) {
    FILE* fp = _wfopen(utf8_to_wstring(path).c_str(), L"wb");
    if (!fp)
        return;
    fwrite(data.data(), sizeof(float), data.size(), fp);
    fclose(fp);
}

void write_done(const std::string& msg) {
    FILE* fp = _wfopen(utf8_to_wstring(resolve_local_path("repro.done")).c_str(), L"w");
    if (fp) {
        fputs(msg.c_str(), fp);
        fclose(fp);
    }
}

// Run repro.onnx once on the given session: slice the fp32 payload across the
// model inputs (converting to fp16 where the input dtype asks for it) and
// return every output converted to fp32, concatenated in output order.
std::vector<float> run_once(Ort::Session& session, const std::vector<float>& payload) {
    Ort::AllocatorWithDefaultOptions alloc;
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    const size_t n_in = session.GetInputCount();
    std::vector<Ort::AllocatedStringPtr> in_name_holders;
    std::vector<const char*> in_names;
    std::vector<Ort::Value> in_values;
    // Keep converted buffers alive until Run(). Reserve up front: the created
    // Ort::Value wraps the buffer's data() pointer, so a later push_back must
    // never reallocate the outer vector (dangling tensor data otherwise).
    std::vector<std::vector<uint16_t>> half_buffers;
    std::vector<std::vector<float>> float_buffers;
    half_buffers.reserve(n_in);
    float_buffers.reserve(n_in);
    in_name_holders.reserve(n_in);

    size_t cursor = 0;
    for (size_t i = 0; i < n_in; ++i) {
        in_name_holders.push_back(session.GetInputNameAllocated(i, alloc));
        in_names.push_back(in_name_holders.back().get());

        auto info = session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo();
        auto shape = info.GetShape();
        size_t count = 1;
        for (int64_t d : shape) {
            if (d <= 0)
                throw std::runtime_error("repro.onnx must use static input shapes");
            count *= static_cast<size_t>(d);
        }
        if (cursor + count > payload.size())
            throw std::runtime_error("repro-input.bin shorter than the model inputs");

        const ONNXTensorElementDataType dt = info.GetElementType();
        if (dt == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
            std::vector<uint16_t> half(count);
            for (size_t k = 0; k < count; ++k)
                half[k] = ::xllama::diffusion::float_to_half(payload[cursor + k]);
            half_buffers.push_back(std::move(half));
            in_values.push_back(Ort::Value::CreateTensor(
                mem, half_buffers.back().data(), count * sizeof(uint16_t), shape.data(),
                shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16));
        } else if (dt == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            float_buffers.emplace_back(payload.begin() + cursor, payload.begin() + cursor + count);
            in_values.push_back(Ort::Value::CreateTensor<float>(mem, float_buffers.back().data(),
                                                                count, shape.data(), shape.size()));
        } else {
            throw std::runtime_error("unsupported repro input dtype (want fp16/fp32)");
        }
        cursor += count;
    }

    const size_t n_out = session.GetOutputCount();
    std::vector<Ort::AllocatedStringPtr> out_name_holders;
    std::vector<const char*> out_names;
    for (size_t i = 0; i < n_out; ++i) {
        out_name_holders.push_back(session.GetOutputNameAllocated(i, alloc));
        out_names.push_back(out_name_holders.back().get());
    }

    auto outputs = session.Run(Ort::RunOptions{nullptr}, in_names.data(), in_values.data(), n_in,
                               out_names.data(), n_out);

    std::vector<float> flat;
    for (auto& out : outputs) {
        auto info = out.GetTensorTypeAndShapeInfo();
        const size_t count = info.GetElementCount();
        if (info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
            const auto* p = out.GetTensorData<uint16_t>();
            auto conv = ::xllama::diffusion::from_half(p, count);
            flat.insert(flat.end(), conv.begin(), conv.end());
        } else {
            const float* p = out.GetTensorData<float>();
            flat.insert(flat.end(), p, p + count);
        }
    }
    return flat;
}

} // namespace

void run_oprepro() {
    set_cwd_to_local_folder();
    log_output("[xllama] oprepro: single-op CPU-vs-DML repro run\n");
    try {
        const std::string model_path = resolve_local_path("repro.onnx");
        const std::vector<float> payload = read_f32_file(resolve_local_path("repro-input.bin"));
        if (payload.empty())
            throw std::runtime_error("repro-input.bin missing or empty");
        log_output("[xllama] oprepro: payload " + std::to_string(payload.size()) + " f32\n");

        Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "oprepro");
        log_output("[xllama] oprepro: env ready\n");

        {
            Ort::SessionOptions so;
            so.SetGraphOptimizationLevel(ORT_DISABLE_ALL); // the op itself, not a rewrite
            Ort::Session cpu(env, utf8_to_wstring(model_path).c_str(), so);
            log_output("[xllama] oprepro: CPU session created\n");
            write_f32_file(resolve_local_path("repro-out-cpu.bin"), run_once(cpu, payload));
            log_output("[xllama] oprepro: CPU run done\n");
        }
        {
            // Same shape as the diffusion sessions (diffuse.cpp make_session),
            // with optimizations off so DML executes the recorded op as-is.
            Ort::SessionOptions so;
            so.SetExecutionMode(ORT_SEQUENTIAL);
            so.DisableMemPattern();
            so.SetGraphOptimizationLevel(ORT_DISABLE_ALL);
            Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_DML(so, 0));
            Ort::Session dml(env, utf8_to_wstring(model_path).c_str(), so);
            log_output("[xllama] oprepro: DML session created\n");
            write_f32_file(resolve_local_path("repro-out-dml.bin"), run_once(dml, payload));
            log_output("[xllama] oprepro: DML run done\n");
        }

        write_done("ok");
        log_output("[xllama] oprepro: done\n");
    } catch (const Ort::Exception& e) {
        log_output(std::string("[xllama] oprepro ORT error: ") + e.what() + "\n");
        write_done(std::string("error:") + e.what());
    } catch (const std::exception& e) {
        log_output(std::string("[xllama] oprepro error: ") + e.what() + "\n");
        write_done(std::string("error:") + e.what());
    }
}

} // namespace xllama::bridge

#else

namespace xllama::bridge {
void run_oprepro() {}
} // namespace xllama::bridge

#endif // XLLAMA_UWP
