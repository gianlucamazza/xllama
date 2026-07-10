// Copyright (c) 2024 Venere Labs
// SPDX-License-Identifier: MIT

#include "xllama/inference.h"
#include "xllama/path_utils.h"
#include "xllama/platform.h"
#include "xllama/utf8_utils.h"

#include <cstdio>
#include <ctime>
#include <string>

namespace xllama {

// ---------------------------------------------------------------------------
// Write bench CSV row
// ---------------------------------------------------------------------------

void write_bench_csv(const InferenceParams& params, const InferenceResult& res,
                     const char* host_label) {
    if (!res.success)
        return;

    const std::string csv_path = resolve_local_path("bench-result.csv");

#ifdef XLLAMA_UWP
    FILE* fp = _wfopen(utf8_to_wstring(csv_path).c_str(), L"w");
#else
    FILE* fp = std::fopen(csv_path.c_str(), "w");
#endif
    if (!fp)
        return;

    const char* header = "model,quant,backend,n_ctx,n_threads,"
                         "prompt_tok_s,decode_tok_s,peak_ws_mb,load_ms,"
                         "gpu_mem_mb,gpu_budget_mb,host,date\n";
    fputs(header, fp);

    double prompt_tok_s = (res.n_p_eval > 0 && res.t_p_eval_ms > 0)
                              ? static_cast<double>(res.n_p_eval) / (res.t_p_eval_ms / 1000.0)
                              : 0.0;
    double decode_tok_s = (res.n_eval > 0 && res.t_eval_ms > 0)
                              ? static_cast<double>(res.n_eval) / (res.t_eval_ms / 1000.0)
                              : 0.0;

    time_t now = time(nullptr);
    char date_buf[32];
    struct tm* tm_utc = gmtime(&now);
    strftime(date_buf, sizeof(date_buf), "%Y-%m-%dT%H:%M:%SZ", tm_utc);

    // Model label = the basename. Do NOT blind-strip at the last '.': model dir
    // names contain dots (e.g. "smollm2-1.7b-cpu-int4" was truncated to
    // "smollm2-1"). Take the basename and strip only a trailing *known model*
    // extension (.gguf for llama.cpp, .onnx if a file path is passed) — the ORT
    // path passes a directory with no extension, which is left intact.
    std::string model_name = params.model_path;
    auto slash = model_name.find_last_of("/\\");
    if (slash != std::string::npos)
        model_name = model_name.substr(slash + 1);
    for (const std::string& ext : {std::string(".gguf"), std::string(".onnx")}) {
        if (model_name.size() > ext.size() &&
            model_name.compare(model_name.size() - ext.size(), ext.size(), ext) == 0) {
            model_name = model_name.substr(0, model_name.size() - ext.size());
            break;
        }
    }

    // Which backend actually ran this model (unified builds dispatch at runtime).
#if defined(XLLAMA_USE_ORT) && defined(XLLAMA_USE_LLAMA)
    const bool is_llama = model_uses_llama_backend(params.model_path);
#elif defined(XLLAMA_USE_LLAMA)
    const bool is_llama = true;
#else
    const bool is_llama = false;
#endif

#ifdef XLLAMA_USE_ORT
    // Derive the EP + quant labels from the model directory name (convention:
    // smollm2-<size>-<cpu|dml>-<int4|fp16>) instead of hardcoding — a DML fp16
    // model was previously mislabelled as "ort-genai-cpu"/"int4". gpu_mem_mb > 0
    // corroborates DML execution (GPU memory resident after load).
    const bool is_dml = model_name.find("dml") != std::string::npos || res.gpu_mem_mb > 0;
    const bool is_fp16 = model_name.find("fp16") != std::string::npos;
    const char* backend = is_dml ? "ort-genai-dml" : "ort-genai-cpu";
    const char* quant = is_fp16 ? "fp16" : "int4";
#else
    const char* backend = "cpu";
    const char* quant = "Q4_K_M";
#endif
    if (is_llama) {
        // GGUF run: keep the llamacpp-lane labels (matches phase35-llamacpp-scaling.csv)
        // even in unified builds, where the ORT labels above would win.
        backend = "cpu";
        quant = "Q4_K_M";
    }

    const int used_threads = params.n_threads > 0
                                 ? params.n_threads
                                 : (is_llama ? detect_threads_llama() : detect_threads());
    fprintf(fp, "%s,%s,%s,%d,%d,%.2f,%.2f,%zu,%.0f,%zu,%zu,%s,%s\n", model_name.c_str(), quant,
            backend, params.n_ctx, used_threads, prompt_tok_s, decode_tok_s, res.peak_ws_mb,
            res.t_load_ms, res.gpu_mem_mb, res.gpu_budget_mb, host_label ? host_label : "unknown",
            date_buf);
    fclose(fp);

    // Write done marker
    const std::string done_path = resolve_local_path("bench-result.csv.done");
#ifdef XLLAMA_UWP
    FILE* done = _wfopen(utf8_to_wstring(done_path).c_str(), L"w");
#else
    FILE* done = std::fopen(done_path.c_str(), "w");
#endif
    if (done) {
        fputs("done\n", done);
        fclose(done);
    }

    log_output("[xllama] bench-result.csv written\n");
}

} // namespace xllama
