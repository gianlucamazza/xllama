// Copyright (c) 2024 Venere Labs
// SPDX-License-Identifier: MIT

#include "xllama/inference.h"
#include "xllama/path_utils.h"
#include "xllama/platform.h"
#include "xllama/utf8_utils.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <ctime>
#include <string>

namespace xllama {
namespace {

std::string to_lower_copy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

// Extract a quant / EP label from a model path or GGUF filename.
// Longer tokens first so Q3_K_XL wins over Q3_K_L / Q3, etc.
const char* quant_token_in(const std::string& haystack_lower) {
    static constexpr const char* kTokens[] = {
        "ud-iq2_m", "q3_k_xl", "q3_k_l", "q3_k_m", "q3_k_s", "q4_k_m", "q4_k_s",
        "q5_k_m",   "q5_k_s",  "q6_k",   "q8_0",   "iq2_m",  "iq3_m",  "iq3_xs",
        "iq4_xs",   "iq4_nl",  "q4_0",   "q5_0",   "q2_k",   "fp16",   "int4",
    };
    for (const char* tok : kTokens) {
        if (haystack_lower.find(tok) != std::string::npos)
            return tok;
    }
    return nullptr;
}

// Canonical CSV quant label (uppercase GGUF-style; ORT uses lowercase fp16/int4).
std::string format_quant_label(const char* token_lower) {
    if (!token_lower)
        return {};
    std::string s(token_lower);
    // ORT EP-style labels stay lowercase.
    if (s == "fp16" || s == "int4")
        return s;
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    return s;
}

std::string detect_quant_label(const std::string& model_path, bool is_llama) {
    // Prefer tokens in the path/dir name, then the first GGUF basename if any.
    std::string probe = model_path;
    if (is_llama) {
        const std::string resolved = resolve_model_path(model_path);
        const std::string gguf = first_gguf_in_dir(resolved);
        if (!gguf.empty()) {
            auto slash = gguf.find_last_of("/\\");
            probe = (slash != std::string::npos) ? gguf.substr(slash + 1) : gguf;
        }
    }
    // Also scan the original path (catalogue ids like llama32-3b-q3ks rarely
    // embed quant; GGUF filenames do).
    const std::string lower_path = to_lower_copy(model_path);
    const std::string lower_probe = to_lower_copy(probe);
    if (const char* t = quant_token_in(lower_probe))
        return format_quant_label(t);
    if (const char* t = quant_token_in(lower_path))
        return format_quant_label(t);

    if (is_llama)
        return "Q4_K_M"; // legacy GGUF default
    // ORT: dir-name convention smollm2-*-{cpu|dml}-{int4|fp16}
    if (lower_path.find("fp16") != std::string::npos)
        return "fp16";
    return "int4";
}

} // namespace

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

    const std::string quant = detect_quant_label(params.model_path, is_llama);

    const char* backend = "cpu";
#ifdef XLLAMA_USE_ORT
    if (!is_llama) {
        // Derive the EP label from the model directory name (convention:
        // smollm2-<size>-<cpu|dml>-<int4|fp16>). gpu_mem_mb > 0 corroborates DML.
        const bool is_dml = model_name.find("dml") != std::string::npos || res.gpu_mem_mb > 0;
        backend = is_dml ? "ort-genai-dml" : "ort-genai-cpu";
    }
#endif
    if (is_llama)
        backend = "cpu";

    const int used_threads = params.n_threads > 0
                                 ? params.n_threads
                                 : (is_llama ? detect_threads_llama() : detect_threads());
    fprintf(fp, "%s,%s,%s,%d,%d,%.2f,%.2f,%zu,%.0f,%zu,%zu,%s,%s\n", model_name.c_str(),
            quant.c_str(), backend, params.n_ctx, used_threads, prompt_tok_s, decode_tok_s,
            res.peak_ws_mb, res.t_load_ms, res.gpu_mem_mb, res.gpu_budget_mb,
            host_label ? host_label : "unknown", date_buf);
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
