// Copyright (c) 2024 Gianluca Mazza
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
    // Order matters: the first match wins, so a longer label must precede the
    // shorter one it contains ("ud-iq3_s" before "iq3_s", "iq3_xxs" before
    // "iq3_xs"). Getting that wrong silently truncates the label rather than
    // failing, which is how "UD-IQ3_S" would have been recorded as "IQ3_S".
    static constexpr const char* kTokens[] = {
        "ud-iq1_m",   "ud-iq1_s",  "ud-iq2_m",  "ud-iq2_s", "ud-iq3_m", "ud-iq3_s", "ud-iq4_xs",
        "ud-q2_k_xl", "ud-q3_k_m", "ud-q4_k_m", "q3_k_xl",  "q3_k_l",   "q3_k_m",   "q3_k_s",
        "q4_k_m",     "q4_k_s",    "q5_k_m",    "q5_k_s",   "q6_k",     "q8_0",     "iq1_m",
        "iq1_s",      "iq2_xxs",   "iq2_xs",    "iq2_m",    "iq2_s",    "iq3_xxs",  "iq3_xs",
        "iq3_m",      "iq3_s",     "iq4_xs",    "iq4_nl",   "q4_0",     "q5_0",     "q2_k",
        "fp16",       "int4",
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

    // No recognisable token. Say so, rather than guessing a plausible label: an
    // invented "Q4_K_M" reads as measured fact in the CSV and cannot be told from
    // a real one, while an explicit "unknown" is visible to both a reader and to
    // benchmark-summary.json (which can override it with `quant_label`). This
    // fallback silently mislabelled the first LFM2.5-8B-A1B run as Q4_K_M — a
    // quant whose 5156 MB would not even fit the measured heap ceiling.
    if (is_llama)
        return "unknown";
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

    // n_prompt_tok / n_gen_tok: the actual prefill and generated token counts.
    // Without them a row cannot be interpreted — the rates alone do not say at
    // what prompt length a run was measured, nor how many tokens it generated, so
    // turn time (prefill + decode) is not reconstructible. Both matter: generation
    // is capped by the context window (prompt + new <= n_ctx) and can stop early
    // on EOG, so n_gen_tok is NOT the requested n_predict. Measured 2026-07-21: a
    // 1574-token prompt at n_ctx 2048 caps new tokens at 474 and generated 277.
    // Placed before `host` so the positional parsing in bench-xbox-ort.sh
    // ($3 backend, $7 decode_tok_s) keeps working.
    // max_length: #130. On DirectML this is what actually governs prefill
    // throughput — the same 1289-token prompt runs at 130 tok/s with max_length
    // 1801 and 611 tok/s with 2048. n_ctx does not capture it (a control run at
    // n_ctx 3072 holding max_length at 1801 reproduced the slow figure), and
    // neither do n_prompt_tok and n_gen_tok, since generation can stop early on
    // EOG well below the cap. Appended before `host` for the same reason as the
    // token counts: bench-xbox-ort.sh parses $3 and $7 positionally.
    // run_index (W1.1/#F1) is appended LAST, after date: every earlier column is
    // parsed positionally somewhere (bench-xbox-ort.sh reads $3 backend, $7
    // decode_tok_s, $14 max_length), so a new field can only go on the end
    // without shifting them. 0 = single-run / legacy; a repeated bench writes the
    // repetition number so the summary generator can report a spread instead of a
    // pre-averaged point.
    const char* header = "model,quant,backend,n_ctx,n_threads,"
                         "prompt_tok_s,decode_tok_s,peak_ws_mb,load_ms,"
                         "gpu_mem_mb,gpu_budget_mb,n_prompt_tok,n_gen_tok,max_length,host,date,"
                         "run_index,prefill_ms,ttft_ms\n";
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
    fprintf(fp, "%s,%s,%s,%d,%d,%.2f,%.2f,%zu,%.0f,%zu,%zu,%d,%d,%d,%s,%s,%d,%.1f,%.1f\n",
            model_name.c_str(), quant.c_str(), backend, params.n_ctx, used_threads, prompt_tok_s,
            decode_tok_s, res.peak_ws_mb, res.t_load_ms, res.gpu_mem_mb, res.gpu_budget_mb,
            res.n_p_eval, res.n_eval, res.max_length, host_label ? host_label : "unknown", date_buf,
            params.run_index, res.t_p_eval_ms, res.t_first_token_ms);
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
