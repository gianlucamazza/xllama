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
                         "prompt_tok_s,decode_tok_s,peak_ws_mb,load_ms,host,date\n";
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

    std::string model_name = params.model_path;
    auto dot = model_name.rfind('.');
    if (dot != std::string::npos)
        model_name = model_name.substr(0, dot);

#ifdef XLLAMA_USE_ORT
    const char* backend = "directml";
    const char* quant   = "int4-awq";
#else
    const char* backend = "cpu";
    const char* quant   = "Q4_K_M";
#endif
    fprintf(fp, "%s,%s,%s,%d,%d,%.2f,%.2f,%zu,%.0f,%s,%s\n", model_name.c_str(), quant, backend,
            params.n_ctx, params.n_threads > 0 ? params.n_threads : detect_threads(), prompt_tok_s,
            decode_tok_s, res.peak_ws_mb, res.t_load_ms, host_label ? host_label : "unknown",
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
