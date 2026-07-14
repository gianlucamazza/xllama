// Copyright (c) 2024 Venere Labs
// SPDX-License-Identifier: MIT

#include "xllama/chat_prompt.h"
#include "xllama/cli.h"
#include "xllama/inference.h"
#include "xllama/membw.h"

#include <cstdio>

int main(int argc, char** argv) {
    xllama::InferenceParams params;
    if (!xllama::parse_cli_args(argc, argv, params))
        return 1;

    // --membw: model-free CPU memory-bandwidth micro-bench. Runs a single-thread
    // pass and a full-width pass so the ratio (scaling) is visible; prints the CSV
    // row so it can be appended to a results file.
    if (params.run_membw) {
        const xllama::MembwResult st = xllama::measure_membw(/*bytes=*/0x10000000, 5, 1);
        const xllama::MembwResult mt = xllama::measure_membw(/*bytes=*/0x10000000, 5, 0);
        std::printf("membw (best of 5, %zu MB buffer)\n", st.buffer_bytes / (1024 * 1024));
        std::printf("  1 thread : read %.1f  copy %.1f  triad %.1f GB/s\n", st.read_gbs,
                    st.copy_gbs, st.triad_gbs);
        std::printf("  %d threads: read %.1f  copy %.1f  triad %.1f GB/s\n", mt.threads,
                    mt.read_gbs, mt.copy_gbs, mt.triad_gbs);
        std::printf("%s%s", xllama::membw_csv_header(),
                    xllama::format_membw_row(mt, "host").c_str());
        return 0;
    }

    // --chat: wrap the raw prompt with the model's chat template (ChatML or
    // Gemma, selected by model name) and stop on its stop token. Without this the
    // CLI feeds the prompt verbatim and generates to n_predict.
    if (params.chat_template) {
        const xllama::ChatFormat fmt = xllama::chat_format_for(params.model_path);
        params.prompt = fmt.render_prompt(/*system=*/"", /*history=*/{}, params.prompt);
        params.stop_sequences = fmt.stop_sequences;
    }

    auto res = xllama::run_inference(params);
    return res.success ? 0 : 1;
}
