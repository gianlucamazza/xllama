// Copyright (c) 2024 Venere Labs
// SPDX-License-Identifier: MIT

#include "xllama/chat_prompt.h"
#include "xllama/cli.h"
#include "xllama/inference.h"

int main(int argc, char** argv) {
    xllama::InferenceParams params;
    if (!xllama::parse_cli_args(argc, argv, params))
        return 1;

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
