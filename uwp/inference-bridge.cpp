// Copyright (c) 2024 Venere Labs
// SPDX-License-Identifier: MIT

#include "inference-bridge.h"

#include "xllama/inference.h"
#include "xllama/path_utils.h"
#include "xllama/platform.h"
#include "xllama/utf8_utils.h"

#include <cstdio>
#include <string>

namespace xllama::bridge {

// ---------------------------------------------------------------------------
// main_loop (called from UWP bench mode background thread)
// ---------------------------------------------------------------------------

void main_loop() {
#ifdef XLLAMA_UWP
    // Read prompt from LocalFolder/prompt.txt, fallback to default.
    // SmolLM2-360M-Instruct uses ChatML format; bare text triggers EOS immediately.
    std::string user_prompt = "Hello from Xbox Series S. Tell me about your architecture.";
    {
        std::string prompt_path = resolve_local_path("prompt.txt");
        FILE* pf = _wfopen(utf8_to_wstring(prompt_path).c_str(), L"r");
        if (pf) {
            char buf[8192] = {};
            size_t n = fread(buf, 1, sizeof(buf) - 1, pf);
            fclose(pf);
            if (n > 0) {
                user_prompt = buf;
                // Strip trailing whitespace/newlines
                while (!user_prompt.empty() &&
                       (user_prompt.back() == '\n' || user_prompt.back() == '\r' ||
                        user_prompt.back() == ' '))
                    user_prompt.pop_back();
            }
        }
    }
    // Wrap with ChatML template (required for SmolLM2-Instruct).
    std::string prompt = "<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n"
                         "<|im_start|>user\n" +
                         user_prompt +
                         "<|im_end|>\n"
                         "<|im_start|>assistant\n";

    // Read model directory/filename from LocalFolder/model.txt, fallback to default.
    std::string model_name = "smollm2-360m-cpu-int4";
    {
        std::string model_cfg = resolve_local_path("model.txt");
        FILE* mf = _wfopen(utf8_to_wstring(model_cfg).c_str(), L"r");
        if (mf) {
            char buf[512] = {};
            size_t n = fread(buf, 1, sizeof(buf) - 1, mf);
            fclose(mf);
            if (n > 0) {
                model_name = buf;
                while (!model_name.empty() &&
                       (model_name.back() == '\n' || model_name.back() == '\r' ||
                        model_name.back() == ' '))
                    model_name.pop_back();
            }
        }
    }

    log_output("[xllama] bench model: " + model_name + "\n");
    log_output("[xllama] bench prompt: " + prompt.substr(0, 80) + "...\n");

    InferenceParams params;
    params.model_path = model_name;
    params.prompt = prompt;
    params.n_predict = 512;

    InferenceResult res = ::xllama::run_inference(params);
    xllama::write_bench_csv(params, res, "xbox-series-s");
#endif
}

} // namespace xllama::bridge
