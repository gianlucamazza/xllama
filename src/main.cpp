// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include "xllama/chat_prompt.h" // kDefaultSystemPrompt
#include "xllama/cli.h"
#include "xllama/inference.h"
#include "xllama/membw.h"
#include "xllama/training.h"
#ifdef XLLAMA_DEVICE_TRAIN
    #include "xllama/device_train.h"
#endif

#include <cstdio>
#include <cstdlib>
#include <string>

int main(int argc, char** argv) {
    xllama::InferenceParams params;
    params.echo_stdout = true; // interactive CLI streams pieces as they decode
    if (!xllama::parse_cli_args(argc, argv, params))
        return 1;

    // --training-capabilities: RE-backed matrix (no model).
    if (params.run_training_capabilities) {
        const xllama::TrainingCapabilityInfo* caps = nullptr;
        const size_t n = xllama::training_capabilities(&caps);
        std::printf("xllama training capabilities (see docs/training-architecture.md)\n");
        std::printf("%-32s %-10s %-12s %s\n", "name", "available", "status", "reason");
        for (size_t i = 0; i < n; ++i) {
            const auto& c = caps[i];
            std::printf("%-32s %-10s %-12s %s\n", c.name, c.available ? "yes" : "no", c.status,
                        c.reason);
        }
        return 0;
    }

    // --validate-train-job: training pillar — parse + validate job JSON only.
    if (params.run_validate_train_job) {
        xllama::TrainingJob job;
        std::string err;
        if (!xllama::load_training_job_file(params.train_job_path, job, &err)) {
            std::fprintf(stderr, "validate-train-job FAIL: %s\n", err.c_str());
            return 1;
        }
        std::printf("validate-train-job PASS: %s\n",
                    xllama::format_training_job_summary(job).c_str());
        return 0;
    }

    // --train-job: partial_ft runs in-process (Lane B engine); lora_peft
    // shells out to the host exploration runner (PEFT + merge + eval).
    if (params.run_train_job) {
        xllama::TrainingJob job;
        std::string err;
        if (!xllama::load_training_job_file(params.train_job_path, job, &err)) {
            std::fprintf(stderr, "train-job validate FAIL: %s\n", err.c_str());
            return 1;
        }
#ifdef XLLAMA_DEVICE_TRAIN
        if (job.method == xllama::TrainMethod::PartialFt) {
            std::fprintf(stderr, "train-job: %s\n",
                         xllama::format_training_job_summary(job).c_str());
            xllama::DeviceTrainCallbacks cb;
            cb.on_status = [](const std::string& line) {
                std::fprintf(stderr, "train-job: %s\n", line.c_str());
            };
            const xllama::TrainingResult r = xllama::run_device_train_job(job, cb);
            if (!r.success) {
                std::fprintf(stderr, "train-job FAIL: %s\n", r.error_msg.c_str());
                return 1;
            }
            std::printf("train-job PASS: merged=%s last_loss=%.4f wall=%.1fs peak_ws=%zuMB\n",
                        r.merged_gguf_path.c_str(), r.last_loss, r.wall_seconds, r.peak_ws_mb);
            return 0;
        }
#endif
        const char* runner = std::getenv("XLLAMA_TRAIN_RUNNER");
        std::string cmd;
        if (runner && runner[0] != '\0') {
            cmd = std::string(runner) + " " + params.train_job_path;
        } else {
            // Default: repo-relative host runner (cwd expected = repo root).
            cmd = std::string("training/host/run_job.sh ") + params.train_job_path;
        }
        std::fprintf(stderr, "train-job: %s\n", xllama::format_training_job_summary(job).c_str());
        std::fprintf(stderr, "train-job: exec %s\n", cmd.c_str());
        const int rc = std::system(cmd.c_str());
        if (rc != 0) {
            std::fprintf(stderr, "train-job FAIL: runner exit %d\n", rc);
            return rc == -1 ? 1 : (rc >> 8);
        }
        return 0;
    }

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
        // --system overrides it; the default matches the chat UI and the API
        // endpoint, because an empty system turn makes small instruct models
        // hallucinate the next role instead of answering (see api-server.cpp).
        const std::string system = params.system_prompt.empty()
                                       ? std::string(xllama::kDefaultSystemPrompt)
                                       : params.system_prompt;
        params.prompt = fmt.render_prompt(system, /*history=*/{}, params.prompt);
        params.stop_sequences = fmt.stop_sequences;
    }

    auto res = xllama::run_inference(params);
    return res.success ? 0 : 1;
}
