// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include "xllama/cli.h"

#include <cstdlib>
#include <getopt.h>
#include <string>

namespace xllama {

static void print_help(const char* prog) {
    std::fprintf(stdout,
                 "xllama-cli — llama.cpp bridge for Xbox Series S (Linux dev build)\n\n"
                 "Usage: %s -m <model.gguf> -p <prompt> [options]\n\n"
                 "Required:\n"
                 "  -m, --model <path>   Path to GGUF model file\n"
                 "  -p, --prompt <text>  Prompt text\n\n"
                 "Optional:\n"
                 "  -n, --n-predict <N>  Max tokens to generate (default: 128)\n"
                 "  -c, --ctx <N>        Context size (default: 2048)\n"
                 "  -t, --threads <N>    Thread count; 0 = auto (default: 0)\n"
                 "      --temp <float>   Sampling temperature (default: 0.8)\n"
                 "      --seed <int>     RNG seed; 0 = random (default: 0)\n"
                 "      --top-p <float>  Nucleus sampling cutoff (default: 0.9)\n"
                 "      --top-k <N>      Top-k cutoff (default: 40)\n"
                 "      --repetition-penalty <float>\n"
                 "                       Repetition penalty over the last 64 tokens\n"
                 "                       (default: 1.1); 0 disables the stage\n"
                 "      --system <text>  System message for --chat (default: the built-in\n"
                 "                       \"You are a helpful AI assistant.\")\n"
                 "      --chat           Wrap the prompt with the model's chat template\n"
                 "                       (ChatML, Gemma, Llama 3 or Phi 3 by model name;\n"
                 "                       Qwen no-think suffix) and stop on its stop token\n"
                 "      --batch <N>      Logical prefill batch; 0 = llama default 2048\n"
                 "      --ubatch <N>     Physical prefill chunk (TTFT sweep knob);\n"
                 "                       0 = llama default 512\n"
                 "      --kv-q8          q8_0 KV cache + flash attention (#171); falls\n"
                 "                       back to F16 KV if the arch refuses flash attn\n"
                 "      --prompt-lookup  Draft-free n-gram speculative decoding (GGUF;\n"
                 "                       Phase 15 W2 #210). Default off; host timings are\n"
                 "                       not product claims — console A/B decides ship\n"
                 "      --membw          Run the CPU memory-bandwidth micro-bench and\n"
                 "                       exit (no model needed); prints read/copy/triad GB/s\n"
                 "      --diskbw         Run the disk read-bandwidth micro-bench and exit\n"
                 "                       (no model needed); seq+rnd read GB/s, 4 GiB test file\n"
                 "      --gpubw          Phase 15 W3 (#211) GPU STREAM probe (D3D12; host\n"
                 "                       without D3D12 reports unavailable — no fake GB/s)\n"
                 "      --gpugemv        Phase 15 H6.1 (#228) Q4_K GEMV density probe\n"
                 "                       (D3D12; host without D3D12 reports unavailable)\n"
                 "      --ramceil        Probe how much heap this process can commit and\n"
                 "                       exit (no model needed); prints the CSV rows\n"
                 "      --greedy         Deterministic argmax decode (implies logit parity);\n"
                 "                       overrides --temp/--seed for reproducible output\n"
                 "      --dump-logits <path>\n"
                 "                       Write the last prefill-token logits (float32) to\n"
                 "                       <path> and metadata to <path>.json, then continue\n"
                 "      --validate-train-job <job.json>\n"
                 "                       Parse/validate a training job manifest and exit\n"
                 "                       (training pillar; no model load)\n"
                 "      --train-job <job.json>\n"
                 "                       Run lora_peft via the host runner, or partial_ft\n"
                 "                       in-process when the Lane B engine is compiled\n"
                 "      --training-capabilities\n"
                 "                       Print the training-pillar capability matrix\n"
                 "                       (available/experimental/designed/research/rejected)\n"
                 "      --lora <path>    GGUF LoRA adapter (llama.cpp only; runtime load)\n"
                 "      --lora-scale <f> LoRA scale (default: 1.0)\n"
                 "  -h, --help           Show this message\n",
                 prog);
}

static void print_usage(const char* prog) {
    std::fprintf(stderr, "Error: --model and --prompt are required.\n\n");
    print_help(prog);
}

bool parse_cli_args(int argc, char** argv, InferenceParams& out) {
    optind = 1; // reset getopt_long state for re-entrant use
    opterr = 0; // silence getopt error messages

    static const struct option long_opts[] = {
        {"model", required_argument, nullptr, 'm'},
        {"prompt", required_argument, nullptr, 'p'},
        {"n-predict", required_argument, nullptr, 'n'},
        {"ctx", required_argument, nullptr, 'c'},
        {"threads", required_argument, nullptr, 't'},
        {"temp", required_argument, nullptr, 1},
        {"seed", required_argument, nullptr, 2},
        {"chat", no_argument, nullptr, 3},
        {"batch", required_argument, nullptr, 4},
        {"ubatch", required_argument, nullptr, 5},
        {"membw", no_argument, nullptr, 6},
        {"greedy", no_argument, nullptr, 7},
        {"dump-logits", required_argument, nullptr, 8},
        {"validate-train-job", required_argument, nullptr, 9},
        {"train-job", required_argument, nullptr, 10},
        {"training-capabilities", no_argument, nullptr, 11},
        {"lora", required_argument, nullptr, 12},
        {"lora-scale", required_argument, nullptr, 13},
        {"top-p", required_argument, nullptr, 14},
        {"top-k", required_argument, nullptr, 15},
        {"repetition-penalty", required_argument, nullptr, 16},
        {"system", required_argument, nullptr, 17},
        {"kv-q8", no_argument, nullptr, 18},
        {"ramceil", no_argument, nullptr, 19},
        {"prompt-lookup", no_argument, nullptr, 20},
        {"gpubw", no_argument, nullptr, 21},
        {"gpugemv", no_argument, nullptr, 22},
        {"diskbw", no_argument, nullptr, 23},
        {"help", no_argument, nullptr, 'h'},
        {nullptr, 0, nullptr, 0}};

    int opt;
    while ((opt = getopt_long(argc, argv, "m:p:n:c:t:h", long_opts, nullptr)) != -1) {
        switch (opt) {
        case 'm':
            out.model_path = optarg;
            break;
        case 'p':
            out.prompt = optarg;
            break;
        case 'n':
            out.n_predict = std::atoi(optarg);
            break;
        case 'c':
            out.n_ctx = std::atoi(optarg);
            break;
        case 't':
            out.n_threads = std::atoi(optarg);
            break;
        case 1:
            out.temperature = static_cast<float>(std::atof(optarg));
            break;
        case 2:
            out.seed = static_cast<uint32_t>(std::atoi(optarg));
            break;
        case 3:
            out.chat_template = true;
            break;
        case 4:
            out.n_batch = std::atoi(optarg);
            break;
        case 5:
            out.n_ubatch = std::atoi(optarg);
            break;
        case 6:
            out.run_membw = true;
            break;
        case 7:
            out.greedy = true;
            break;
        case 8:
            out.dump_logits_path = optarg;
            break;
        case 9:
            out.run_validate_train_job = true;
            out.train_job_path = optarg;
            break;
        case 10:
            out.run_train_job = true;
            out.train_job_path = optarg;
            break;
        case 11:
            out.run_training_capabilities = true;
            break;
        case 12:
            out.lora_path = optarg;
            break;
        case 13:
            out.lora_scale = static_cast<float>(std::atof(optarg));
            break;
        case 14:
            out.top_p = static_cast<float>(std::atof(optarg));
            break;
        case 15:
            out.top_k = std::atoi(optarg);
            break;
        case 16:
            out.repetition_penalty = static_cast<float>(std::atof(optarg));
            break;
        case 17:
            out.system_prompt = optarg;
            break;
        case 18:
            out.kv_q8 = true;
            break;
        case 19:
            out.run_ramceil = true;
            break;
        case 20:
            out.prompt_lookup = true;
            break;
        case 21:
            out.run_gpubw = true;
            break;
        case 22:
            out.run_gpugemv = true;
            break;
        case 23:
            out.run_diskbw = true;
            break;
        case 'h':
            print_help(argv[0]);
            std::exit(0);
        default:
            print_usage(argv[0]);
            return false;
        }
    }

    // --membw / --gpubw / --gpugemv / --diskbw / --ramceil / train-job: model/prompt
    // not required.
    if (out.run_membw)
        return true;
    if (out.run_gpubw)
        return true;
    if (out.run_gpugemv)
        return true;
    if (out.run_diskbw)
        return true;
    if (out.run_ramceil)
        return true;
    if (out.run_training_capabilities)
        return true;
    if (out.run_validate_train_job || out.run_train_job) {
        if (out.train_job_path.empty()) {
            std::fprintf(stderr, "Error: train job path is required.\n");
            return false;
        }
        return true;
    }

    if (out.model_path.empty() || out.prompt.empty()) {
        print_usage(argv[0]);
        return false;
    }
    return true;
}

} // namespace xllama
