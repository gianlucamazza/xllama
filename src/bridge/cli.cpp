// Copyright (c) 2024 Venere Labs
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
                 "      --chat           Wrap the prompt with the model's chat template\n"
                 "                       (ChatML/Gemma by model name) and stop on its stop token\n"
                 "      --batch <N>      Logical prefill batch; 0 = llama default 2048\n"
                 "      --ubatch <N>     Physical prefill chunk (TTFT sweep knob);\n"
                 "                       0 = llama default 512\n"
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

    static const struct option long_opts[] = {{"model", required_argument, nullptr, 'm'},
                                              {"prompt", required_argument, nullptr, 'p'},
                                              {"n-predict", required_argument, nullptr, 'n'},
                                              {"ctx", required_argument, nullptr, 'c'},
                                              {"threads", required_argument, nullptr, 't'},
                                              {"temp", required_argument, nullptr, 1},
                                              {"seed", required_argument, nullptr, 2},
                                              {"chat", no_argument, nullptr, 3},
                                              {"batch", required_argument, nullptr, 4},
                                              {"ubatch", required_argument, nullptr, 5},
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
        case 'h':
            print_help(argv[0]);
            std::exit(0);
        default:
            print_usage(argv[0]);
            return false;
        }
    }

    if (out.model_path.empty() || out.prompt.empty()) {
        print_usage(argv[0]);
        return false;
    }
    return true;
}

} // namespace xllama
