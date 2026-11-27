// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include "xllama/cli.h"
#include "xllama/api_policy.h"   // validate_api_number
#include "xllama/api_sampling.h" // find_api_sampling_field
#include "xllama/local_knob.h"   // parse_knob_i64
#include "xllama/session_spec.h" // kGgufNThreadsMax

#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <getopt.h>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>

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
                 "      --seed <int>     RNG seed; -1 = random (default: -1)\n"
                 "      --top-p <float>  Nucleus sampling cutoff (default: 0.9)\n"
                 "      --top-k <N>      Top-k cutoff (default: 40)\n"
                 "      --repetition-penalty <float>\n"
                 "                       Repetition penalty over the last 64 tokens\n"
                 "                       (default: 1.1); 0 disables the stage\n"
                 "      --system <text>  System message for --chat (default: the built-in\n"
                 "                       \"You are a helpful AI assistant.\")\n"
                 "      --chat           Wrap the prompt with the model's chat template\n"
                 "                       (ChatML, Gemma, Llama 3, Phi 3 or Granite 4 by\n"
                 "                       model name; Qwen no-think suffix) and stop on its\n"
                 "                       stop token\n"
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
                 "      --gpugemv        Phase 15 H6.2 (#228) Q4_K GEMV density probe\n"
                 "                       (host: tiny-tile wave32 TDD; Linux reports d3d12\n"
                 "                       unavailable — no fake GB/s. Console A/B is gpugemv.flag)\n"
                 "      --gpu-layers <n> GGUF layers on the d3d12 backend (0 = CPU; Windows\n"
                 "                       only — Linux has no D3D12 device and stays on CPU)\n"
                 "      --gpustep        GGUF GPU decode probe D1 (docs/gguf-gpu-decode.md):\n"
                 "                       prints the cost-model projection per model; Linux\n"
                 "                       reports d3d12 unavailable. Console run is gpustep.flag\n"
                 "      --gpustep-verdict <csv>\n"
                 "                       Evaluate a gpustep CSV against the D1 gates and print\n"
                 "                       the ladder (no model needed)\n"
                 "      --ramceil        Probe how much heap this process can commit and\n"
                 "                       exit (no model needed); prints the CSV rows\n"
                 "      --greedy         Deterministic argmax decode (implies logit parity);\n"
                 "                       overrides --temp/--seed for reproducible output\n"
                 "      --dump-logits <path>\n"
                 "                       Write the last prefill-token logits (float32) to\n"
                 "                       <path> and metadata to <path>.json, then continue\n"
                 "      --dump-logits-stride <N>\n"
                 "                       With --dump-logits: also write every Nth prompt\n"
                 "                       position's logits to <path>.positions.bin\n"
                 "      --capture-tensors <dir>\n"
                 "                       #420 diagnostic: write prefill tensors matching\n"
                 "                       --capture-spec to <dir> (llama.cpp path only)\n"
                 "      --capture-spec <patterns[;mode=last|full][;max_mb=N]>\n"
                 "                       Graph tensor names, e.g. 'l_out-*,result_output'.\n"
                 "                       ';matmul_inputs=l_out-0' adds the first layer's\n"
                 "                       weight MUL_MATs and their operands (ADR 0010)\n"
                 "      --validate-train-job <job.json>\n"
                 "                       Parse/validate a training job manifest and exit\n"
                 "                       (training pillar; no model load)\n"
                 "      --train-job <job.json>\n"
                 "                       Run lora_peft via the host runner, or partial_ft\n"
                 "                       in-process when the Lane B engine is compiled\n"
                 "      --training-capabilities\n"
                 "                       Print the training-pillar capability matrix\n"
                 "                       (available/experimental/designed/research/rejected)\n"
                 "      --embed          Run embedding smoke test: load model, embed one or\n"
                 "                       more inputs, print width/L2-norm/tokens/peak-RSS.\n"
                 "                       Requires -m <model.gguf>. Pass inputs via -p <text>\n"
                 "                       (repeatable). Optional: --dimensions <N> (0=native)\n"
                 "      --dimensions <N> Requested embedding dimensions for --embed (0=native)\n"
                 "      --lora <path>    GGUF LoRA adapter (llama.cpp only; runtime load)\n"
                 "      --lora-scale <f> LoRA scale (default: 1.0)\n"
                 "      --decode-progress-every <N>\n"
                 "                       Log decode progress every N tokens (default: 64;\n"
                 "                       0 disables). Also XLLAMA_DECODE_PROGRESS_EVERY\n"
                 "      --decode-progress-ms <ms>\n"
                 "                       Log decode progress every T ms (default: 15000;\n"
                 "                       0 disables). Also XLLAMA_DECODE_PROGRESS_MS\n"
                 "      --infer-stall-timeout <s>\n"
                 "                       Abort if prefill/decode makes no progress for S\n"
                 "                       seconds (default: 180; 0 disables). Also\n"
                 "                       XLLAMA_INFER_STALL_TIMEOUT_S\n"
                 "  -h, --help           Show this message\n",
                 prog);
}

static void print_usage(const char* prog) {
    std::fprintf(stderr, "Error: --model and --prompt are required.\n\n");
    print_help(prog);
}

// Whole numbers: the same reader as the LocalState knobs (local_knob.h),
// base 10 with an optional sign, nothing after the digits; then the option's
// range.
template <class Int>
static bool parse_whole(const char* text, Int& out, long long lo, long long hi) {
    const std::optional<long long> v = parse_knob_i64(text);
    if (!v || *v < lo || *v > hi)
        return false;
    out = static_cast<Int>(*v);
    return true;
}

// A finite decimal, the whole text, independent of the locale, parsed in the
// destination's type so a value it cannot hold (1e100 or 1e-100 for a float)
// is refused rather than rounded to infinity or zero.
template <class Real = double> static std::optional<double> parse_real(std::string_view text) {
    if (!text.empty() && text.front() == '+') {
        text.remove_prefix(1);
        if (!text.empty() && text.front() == '-')
            return std::nullopt; // "+-2"
    }
    Real v = 0;
    const char* end = text.data() + text.size();
    const auto parsed = std::from_chars(text.data(), end, v);
    if (text.empty() || parsed.ec != std::errc{} || parsed.ptr != end || !std::isfinite(v))
        return std::nullopt;
    return static_cast<double>(v);
}

// A sampling option takes the bounds a LAN API request has (api_sampling.h),
// so the CLI, the API and the catalogue refuse the same values. |error| gets
// the table's message, e.g. "temperature must be between 0 and 2".
template <class T>
static bool parse_sampling(const char* key, const char* text, T& out, std::string& error) {
    const ApiSamplingField* field = find_api_sampling_field(key);
    using Real = std::conditional_t<std::is_floating_point_v<T>, T, double>;
    const std::optional<double> v = parse_real<Real>(text);
    if (!field || !v) {
        error = std::string(key) + " must be a number";
        return false;
    }
    error = validate_api_number(key, *v, field->lo, field->hi, field->whole);
    if (!error.empty())
        return false;
    out = static_cast<T>(*v);
    return true;
}

bool parse_cli_args(int argc, char** argv, InferenceParams& out) {
    optind = 1; // reset getopt_long state for re-entrant use
    opterr = 0; // silence getopt error messages

    // Env first so CLI flags win. Session/UWP apply the same overlay at their edge.
    {
        InferenceProgressConfig cfg = out.progress();
        overlay_progress_env(cfg);
        out.decode_log_every_n = cfg.decode_log_every_n;
        out.decode_log_every_ms = cfg.decode_log_every_ms;
        out.stall_timeout_s = cfg.stall_timeout_s;
    }

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
        {"embed", no_argument, nullptr, 24},
        {"dimensions", required_argument, nullptr, 25},
        {"gpustep", no_argument, nullptr, 26},
        {"gpustep-verdict", required_argument, nullptr, 27},
        {"gpu-layers", required_argument, nullptr, 28},
        {"decode-progress-every", required_argument, nullptr, 29},
        {"decode-progress-ms", required_argument, nullptr, 30},
        {"infer-stall-timeout", required_argument, nullptr, 31},
        {"capture-tensors", required_argument, nullptr, 32},
        {"capture-spec", required_argument, nullptr, 33},
        {"dump-logits-stride", required_argument, nullptr, 34},
        {"help", no_argument, nullptr, 'h'},
        {nullptr, 0, nullptr, 0}};

    for (;;) {
        int long_index = -1;
        const int opt = getopt_long(argc, argv, "m:p:n:c:t:h", long_opts, &long_index);
        if (opt == -1)
            break;
        bool valid = true;
        std::string sampling_error; // the API table's message for a sampling option
        switch (opt) {
        case 'm':
            out.model_path = optarg;
            break;
        case 'p':
            out.prompt = optarg;
            out.embed_inputs.push_back(optarg); // Collect for --embed mode
            break;
        case 'n':
            valid = parse_whole(optarg, out.n_predict, 0, std::numeric_limits<int>::max());
            break;
        case 'c':
            valid = parse_whole(optarg, out.n_ctx, 0, std::numeric_limits<int>::max());
            break;
        case 't':
            valid = parse_whole(optarg, out.n_threads, 0, kGgufNThreadsMax);
            break;
        case 1:
            valid = parse_sampling("temperature", optarg, out.temperature, sampling_error);
            break;
        case 2: {
            // Signed 32-bit seeds retain their prior bit pattern (-1 is random).
            // The unsigned spelling can address every GGUF seed directly.
            int64_t seed{};
            valid = parse_sampling("seed", optarg, seed, sampling_error);
            if (valid)
                out.seed = static_cast<uint32_t>(seed);
            break;
        }
        case 3:
            out.chat_template = true;
            break;
        case 4:
            valid = parse_whole(optarg, out.n_batch, 0, std::numeric_limits<int>::max());
            break;
        case 5:
            valid = parse_whole(optarg, out.n_ubatch, 0, std::numeric_limits<int>::max());
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
            if (const std::optional<double> v = parse_real<float>(optarg))
                out.lora_scale = static_cast<float>(*v);
            else
                valid = false;
            break;
        case 14:
            valid = parse_sampling("top_p", optarg, out.top_p, sampling_error);
            break;
        case 15:
            valid = parse_sampling("top_k", optarg, out.top_k, sampling_error);
            break;
        case 16:
            valid = parse_sampling("repetition_penalty", optarg, out.repetition_penalty,
                                   sampling_error);
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
        case 24:
            out.run_embed = true;
            break;
        case 25:
            valid = parse_whole(optarg, out.embed_dimensions, 0, std::numeric_limits<int>::max());
            break;
        case 26:
            out.run_gpustep = true;
            break;
        case 27:
            out.gpustep_verdict_csv = optarg;
            break;
        case 28:
            valid = parse_whole(optarg, out.n_gpu_layers, 0, std::numeric_limits<int>::max());
            break;
        case 29:
            valid = parse_whole(optarg, out.decode_log_every_n, 0, std::numeric_limits<int>::max());
            break;
        case 30:
            valid =
                parse_whole(optarg, out.decode_log_every_ms, 0, std::numeric_limits<int>::max());
            break;
        case 31:
            valid = parse_whole(optarg, out.stall_timeout_s, 0, std::numeric_limits<int>::max());
            break;
        case 32:
            out.tensor_capture_dir = optarg;
            break;
        case 33:
            out.tensor_capture_spec = optarg;
            break;
        case 34:
            valid = parse_whole(optarg, out.dump_logits_stride, 1, std::numeric_limits<int>::max());
            break;
        case 'h':
            print_help(argv[0]);
            std::exit(0);
        default:
            print_usage(argv[0]);
            return false;
        }
        if (!valid) {
            const std::string flag = long_index >= 0
                                         ? std::string("--") + long_opts[long_index].name
                                         : std::string("-") + static_cast<char>(opt);
            std::cerr << "Invalid " << flag << " '" << optarg << "'";
            if (!sampling_error.empty())
                std::cerr << ": " << sampling_error;
            std::cerr << "\n";
            return false;
        }
    }

    // --membw / --gpubw / --gpugemv / --gpustep / --diskbw / --ramceil / train-job / --embed:
    // model/prompt not required.
    if (out.run_membw)
        return true;
    if (out.run_gpubw)
        return true;
    if (out.run_gpugemv)
        return true;
    if (out.run_gpustep || !out.gpustep_verdict_csv.empty())
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
    if (out.run_embed) {
        if (out.model_path.empty()) {
            std::fprintf(stderr, "Error: --embed requires -m <model.gguf>.\n");
            return false;
        }
        if (out.embed_inputs.empty()) {
            std::fprintf(stderr, "Error: --embed requires at least one -p <input>.\n");
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
