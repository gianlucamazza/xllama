#include "llama-bridge.h"

#include <cstdio>
#include <cstring>
#include <string>

static void print_usage(const char* prog) {
    std::fprintf(stderr,
        "xllama-cli — llama.cpp bridge for Xbox Series S (Linux dev build)\n\n"
        "Usage: %s -m <model.gguf> -p <prompt> [options]\n\n"
        "Options:\n"
        "  -m <path>   Path to GGUF model file\n"
        "  -p <text>   Prompt text\n"
        "  -n <int>    Max tokens to generate (default: 128)\n"
        "  --help      Show this message\n",
        prog);
}

int main(int argc, char** argv) {
    std::string model_path;
    std::string prompt;
    int n_predict = 128;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (std::strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (std::strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        } else if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n_predict = std::atoi(argv[++i]);
        } else {
            std::fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    if (model_path.empty() || prompt.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    xllama::bridge::InferenceParams params;
    params.model_path = model_path;
    params.prompt     = prompt;
    params.n_predict  = n_predict;

    return xllama::bridge::run_inference(params);
}
