#include "llama-bridge.h"

#include "xllama/inference.h"
#include "xllama/platform.h"
#include "xllama/path_utils.h"
#include "xllama/utf8_utils.h"

#include <cstdio>
#include <string>

namespace xllama::bridge {

// ---------------------------------------------------------------------------
// main_loop (called from UWP App::Run background thread)
// ---------------------------------------------------------------------------

void main_loop() {
#ifdef XLLAMA_UWP
    // Read prompt from LocalFolder/prompt.txt, fallback to default.
    std::string prompt = "Hello from Xbox Series S. Tell me about your architecture.";
    {
        std::string prompt_path = resolve_local_path("prompt.txt");
#ifdef _WIN32
        FILE* pf = _wfopen(utf8_to_wstring(prompt_path).c_str(), L"r");
#else
        FILE* pf = std::fopen(prompt_path.c_str(), "r");
#endif
        if (pf) {
            char buf[8192] = {};
            size_t n = fread(buf, 1, sizeof(buf) - 1, pf);
            fclose(pf);
            if (n > 0) prompt = buf;
        }
    }

    // Read model filename from LocalFolder/model.txt, fallback to default.
    std::string model_filename = "qwen3-1.7b-Q4_K_M.gguf";
    {
        std::string model_cfg = resolve_local_path("model.txt");
#ifdef _WIN32
        FILE* mf = _wfopen(utf8_to_wstring(model_cfg).c_str(), L"r");
#else
        FILE* mf = std::fopen(model_cfg.c_str(), "r");
#endif
        if (mf) {
            char buf[512] = {};
            size_t n = fread(buf, 1, sizeof(buf) - 1, mf);
            fclose(mf);
            if (n > 0) {
                model_filename = buf;
                // trim trailing whitespace
                while (!model_filename.empty() &&
                       (model_filename.back() == '\n' || model_filename.back() == '\r' ||
                        model_filename.back() == ' '))
                    model_filename.pop_back();
            }
        }
    }

    log_output("[xllama] model: " + model_filename + "\n");
    log_output("[xllama] prompt: " + prompt.substr(0, 80) + "...\n");

    InferenceParams params;
    params.model_path = model_filename;
    params.prompt     = prompt;
    params.n_predict  = 128;

    InferenceResult res = run_inference(params);
    xllama::write_bench_csv(params, res, "xbox-series-s");
#endif
    // Linux: no-op (use xllama-cli directly)
}

} // namespace xllama::bridge
