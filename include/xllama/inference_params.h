#pragma once

#include <atomic>
#include <functional>
#include <string>

namespace xllama {

// ---------------------------------------------------------------------------
// Inference configuration
// ---------------------------------------------------------------------------
struct InferenceParams {
    std::string model_path;  // Linux: absolute path; UWP: filename in LocalFolder
    std::string prompt;
    int         n_predict   = 128;
    int         n_ctx       = 2048;
    int         n_threads   = 0;    // 0 = auto-detect
    float       temperature = 0.8f;
    uint32_t    seed        = 0xFFFFFFFF; // 0xFFFFFFFF = LLAMA_DEFAULT_SEED

    // UI callbacks (optional). Called from the inference thread — must marshal
    // to the UI thread before touching XAML controls.
    std::function<void(const std::string&)> on_status; // e.g. "loading model"
    std::function<void(const std::string&)> on_token;  // per-token text piece

    // Set to true from the UI thread to request early termination.
    std::atomic<bool>* abort_flag = nullptr;
};

struct InferenceResult {
    bool        success          = false;
    double      t_load_ms        = 0.0;
    double      t_p_eval_ms      = 0.0;
    double      t_eval_ms        = 0.0;
    int         n_p_eval         = 0;
    int         n_eval           = 0;
    size_t      peak_ws_mb       = 0;
    std::string output_text;
    std::string error_msg;
};

} // namespace xllama
