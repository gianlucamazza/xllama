#pragma once

#include <string>

namespace xllama::bridge {

struct InferenceParams {
    std::string model_path;  // Linux: absolute path; UWP: relative to LocalFolder/models/
    std::string prompt;
    int         n_predict  = 128;
    int         n_ctx      = 2048;
    int         n_threads  = 0;    // 0 = auto-detect
    float       temperature = 0.8f;
};

struct InferenceResult {
    bool        success          = false;
    double      t_load_ms        = 0.0;
    double      t_p_eval_ms      = 0.0;
    double      t_eval_ms        = 0.0;
    int         n_p_eval         = 0;
    int         n_eval           = 0;
    size_t      peak_ws_mb       = 0;   // peak working set in MB
    std::string output_text;
    std::string error_msg;
};

// Run inference synchronously. Returns InferenceResult with metrics.
// On Linux: reads model_path as absolute filesystem path.
// On UWP: model_path is relative to ApplicationData::LocalFolder\models\.
InferenceResult run_inference(const InferenceParams& params);

// Called from UWP App::Run() on a background thread.
// Reads config from LocalFolder/config.json (or uses defaults).
// On Linux: no-op.
void main_loop();

} // namespace xllama::bridge
