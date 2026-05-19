#pragma once

#include <string>

namespace xllama::bridge {

struct InferenceParams {
    std::string model_path;
    std::string prompt;
    int         n_predict  = 128;
    int         n_ctx      = 2048;
    int         n_threads  = 0;    // 0 = auto-detect
    float       temperature = 0.8f;
};

// Run inference synchronously. Returns 0 on success, non-zero on error.
// On Linux: reads model from model_path directly via fopen.
// On UWP: model_path must be relative to ApplicationData::LocalFolder;
//         use CreateFileMappingFromApp instead of POSIX mmap (Phase 1).
int run_inference(const InferenceParams& params);

// Called from UWP App::Run() on a background thread (no-op on Linux).
void main_loop();

} // namespace xllama::bridge
