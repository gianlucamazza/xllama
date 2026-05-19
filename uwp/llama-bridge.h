#pragma once

#include "xllama/inference_params.h"

namespace xllama::bridge {

// Thin wrappers kept for backward compatibility with existing UWP code.
using InferenceParams  = xllama::InferenceParams;
using InferenceResult  = xllama::InferenceResult;

inline InferenceResult run_inference(const InferenceParams& params) {
    return xllama::run_inference(params);
}

// Called from UWP App::Run() on a background thread.
// Reads config from LocalFolder/config files (or uses defaults).
// On Linux: no-op.
void main_loop();

} // namespace xllama::bridge
