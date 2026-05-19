#pragma once

#include "xllama/inference_params.h"

namespace xllama {

// Run inference synchronously. Returns InferenceResult with metrics.
// On Linux: reads model_path as absolute filesystem path.
// On UWP: model_path is relative to ApplicationData::LocalFolder\\models
InferenceResult run_inference(const InferenceParams& params);

// Write bench CSV row to the resolved local path.
void write_bench_csv(const InferenceParams& params,
                     const InferenceResult& res,
                     const char* host_label = nullptr);

} // namespace xllama
