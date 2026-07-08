// Copyright (c) 2024 Venere Labs
// SPDX-License-Identifier: MIT

#pragma once

#include "xllama/inference.h"
#include "xllama/inference_params.h"

namespace xllama::bridge {

// Thin wrappers kept for backward compatibility with existing UWP code.
using InferenceParams = xllama::InferenceParams;
using InferenceResult = xllama::InferenceResult;

inline InferenceResult run_inference(const InferenceParams& params) {
    return xllama::run_inference(params);
}

// Called from UWP App on a background thread (bench mode).
void main_loop();

// Image-generation spike (Stage: image DirectML). Runs a compute-bound conv
// model (proxy for a diffusion UNet step) through the plain ONNX Runtime
// DirectML EP and a CPU EP control, measuring forward-pass latency + GFLOP/s to
// test whether the GPU wins on compute-bound fp16 batch workloads (unlike M=1
// text decode). Triggered by LocalFolder\image.flag in the headless path.
void run_image_spike();
void run_diffuse();

} // namespace xllama::bridge
