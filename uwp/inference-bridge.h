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

} // namespace xllama::bridge
