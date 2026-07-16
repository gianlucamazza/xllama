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

// CPU memory-bandwidth micro-bench. Triggered by LocalFolder\membw.flag; writes
// membw-result.csv (+ .done marker) to LocalState. Pins the DRAM-bandwidth
// ceiling behind the bandwidth-bound decode number (see docs/benchmarks.md).
void run_membw();

// Diffusion pipeline (SD-Turbo on plain ORT DirectML). Triggered by
// LocalFolder\diffuse.flag (headless) or diffuse-inproc.flag (in-process
// experiment) — see diffuse.cpp for the model contract.
void run_diffuse();

// Logit-parity dump: greedy 1-token forward pass on the ORT backend, writing the
// last prefill-token logits to LocalState\logits.bin (+ .json sidecar, + a
// logits.done marker). Triggered by LocalFolder\logits.flag. Reads the raw prompt
// from prompt.txt and the model from model.txt (same config files as bench).
// scripts/validate-logit-parity.sh pulls the dump and diffs it against the
// llama.cpp golden via scripts/compare-logits.py.
void run_logits();

} // namespace xllama::bridge
