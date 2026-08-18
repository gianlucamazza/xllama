// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#pragma once

#include "xllama/device_train.h"
#include "xllama/inference.h"
#include "xllama/inference_params.h"
#include "xllama/training_params.h"

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

// Phase 15 W3 (#211): own D3D12 compute STREAM read (~1 GB VRAM) + checksum.
// Triggered by LocalFolder\gpubw.flag; writes gpubw-result.csv (+ .done).
// Kill criterion 100 GB/s is documented in docs/phase15-re-opt.md — not invented here.
void run_gpubw();

// Phase 15 H6.1 (#228): Q4_K GEMV density probe (dequant in register).
// Triggered by LocalFolder\gpugemv.flag; writes gpugemv-result.csv (+ .done).
// Soft density gate 40 GB/s packed — docs/phase15-re-opt.md; not a product backend.
void run_gpugemv();

// Heap-ceiling probe. Triggered by LocalFolder\ramceil.flag; writes
// ramceil-result.csv (+ .done marker holding the stop reason) to LocalState.
// Measures how much heap the process can actually commit under GameOS — the
// number that decides which model quants are admissible, since GGUF weights
// are read into the heap and not mapped (see include/xllama/ramceil.h).
void run_ramceil();

// Phase 16 WS-F (card H16.6): can an AppContainer app on GameOS capture audio?
// Triggered by LocalFolder\mic.flag; writes mic-result.json (+ .done) holding
// the WinRT status enums BY NAME, not a boolean — AccessDenied (sandbox says
// no) and DeviceNotAvailable (no headset plugged in) are different answers, and
// only the first is a verdict on WS-F. See docs/uwp-constraints.md.
void run_mic_probe();

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

// Single-op CPU-vs-DML repro (#111): loads LocalState\repro.onnx with a plain
// ORT CPU session and a DML session, feeds repro-input.bin and writes
// repro-out-cpu.bin / repro-out-dml.bin (+ repro.done marker). Triggered by
// LocalFolder\oprepro.flag; driven by scripts/validate-op-repro.sh. Used to
// isolate broken DML kernels (first target: (Skip)SimplifiedLayerNormalization
// -> MVN2 UseMean=false, the #91 root cause).
void run_oprepro();

// Lane B on-device training (ggml-opt partial FT). Triggered by
// LocalFolder\train.flag; reads the job from LocalState\training\job.json
// (paths inside the job are resolved relative to LocalState), runs
// prepare → train → export → evaluate in-process on the CPU, and writes
// <out_dir>\result.json plus a training\result.done marker. See
// docs/training-architecture.md Lane B and scripts/validate-console-training.sh
// device-train.
void run_train();

// Shared runner used by headless train.flag and the in-app personalize path
// (#116). Localizes relative job paths to LocalState, writes
// training/progress.json on each progress tick, and returns the engine result.
// Does NOT exit the process — safe to call while XAML is up.
// XLLAMA_DEVICE_TRAIN builds only; otherwise returns success=false.
xllama::TrainingResult run_train_job_localized(const xllama::TrainingJob& job,
                                               const xllama::DeviceTrainCallbacks& cb = {});

} // namespace xllama::bridge
