// Copyright (c) 2024 Venere Labs
// SPDX-License-Identifier: MIT
// Training pillar API — job validation, stage vocabulary, device gates.
// Execution of PEFT lives in training/host (Python); this header is the
// platform contract so inference and training stay cleanly separated.
#pragma once

#include "xllama/training_params.h"

#include <string>

namespace xllama {

// Human-readable stage name (stable for logs / result.json).
const char* training_stage_name(TrainStage stage);
const char* training_method_name(TrainMethod method);
const char* training_device_name(TrainDevice device);

// Host = true; Device = false until an on-console backend is implemented.
bool training_device_supported(TrainDevice device);

// Pure validation: required fields, ranges, device/method exploration gates.
// On failure sets *err and returns false.
bool validate_training_job(const TrainingJob& job, std::string* err = nullptr);

// Minimal JSON object parser for training job manifests (no external JSON lib).
// Supports the keys used by training/jobs/*.json; unknown keys are ignored.
bool parse_training_job_json(const std::string& json, TrainingJob& out, std::string* err = nullptr);

// Load a job file from disk then parse + validate.
bool load_training_job_file(const std::string& path, TrainingJob& out, std::string* err = nullptr);

// One-line summary for CLI / logs.
std::string format_training_job_summary(const TrainingJob& job);

} // namespace xllama
