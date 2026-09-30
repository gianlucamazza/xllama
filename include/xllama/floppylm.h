// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
#pragma once
#include "xllama/device_train.h"
#include <string>
namespace xllama {
// Independent scalar training lane; never invokes the cached inference graph.
TrainingResult run_floppylm_job(const TrainingJob& job, const DeviceTrainCallbacks& cb = {});
// Numerical probe with explicit initial weights and batches. Used by host/device parity gates.
std::string floppylm_probe(const std::string& request);
bool validate_floppylm_job_json(const std::string& text, TrainingJob& out, std::string* err);
} // namespace xllama
