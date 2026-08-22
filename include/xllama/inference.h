// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#pragma once

#include "xllama/inference_params.h"

namespace xllama {

/// Run inference synchronously.
///
/// Loads the model from `params.model_path`, runs a prefill+decode pass,
/// and returns an `InferenceResult` with timing metrics and generated text.
///
/// @param params  Inference configuration (model path, prompt, sampling, etc.)
/// @return        InferenceResult with `success`, `output_text`, timing fields,
///                and `error_msg` on failure.
///
/// Path semantics:
///   - Linux:  `model_path` is an absolute filesystem path.
///   - UWP:    `model_path` is relative to
///             `ApplicationData::LocalFolder\models\`.
InferenceResult run_inference(const InferenceParams& params);

/// Write a benchmark CSV row to the resolved local path.
///
/// Appends a single CSV line containing the run index, host label, and all
/// timing metrics from @p res. Used by the bench pipeline to collect data.
///
/// @param params  InferenceParams (provides model_path for CSV path resolution).
/// @param res     InferenceResult with timing metrics.
/// @param host_label  Optional host identifier written into the CSV row.
void write_bench_csv(const InferenceParams& params, const InferenceResult& res,
                     const char* host_label = nullptr);

} // namespace xllama
