// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#pragma once

#include "xllama/inference_params.h"

namespace xllama {

// Parse command-line arguments into InferenceParams.
// Returns true on success and false if parsing fails.
// On --help, prints usage to stdout and exits successfully.
bool parse_cli_args(int argc, char** argv, InferenceParams& out);

} // namespace xllama
