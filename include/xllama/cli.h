#pragma once

#include "xllama/inference_params.h"

namespace xllama {

// Parse command-line arguments into InferenceParams.
// Returns true on success, false if parsing failed or --help was requested.
// On --help, prints usage to stderr and returns false.
bool parse_cli_args(int argc, char** argv, InferenceParams& out);

} // namespace xllama
