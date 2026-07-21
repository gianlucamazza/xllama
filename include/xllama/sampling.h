// Copyright (c) 2024 Venere Labs
// SPDX-License-Identifier: MIT
// Sampling defaults shared by every surface, and the one place the llama.cpp
// sampler chain is assembled.
//
// #125: InferenceParams (CLI, bench) and GenerateParams (GUI, LAN API) each
// carried their own sampling fields and each built their own chain. They had
// drifted into two different samplers: the CLI ran temp -> dist only, with no
// top_p, top_k or repetition penalty at all, so a generation seen in the GUI
// could not be reproduced from the command line. Nothing detected it because
// every marker gate runs --greedy, which bypasses sampling entirely.
//
// The defaults below are the single source; both structs initialise from them
// and tests/test_sampling.cpp pins that they agree. The chain builder is shared
// for the same reason: the stage ORDER is part of the behaviour, not a detail.
#pragma once

#include <cstdint>

namespace xllama {
namespace sampling_defaults {

inline constexpr float kTemperature = 0.8f;
inline constexpr float kTopP = 0.9f;
inline constexpr int kTopK = 40;
inline constexpr float kRepetitionPenalty = 1.1f;
// Window the repetition penalty looks back over. Not exposed as a flag —
// llama.cpp's own default is 64 and no surface has ever varied it.
inline constexpr int kPenaltyLastN = 64;
inline constexpr uint32_t kSeed = 0xFFFFFFFF; // LLAMA_DEFAULT_SEED

} // namespace sampling_defaults

// The sampling knobs, in one shape. Both InferenceParams and GenerateParams
// expose these values; this struct is what crosses the boundary when one surface
// hands its configuration to another (CLI -> run_inference, API -> Session).
struct SamplingConfig {
    float temperature = sampling_defaults::kTemperature;
    float top_p = sampling_defaults::kTopP;
    int top_k = sampling_defaults::kTopK;
    float repetition_penalty = sampling_defaults::kRepetitionPenalty;
    uint32_t seed = sampling_defaults::kSeed;
    // Deterministic decode: argmax instead of sampling. Prerequisite for
    // cross-backend logit parity (llama.cpp and ORT must agree token-for-token).
    bool greedy = false;

    // Argmax is also what temperature 0 means, and the full chain must NOT run
    // in that case: the repetition penalty reweighs prompt tokens BEFORE the
    // temp stage's argmax and can flip the top token (observed on device —
    // LFM2.5 answered "User\n\n<|end|>" at temperature 0 through the endpoint
    // while pure argmax on the same prompt answered "Hello!").
    bool is_greedy() const {
        return greedy || temperature <= 0.0f;
    }
};

} // namespace xllama
