// Copyright (c) 2024 Gianluca Mazza
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
//
// #175 (decided 2026-07-26): sampler STATE follows the KV lifecycle on both
// backends — it lives as long as the conversation and resets with reset_kv or
// a sampling change (llama: chain kept across reuse turns in LlamaSession;
// ORT: the persistent generator always worked this way). The window WIDTH
// still differs by design: llama penalizes the last 64 generated tokens, ORT
// GenAI penalizes the whole sequence and its C API exposes no window — the
// short window is kept deliberately (whole-sequence penalties are the known
// cause of long-chat degradation), not an alignment gap to fix.
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

// Would these two configs assemble the same sampler chain? The #175 chain-reuse
// guard: a persistent chain may only be reused while every stage parameter is
// unchanged (both backends rebuild their sampler state on any mismatch — see
// LlamaSession::generate and OrtSession::sampling_matches).
inline bool same_chain(const SamplingConfig& a, const SamplingConfig& b) {
    if (a.is_greedy() != b.is_greedy())
        return false;
    if (a.is_greedy())
        return true; // greedy chains have no other parameters
    return a.temperature == b.temperature && a.top_p == b.top_p && a.top_k == b.top_k &&
           a.repetition_penalty == b.repetition_penalty && a.seed == b.seed;
}

} // namespace xllama
