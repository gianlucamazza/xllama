// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
// Internal: the ORT GenAI search parameters, applied in exactly one place — the
// twin of sampler_chain.h for the llama.cpp path. Needs the ORT C headers, so it
// stays out of the public include/ tree.
//
// #125 unified the llama.cpp sampler chain but left the ORT search-parameter
// config hand-duplicated across run_inference (CLI/bench) and OrtSession
// (GUI/API). They then diverged: OrtSession dropped the greedy guard, so a
// GUI/API user at temperature 0 on a routed DML model ran the full chain —
// repetition penalty reweighing tokens before the argmax, which flips the top
// token (observed: LFM2.5 answered "User\n\n<|end|>" at temp 0 through the
// endpoint). This is that guard, in one place both callers use.
#pragma once

#include "ort_genai_c.h"
#include "xllama/ort_raii.h"
#include "xllama/sampling.h"

namespace xllama {

// Apply temperature and the sampling stages to |params|. The caller sets
// max_length itself (it differs: derived on the stateless path, the context on
// the chat path). Greedy — which includes temperature 0 (SamplingConfig::
// is_greedy) — pins the choice with do_sample=false + top_k=1 and MUST skip the
// reweighting stages. Otherwise the three search numbers are set EXPLICITLY:
// leaving them unset does not mean "our defaults", it means whatever
// genai_config.json ships with, a third configuration nobody chose.
inline void apply_ort_sampling(OgaGeneratorParams* params, const SamplingConfig& sc) {
    oga_check(OgaGeneratorParamsSetSearchNumber(params, "temperature",
                                                static_cast<double>(sc.temperature)),
              "SetSearchNumber temperature");
    if (sc.is_greedy()) {
        oga_check(OgaGeneratorParamsSetSearchBool(params, "do_sample", false),
                  "SetSearchBool do_sample");
        oga_check(OgaGeneratorParamsSetSearchNumber(params, "top_k", 1.0), "SetSearchNumber top_k");
        return;
    }
    oga_check(OgaGeneratorParamsSetSearchNumber(params, "top_p", static_cast<double>(sc.top_p)),
              "SetSearchNumber top_p");
    oga_check(OgaGeneratorParamsSetSearchNumber(params, "top_k", static_cast<double>(sc.top_k)),
              "SetSearchNumber top_k");
    oga_check(OgaGeneratorParamsSetSearchNumber(params, "repetition_penalty",
                                                static_cast<double>(sc.repetition_penalty)),
              "SetSearchNumber repetition_penalty");
}

} // namespace xllama
