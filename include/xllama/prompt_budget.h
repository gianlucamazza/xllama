// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
// Exact context budgeting: the ONE place that decides how much history a prompt
// carries. Kept WinRT-free and tokenizer-agnostic so it is host-testable.
#pragma once

#include "xllama/chat_prompt.h"

#include <functional>
#include <string>
#include <vector>

namespace xllama {

// Counts tokens the way the model that will GENERATE counts them —
// Session::count_tokens bound to the resident session. Nothing here estimates
// from character counts: the trimmer used to, and the estimate silently paid for
// itself out of the reply, because the generation loop clamps n_predict to
// whatever the context has left (session.cpp, #173). A char/token constant
// cannot be made safe (measured on console: prose 4.6, dense C++ 2.5 real
// chars/token against one estimator), so the budget is enforced where the
// tokenizer is, once. The chars-per-token estimate survives only in
// routing_policy.h, for the EP decision, where being wrong costs a routing
// choice and not an answer.
using TokenCounter = std::function<int(const std::string&)>;

struct PromptFit {
    std::string prompt; // rendered, ready for Session::generate
    int n_tokens = 0;   // exact count of `prompt` (-1 when no counter was given)
    int dropped = 0;    // oldest turns dropped to make it fit
    // False when even the trailing user message does not fit: the caller must
    // NOT generate — Session reports it as "prompt too long" with the numbers.
    bool fits = false;
};

// Render (system, turns, user_text) through |fmt|, dropping the OLDEST turns
// until the prompt plus room for the reply fits |n_ctx|. The room reserved is
// max(n_predict, kReservedGenerationTokens) plus the one token the decode loop
// needs, matching the session's own arithmetic exactly.
//
// The trailing user_text is never dropped — a single oversized message is a
// user-visible error, not something to silently truncate. |count| must be valid;
// with a null counter the result is {fits=false, n_tokens=-1} rather than an
// unverified prompt.
// Cost: O(log n) renders and tokenizations in the number of turns, not O(dropped)
// — "fits" is monotone in how much history is dropped, so the smallest fitting
// prompt is bisected. This runs on the turn's critical path.
PromptFit fit_prompt(const ChatFormat& fmt, const std::string& system,
                     const std::vector<ChatTurn>& turns, const std::string& user_text, int n_ctx,
                     int n_predict, const TokenCounter& count);

} // namespace xllama
