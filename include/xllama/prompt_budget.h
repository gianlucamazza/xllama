// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
// Exact context budgeting. Plain chat uses fit_prompt; a tool proposal uses
// fit_tool_transcript. Both ask the resident tokenizer. Neither estimates from
// characters. Kept WinRT-free so it is host-testable.
#pragma once

#include "xllama/chat_prompt.h"
#include "xllama/tool_transcript.h"

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

// The one admission test both trimmers use: |prompt_tokens| plus the reply's
// room — max(n_predict, kReservedGenerationTokens) plus the decode loop's one
// token — fits |n_ctx|. Computed in 64 bits, so an INT_MAX n_predict cannot
// overflow it (F01); a negative count (a failed tokenizer) never fits.
bool budget_fits(int prompt_tokens, int n_predict, int n_ctx);

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

// Renders the tool transcript the way the GGUF template will. Tests pass a
// stand-in; the LAN handler passes the template bridge. The schema is an
// argument, not a message, so a trimmer cannot drop it by dropping history.
using ToolPromptRenderer = std::function<std::string(const std::vector<ChatMessage>& messages,
                                                     const std::vector<ToolSpec>& tools)>;

struct ToolTranscriptFit {
    std::string prompt;
    std::vector<ChatMessage> messages; // what `prompt` was rendered from
    int n_tokens = 0;                  // exact count of `prompt` (-1 with no counter)
    int dropped = 0;                   // messages removed, not groups
    bool fits = false;
};

// Drop the oldest non-system messages until the prompt plus room for the reply
// fits `n_ctx`. The room is max(n_predict, kReservedGenerationTokens) plus the
// decode-loop token, the same arithmetic as fit_prompt. The tool schema, every
// system message, and the final user or tool message stay. An assistant that
// carries tool_calls and the tool messages that follow it drop together. A
// tool whose assistant was dropped is not left behind. When schema + system +
// the protected tail still do not fit, `fits` is false and `n_tokens` is that
// minimal prompt. A null counter returns fits=false and n_tokens=-1.
ToolTranscriptFit fit_tool_transcript(const std::vector<ChatMessage>& messages,
                                      const std::vector<ToolSpec>& tools, int n_ctx, int n_predict,
                                      const ToolPromptRenderer& render, const TokenCounter& count);

} // namespace xllama
