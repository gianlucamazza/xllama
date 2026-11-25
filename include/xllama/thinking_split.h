// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// The one home of <think> handling for model output: a finished reply split
// into reasoning and content (split_thinking), and the same split applied to
// a stream of pieces (ThinkingStreamSplitter). The GUI shows the content and
// keeps the reasoning out of history; the LAN API returns the reasoning as
// message.reasoning_content / delta.reasoning_content (#564).
#pragma once

#include <string>
#include <string_view>

namespace xllama {

inline constexpr std::string_view kThinkOpenTag = "<think>";
inline constexpr std::string_view kThinkCloseTag = "</think>";

struct ThinkingSplit {
    std::string reasoning;
    std::string content;
};

// Text between <think> and </think> is reasoning, the rest content.
// - Several blocks are joined, separated by a blank line.
// - A reply that starts inside the reasoning ("…</think>answer", when a
//   template or a stop sequence ate the opener) counts everything before the
//   first closer as reasoning.
// - A reply cut off mid-thought ("<think>…" with no closer) is all reasoning
//   from the opener on.
// - A stray closer after the balanced pass is dropped.
// Content is trimmed at both ends, reasoning too.
ThinkingSplit split_thinking(std::string_view text);

// split_thinking on a stream. push() returns what can be emitted now; a piece
// that might be the start of a tag is held until the next piece decides it.
// A reply that starts inside the reasoning without an opener is not detected
// here (the text was already emitted as content); the catalogue templates do
// not open the block in the generation prompt, the model writes <think>.
class ThinkingStreamSplitter {
  public:
    ThinkingSplit push(std::string_view piece);
    // Flush what was held: a partial tag at the end is ordinary text.
    ThinkingSplit finish();

  private:
    void emit(std::string_view text, ThinkingSplit& out);

    std::string m_held;
    bool m_in_reasoning = false;
    bool m_content_started = false; // leading whitespace of the reply is dropped
};

} // namespace xllama
