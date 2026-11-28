// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// Model output uses <think> tags or the Gemma thought channel: a finished reply split
// into reasoning and content (split_thinking), and the same split applied to
// a stream of pieces (ThinkingStreamSplitter). The GUI shows the content and
// keeps the reasoning out of history; the LAN API returns the reasoning as
// message.reasoning_content / delta.reasoning_content (#564).
#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace xllama {

inline constexpr std::string_view kThinkOpenTag = "<think>";
inline constexpr std::string_view kThinkCloseTag = "</think>";

enum class ThinkingMarkup { ThinkTags, GemmaChannel };

// Returns the opening marker already emitted by a template generation prompt, if any.
std::string_view thinking_prefill_at_end(std::string_view prompt, ThinkingMarkup markup);

struct ThinkingSplit {
    std::string reasoning;
    std::string content;
};

// Text between the selected reasoning markers is reasoning, the rest content.
// - Several blocks are joined, separated by a blank line.
// - A reply that starts inside the reasoning ("…</think>answer", when a
//   <think> template or a stop sequence ate the opener) counts everything before the
//   first closer as reasoning.
// - A reply cut off mid-thought ("<think>…" with no closer) is all reasoning
//   from the opener on.
// - A stray closer after the balanced pass is dropped.
// Content is trimmed at both ends, reasoning too.
ThinkingSplit split_thinking(std::string_view text);
ThinkingSplit split_thinking(std::string_view text, ThinkingMarkup markup);

// A byte range [begin, end) of a text.
struct TextSpan {
    std::size_t begin = 0;
    std::size_t end = 0;
};

// Where split_thinking finds the reasoning: one span per block, tags included,
// in order and disjoint. split_thinking is built on it.
std::vector<TextSpan> thinking_spans(std::string_view text);
std::vector<TextSpan> thinking_spans(std::string_view text, ThinkingMarkup markup);

// Counts the generated tokens spent on reasoning, for
// usage.completion_tokens_details.reasoning_tokens (#585). push() takes each
// token's piece as Session::generate reports it through on_token. A token
// counts when any of its bytes lies in a reasoning block, its tags included.
class ReasoningTokenCounter {
  public:
    // |prefill|: reply text the prompt already emitted (ChatFormat::
    // reasoning_prefill), split with the reply but never counted as a token.
    explicit ReasoningTokenCounter(std::string_view prefill = {},
                                   ThinkingMarkup markup = ThinkingMarkup::ThinkTags)
        : m_text(prefill), m_prefill_bytes(prefill.size()), m_markup(markup) {}
    void push(std::string_view piece);
    // |think_blocks|: the reply is split on <think> blocks (a thinking model).
    // |parsed_reasoning|: what a template's parser extracted with its own
    // markup; counted where it occurs verbatim in the reply.
    int count(bool think_blocks, std::string_view parsed_reasoning = {}) const;

  private:
    std::string m_text;
    std::size_t m_prefill_bytes;
    ThinkingMarkup m_markup;
    std::vector<std::size_t> m_ends; // end offset of each token's piece in m_text
};

// split_thinking on a stream. push() returns what can be emitted now; a piece
// that might be the start of a tag is held until the next piece decides it.
// A reply that starts inside the reasoning without an opener is not detected
// here (the text was already emitted as content): a format whose prompt opens
// the block passes it as |prefill| (ChatFormat::reasoning_prefill).
class ThinkingStreamSplitter {
  public:
    explicit ThinkingStreamSplitter(std::string_view prefill = {},
                                    ThinkingMarkup markup = ThinkingMarkup::ThinkTags)
        : m_markup(markup) {
        push(prefill);
    }
    ThinkingSplit push(std::string_view piece);
    // Flush ordinary trailing text. An unfinished Gemma thought header starting
    // with the complete reserved control token is framing.
    ThinkingSplit finish();

  private:
    void emit(std::string_view text, ThinkingSplit& out);

    ThinkingMarkup m_markup;
    std::string m_held;
    bool m_in_reasoning = false;
    bool m_content_started = false; // leading whitespace of the reply is dropped
    // Join nonempty Gemma blocks the same way as split_thinking. Whitespace
    // after the last reasoning byte may still belong inside this block.
    enum class BlockPosition { First, Inside, Between };
    BlockPosition m_gemma_position = BlockPosition::First;
    std::string m_reasoning_whitespace;
};

} // namespace xllama
