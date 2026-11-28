// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
#include "xllama/thinking_split.h"

#include <algorithm>
#include <cctype>

namespace xllama {

namespace {

struct Delimiters {
    std::string_view open, close;
};
Delimiters delimiters(ThinkingMarkup markup) {
    if (markup == ThinkingMarkup::GemmaChannel)
        return {"<|channel>thought\n", "<channel|>"};
    return {kThinkOpenTag, kThinkCloseTag};
}

std::string trimmed(std::string_view s) {
    size_t b = 0, e = s.size();
    while (b < e && std::isspace(static_cast<unsigned char>(s[b])))
        ++b;
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1])))
        --e;
    return std::string(s.substr(b, e - b));
}

void append_block(std::string& reasoning, std::string_view block) {
    const std::string t = trimmed(block);
    if (t.empty())
        return;
    if (!reasoning.empty())
        reasoning += "\n\n";
    reasoning += t;
}

// Length of the longest suffix of |text| that is a proper prefix of |tag|.
size_t partial_tag_suffix(std::string_view text, std::string_view tag) {
    const size_t max = std::min(text.size(), tag.size() - 1);
    for (size_t n = max; n > 0; --n)
        if (text.substr(text.size() - n) == tag.substr(0, n))
            return n;
    return 0;
}

} // namespace

std::string_view thinking_prefill_at_end(std::string_view prompt, ThinkingMarkup markup) {
    const auto open = delimiters(markup).open;
    if (markup == ThinkingMarkup::ThinkTags && !prompt.empty() && prompt.back() == '\n')
        prompt.remove_suffix(1);
    if (prompt.size() >= open.size() && prompt.substr(prompt.size() - open.size()) == open)
        return open;
    return {};
}

std::vector<TextSpan> thinking_spans(std::string_view text, ThinkingMarkup markup) {
    const auto tags = delimiters(markup);
    std::vector<TextSpan> spans;
    size_t pos = 0;
    // Starts inside the reasoning: a closer before any opener.
    const size_t first_close = text.find(tags.close);
    const size_t first_open = text.find(tags.open);
    if (markup == ThinkingMarkup::ThinkTags && first_close != std::string_view::npos &&
        (first_open == std::string_view::npos || first_open > first_close)) {
        pos = first_close + tags.close.size();
        spans.push_back({0, pos});
    }
    for (;;) {
        const size_t o = text.find(tags.open, pos);
        if (o == std::string_view::npos)
            break;
        const size_t c = text.find(tags.close, o + tags.open.size());
        if (c == std::string_view::npos) {
            spans.push_back({o, text.size()}); // cut off mid-thought
            break;
        }
        pos = c + tags.close.size();
        spans.push_back({o, pos});
    }
    return spans;
}

ThinkingSplit split_thinking(std::string_view text, ThinkingMarkup markup) {
    const auto tags = delimiters(markup);
    ThinkingSplit out;
    std::string content;
    size_t pos = 0;
    for (const TextSpan& span : thinking_spans(text, markup)) {
        content.append(text.substr(pos, span.begin - pos));
        std::string_view block = text.substr(span.begin, span.end - span.begin);
        if (block.substr(0, tags.open.size()) == tags.open)
            block.remove_prefix(tags.open.size());
        if (block.size() >= tags.close.size() &&
            block.substr(block.size() - tags.close.size()) == tags.close)
            block.remove_suffix(tags.close.size());
        append_block(out.reasoning, block);
        pos = span.end;
    }
    content.append(text.substr(pos));
    // Stray closers left in the content are tags, not boundaries.
    for (size_t c = content.find(tags.close); c != std::string::npos; c = content.find(tags.close))
        content.erase(c, tags.close.size());
    out.content = trimmed(content);
    return out;
}

void ReasoningTokenCounter::push(std::string_view piece) {
    m_text.append(piece);
    m_ends.push_back(m_text.size());
}

int ReasoningTokenCounter::count(bool think_blocks, std::string_view parsed_reasoning) const {
    std::vector<TextSpan> spans;
    if (think_blocks)
        spans = thinking_spans(m_text, m_markup);
    if (!parsed_reasoning.empty()) {
        const size_t at = m_text.find(parsed_reasoning);
        if (at != std::string::npos) {
            spans.push_back({at, at + parsed_reasoning.size()});
            std::sort(spans.begin(), spans.end(),
                      [](const TextSpan& a, const TextSpan& b) { return a.begin < b.begin; });
        }
    }
    int n = 0;
    size_t begin = m_prefill_bytes;
    size_t s = 0;
    for (const size_t end : m_ends) {
        // Spans sorted by begin; skip those that end at or before this token.
        while (s < spans.size() && spans[s].end <= begin)
            ++s;
        for (size_t k = s; k < spans.size() && spans[k].begin < end; ++k)
            if (spans[k].end > begin && end > begin) {
                ++n;
                break;
            }
        begin = end;
    }
    return n;
}

void ThinkingStreamSplitter::emit(std::string_view text, ThinkingSplit& out) {
    if (text.empty())
        return;
    if (m_in_reasoning) {
        out.reasoning.append(text);
        return;
    }
    if (!m_content_started) {
        size_t b = 0;
        while (b < text.size() && std::isspace(static_cast<unsigned char>(text[b])))
            ++b;
        text.remove_prefix(b);
        if (text.empty())
            return;
        m_content_started = true;
    }
    out.content.append(text);
}

ThinkingSplit ThinkingStreamSplitter::push(std::string_view piece) {
    const auto tags = delimiters(m_markup);
    ThinkingSplit out;
    m_held.append(piece);
    for (;;) {
        const std::string_view tag = m_in_reasoning ? tags.close : tags.open;
        const size_t at = m_held.find(tag);
        // Outside a block a stray closer is dropped, as split_thinking does.
        const size_t stray = m_in_reasoning ? std::string::npos : m_held.find(tags.close);
        if (stray != std::string::npos && (at == std::string::npos || stray < at)) {
            emit(std::string_view(m_held).substr(0, stray), out);
            m_held.erase(0, stray + tags.close.size());
            continue;
        }
        if (at == std::string::npos)
            break;
        emit(std::string_view(m_held).substr(0, at), out);
        m_held.erase(0, at + tag.size());
        m_in_reasoning = !m_in_reasoning;
    }
    // Hold a suffix that could still become a tag (the open or, outside a
    // block, a stray close tag).
    size_t hold = partial_tag_suffix(m_held, m_in_reasoning ? tags.close : tags.open);
    if (!m_in_reasoning)
        hold = std::max(hold, partial_tag_suffix(m_held, tags.close));
    emit(std::string_view(m_held).substr(0, m_held.size() - hold), out);
    m_held.erase(0, m_held.size() - hold);
    return out;
}

ThinkingSplit ThinkingStreamSplitter::finish() {
    ThinkingSplit out;
    emit(m_held, out);
    m_held.clear();
    return out;
}

} // namespace xllama
