// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
// Server-Sent Events framing for the streamed chat completion
// (docs/api-endpoint.md). WinRT-free so the host suite can test it; the UWP
// transport only owns the socket and moves the bytes.
#pragma once

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>

namespace xllama {

// One SSE event: "data: <payload>\n\n". The OpenAI streaming convention puts one
// JSON object per event and closes the stream with sse_done().
std::string sse_event(std::string_view payload);

// The terminal event of a streamed completion.
std::string sse_done();

// An SSE comment line, ": <text>\n\n". Clients skip it (the SSE spec, and the
// OpenAI SDKs' parsers), so it carries bytes without carrying an event.
std::string sse_comment(std::string_view text);

// How long a stream may stay silent before SseKeepalive writes a comment.
inline constexpr std::chrono::milliseconds kSseKeepaliveInterval{5000};

// Writes a comment when a stream has been silent for |interval| (#581). A long
// prefill sends nothing, and a dead socket is noticed only when a write fails,
// so a client that left kept the session prefilling. |write| sends the comment
// and returns false when the client is gone; the keepalive then stops for
// good. touch() records any other write. The thread runs until stop() or the
// destructor, which join it; |write| must not call either.
class SseKeepalive {
  public:
    SseKeepalive(std::chrono::milliseconds interval, std::function<bool()> write);
    ~SseKeepalive();
    SseKeepalive(const SseKeepalive&) = delete;
    SseKeepalive& operator=(const SseKeepalive&) = delete;

    void touch();
    void stop();

  private:
    void run();

    const std::chrono::milliseconds m_interval;
    std::function<bool()> m_write;
    std::mutex m_mu;
    std::condition_variable m_cv;
    std::chrono::steady_clock::time_point m_last;
    bool m_stopped = false;
    std::thread m_thread; // last: starts after the state above exists
};

// False when the payload holds a byte that would break the framing: an event is
// one line, so CR, LF and NUL are refused instead of emitted. JSON encoding
// escapes them, so this catches a builder that forgot to escape.
bool sse_payload_is_safe(std::string_view payload);

// One HTTP/1.1 chunked-transfer frame: "<hex size>\r\n<payload>\r\n". A
// streamed response carries no Content-Length, so every event travels in a chunk.
std::string chunk_frame(std::string_view payload);

// The last chunk of a chunked response.
std::string chunk_final();

// One step over the bytes buffered between token callbacks. A token piece from
// llama_token_to_piece() can end midway through a multibyte character, so the
// streamed writer holds the trailing bytes until the next piece completes them.
//
// `take` bytes end on a code-point boundary and can be emitted as text. When
// `replace` is true the first byte cannot start a valid sequence (or the stream
// ended there): the writer emits U+FFFD, drops that one byte and scans again, so
// malformed input advances instead of stalling the rest of the reply. `wait`
// means the buffer ends with a valid but truncated sequence and `finishing` is
// false.
struct SseUtf8Step {
    std::size_t take = 0;
    bool replace = false;
    bool wait = false;
};

SseUtf8Step sse_utf8_step(std::string_view pending, bool finishing);

// Response head for a streamed completion.
std::string sse_response_head();

} // namespace xllama
