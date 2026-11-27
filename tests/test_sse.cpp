// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include <doctest/doctest.h>

#include <atomic>
#include <chrono>
#include <thread>

#include "xllama/sse.h"

#include <string>

using xllama::chunk_final;
using xllama::chunk_frame;
using xllama::sse_done;
using xllama::sse_event;
using xllama::sse_payload_is_safe;
using xllama::sse_response_head;
using xllama::sse_utf8_step;

namespace {
// Reads one chunked frame back: the size line in hex, then the payload.
bool read_chunk(const std::string& data, std::string& payload, size_t& consumed) {
    const size_t crlf = data.find("\r\n");
    if (crlf == std::string::npos)
        return false;
    size_t size = 0;
    for (const char c : data.substr(0, crlf)) {
        int digit = -1;
        if (c >= '0' && c <= '9')
            digit = c - '0';
        else if (c >= 'a' && c <= 'f')
            digit = c - 'a' + 10;
        if (digit < 0)
            return false;
        size = size * 16 + static_cast<size_t>(digit);
    }
    if (data.size() < crlf + 2 + size + 2)
        return false;
    payload = data.substr(crlf + 2, size);
    consumed = crlf + 2 + size + 2;
    return data.compare(consumed - 2, 2, "\r\n") == 0;
}
} // namespace

TEST_CASE("sse: an event is one data line and a blank line") {
    CHECK(sse_event("{\"a\":1}") == "data: {\"a\":1}\n\n");
    CHECK(sse_event("") == "data: \n\n");
    CHECK(sse_done() == "data: [DONE]\n\n");
}

TEST_CASE("sse: a payload that would break the framing is refused") {
    CHECK(sse_payload_is_safe("{\"content\":\"hi\"}"));
    CHECK(sse_payload_is_safe(""));
    CHECK_FALSE(sse_payload_is_safe("two\nlines"));
    CHECK_FALSE(sse_payload_is_safe("two\rlines"));
    CHECK_FALSE(sse_payload_is_safe(std::string("nul\0here", 8)));
}

TEST_CASE("sse: chunked framing matches HTTP/1.1 hex sizes") {
    CHECK(chunk_frame("hello") == "5\r\nhello\r\n");
    CHECK(chunk_frame("") == "0\r\n\r\n");
    // 26 bytes -> 1a in hex: the size line is lower-case hex, no 0x, no padding.
    const std::string twenty_six(26, 'x');
    CHECK(chunk_frame(twenty_six) == "1a\r\n" + twenty_six + "\r\n");
    CHECK(chunk_final() == "0\r\n\r\n");
}

TEST_CASE("sse: an event inside a chunk reads back byte for byte") {
    const std::string json = "{\"content\":\"hola\"}";
    const std::string framed = chunk_frame(sse_event(json));
    std::string payload;
    size_t consumed = 0;
    REQUIRE(read_chunk(framed, payload, consumed));
    CHECK(consumed == framed.size());
    CHECK(payload == "data: " + json + "\n\n");
}

TEST_CASE("sse: failure after a delta sends only error, DONE and the final HTTP chunk") {
    const std::string delta =
        R"({"choices":[{"delta":{"content":"partial"},"finish_reason":null}]})";
    const std::string error = "generation failed: \"backend\"\r\nrc=-3";
    std::string wire = chunk_frame(sse_event(delta)) +
                       xllama::chat_terminal_frames(xllama::ChatWire::OpenAI, &error);
    const std::string expected_error =
        R"({"error":{"message":"generation failed: \"backend\"\r\nrc=-3","type":"xllama_error"}})";
    for (const std::string& expected :
         {sse_event(delta), sse_event(expected_error), sse_done(), std::string{}}) {
        std::string payload;
        size_t consumed = 0;
        REQUIRE(read_chunk(wire, payload, consumed));
        CHECK(payload == expected);
        wire.erase(0, consumed);
    }
    CHECK(wire.empty());
    CHECK(xllama::chat_terminal_frames(xllama::ChatWire::OpenAI) ==
          chunk_frame(sse_done()) + chunk_final());
}

TEST_CASE("Ollama chat and generate end a failed NDJSON reply with one error line") {
    const std::string error = "generation failed: \"backend\"\r\nrc=-3";
    for (const auto wire : {xllama::ChatWire::OllamaChat, xllama::ChatWire::OllamaGenerate}) {
        const std::string delta =
            wire == xllama::ChatWire::OllamaChat
                ? R"({"message":{"role":"assistant","content":"partial"},"done":false})"
                : R"({"response":"partial","done":false})";
        std::string bytes = chunk_frame(delta + "\n") + xllama::chat_terminal_frames(wire, &error);
        const std::string error_line = R"({"error":"generation failed: \"backend\"\r\nrc=-3"})";
        for (const auto& expected : {delta + "\n", error_line + "\n", std::string{}}) {
            std::string payload;
            size_t consumed = 0;
            REQUIRE(read_chunk(bytes, payload, consumed));
            CHECK(payload == expected);
            bytes.erase(0, consumed);
        }
        CHECK(bytes.empty());
        CHECK(xllama::chat_terminal_frames(wire) == chunk_final());
    }
}

TEST_CASE("sse: UTF-8 steps keep code points whole") {
    // ASCII and complete multibyte sequences are taken as they are.
    CHECK(sse_utf8_step("", false).take == 0);
    CHECK_FALSE(sse_utf8_step("", false).wait);
    CHECK(sse_utf8_step("hola", false).take == 4);
    CHECK(sse_utf8_step("caf\xC3\xA9", false).take == 5);       // café
    CHECK(sse_utf8_step("a\xF0\x9F\x98\x80", false).take == 5); // emoji U+1F600
}

TEST_CASE("sse: a split sequence waits for the next piece") {
    // A piece that ends inside a sequence first yields its complete prefix; the
    // truncated tail is what waits for the next piece.
    auto step = sse_utf8_step("caf\xC3", false);
    CHECK_FALSE(step.wait);
    CHECK_FALSE(step.replace);
    CHECK(step.take == 3);

    step = sse_utf8_step("\xC3", false);
    CHECK(step.wait);
    CHECK(step.take == 0);
    CHECK_FALSE(step.replace);

    step = sse_utf8_step("a\xF0\x9F\x98", false);
    CHECK(step.take == 1);
    step = sse_utf8_step("\xF0\x9F\x98", false);
    CHECK(step.wait);

    // At finish the truncated tail is replaced instead of waiting forever.
    step = sse_utf8_step("\xC3", true);
    CHECK_FALSE(step.wait);
    CHECK(step.replace);
    CHECK(step.take == 1);
}

TEST_CASE("sse: malformed bytes advance instead of stalling") {
    // An invalid lead is replaced, one byte at a time.
    auto step = sse_utf8_step("\x80hola", false);
    CHECK(step.replace);
    CHECK(step.take == 1);

    // A valid ASCII prefix is emitted first; the bad lead follows on the rescan.
    step = sse_utf8_step("a\xC3z", false);
    CHECK_FALSE(step.replace);
    CHECK_FALSE(step.wait);
    CHECK(step.take == 1);

    step = sse_utf8_step("\xC3z", false);
    CHECK(step.replace);
    CHECK(step.take == 1);
}

TEST_CASE("sse: draining recovers the text after a malformed byte") {
    // Mirrors SseStream::drain_pending: replace, drop one byte, rescan.
    const auto drain = [](std::string pending, bool finishing) {
        std::string out;
        while (!pending.empty()) {
            const auto step = sse_utf8_step(pending, finishing);
            if (step.wait)
                break;
            if (step.replace) {
                pending.erase(0, 1);
                out += "\xEF\xBF\xBD";
                continue;
            }
            out += pending.substr(0, step.take);
            pending.erase(0, step.take);
        }
        return out;
    };
    CHECK(drain("a\x80"
                "b",
                false) == "a\xEF\xBF\xBD"
                          "b");
    CHECK(drain("caf\xC3", true) == "caf\xEF\xBF\xBD");
    CHECK(drain("caf\xC3\xA9", false) == "caf\xC3\xA9");
}

TEST_CASE("sse: the response head is an event stream without Content-Length") {
    const std::string head = sse_response_head();
    CHECK(head.find("Content-Type: text/event-stream") != std::string::npos);
    CHECK(head.find("Transfer-Encoding: chunked") != std::string::npos);
    CHECK(head.find("Content-Length") == std::string::npos);
    CHECK(head.find("Access-Control-Allow-Origin: *") != std::string::npos);
    CHECK(head.find("Cache-Control: no-cache") != std::string::npos);
    CHECK(head.size() >= 4);
    CHECK(head.compare(head.size() - 4, 4, "\r\n\r\n") == 0);
}

TEST_CASE("sse: a comment is skipped framing, not an event") {
    CHECK(xllama::sse_comment("keepalive") == ": keepalive\n\n");
}

TEST_CASE("sse: the keepalive writes while the stream is silent and stops on a dead client") {
    using namespace std::chrono_literals;
    std::atomic<int> writes{0};
    {
        xllama::SseKeepalive k(20ms, [&writes] {
            ++writes;
            return true;
        });
        std::this_thread::sleep_for(150ms);
        k.stop();
    }
    CHECK(writes.load() >= 3);

    // Touched more often than the interval: nothing to write.
    std::atomic<int> quiet{0};
    {
        xllama::SseKeepalive k(80ms, [&quiet] {
            ++quiet;
            return true;
        });
        for (int i = 0; i < 10; ++i) {
            std::this_thread::sleep_for(20ms);
            k.touch();
        }
    }
    CHECK(quiet.load() == 0);

    // A failed write ends the keepalive: no retry on a gone client.
    std::atomic<int> failed{0};
    {
        xllama::SseKeepalive k(10ms, [&failed] {
            ++failed;
            return false;
        });
        std::this_thread::sleep_for(100ms);
    }
    CHECK(failed.load() == 1);
}
