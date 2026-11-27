// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include "xllama/api_ollama.h"
#include "xllama/chat_prompt.h"
#include "xllama/reasoning_control.h"
#include "xllama/tool_request.h"

#include <doctest/doctest.h>
#include <nlohmann/json.hpp>

using namespace xllama;
using nlohmann::json;

TEST_CASE("Ollama: generate becomes a chat body with options mapped to the shared fields") {
    const auto request = parse_ollama_request(
        R"({"model":"chat","prompt":"hello","system":"be brief","options":{"num_predict":32,)"
        R"("repeat_penalty":1.2,"stop":["end"],"temperature":0,"seed":7}})",
        OllamaRoute::Generate);
    REQUIRE_MESSAGE(request.error.empty(), request.error);
    CHECK(request.stream);
    CHECK(request.action == OllamaAction::Generate);
    const json body = json::parse(request.body);
    CHECK(body["model"] == "chat");
    CHECK(body["max_completion_tokens"] == 32);
    CHECK(body["repeat_penalty"] == 1.2);
    CHECK(body["stop"] == json::array({"end"}));
    CHECK(body["temperature"] == 0);
    CHECK(body["seed"] == 7);
    REQUIRE(body["messages"].size() == 2);
    CHECK(body["messages"][0]["role"] == "system");
    CHECK(body["messages"][1]["content"] == "hello");
    // The shared plain-chat parser reads it as it reads an OpenAI request.
    const PlainChatRequest plain = parse_plain_chat(request.body);
    CHECK(plain.system == "be brief");
    CHECK(plain.final_user == "hello");

    // -1 is Ollama's unbounded default: xllama keeps its own bounded one.
    const auto unbounded = parse_ollama_request(
        R"({"model":"chat","prompt":"x","options":{"num_predict":-1}})", OllamaRoute::Generate);
    REQUIRE(unbounded.error.empty());
    CHECK_FALSE(json::parse(unbounded.body).contains("max_completion_tokens"));
    CHECK_FALSE(parse_ollama_request(R"({"model":"chat","prompt":"x","stream":false})",
                                     OllamaRoute::Generate)
                    .stream);
}

TEST_CASE("Ollama: an empty request loads the model and answers with JSON") {
    for (const auto& [body, route] :
         {std::pair{R"({"model":"chat","messages":[]})", OllamaRoute::Chat},
          std::pair{R"({"model":"chat"})", OllamaRoute::Chat},
          std::pair{R"({"model":"chat","prompt":""})", OllamaRoute::Generate},
          std::pair{R"({"model":"chat","stream":true})", OllamaRoute::Generate}}) {
        CAPTURE(body);
        const auto load = parse_ollama_request(body, route);
        REQUIRE(load.error.empty());
        CHECK(load.action == OllamaAction::Load);
        CHECK_FALSE(load.stream);
    }
}

TEST_CASE("Ollama: keep_alive takes seconds or a duration, and 0 on an empty request unloads") {
    using namespace std::chrono_literals;
    const auto keep = [](const char* value) {
        return parse_ollama_request(std::string(R"({"model":"chat","prompt":"x","keep_alive":)") +
                                        value + "}",
                                    OllamaRoute::Generate);
    };
    CHECK_FALSE(parse_ollama_request(R"({"model":"chat","prompt":"x"})", OllamaRoute::Generate)
                    .keep_alive.has_value()); // omitted: no expiry
    CHECK_FALSE(keep("null").keep_alive.has_value());
    CHECK(*keep("30").keep_alive == 30000ms);
    CHECK(*keep("0.75").keep_alive == 750ms);
    CHECK(*keep(R"("5m")").keep_alive == 300000ms);
    CHECK(*keep(R"("1m30.5s")").keep_alive == 90500ms);
    CHECK(*keep(R"("750ms")").keep_alive == 750ms);
    CHECK(*keep(R"("0")").keep_alive == 0ms);
    CHECK(*keep("-1").keep_alive == -1ms);
    CHECK(*keep(R"("-5m")").keep_alive == -1ms);
    for (const char* bad : {R"("tomorrow")", R"("5")", R"("")", R"("1e9h")", "true", "[]"}) {
        CAPTURE(bad);
        CHECK(keep(bad).error.find("keep_alive") == 0);
    }
    // A generation with keep_alive 0 still generates; only an empty one unloads.
    CHECK(keep("0").action == OllamaAction::Generate);
    const auto unload =
        parse_ollama_request(R"({"model":"chat","messages":[],"keep_alive":0})", OllamaRoute::Chat);
    REQUIRE(unload.error.empty());
    CHECK(unload.action == OllamaAction::Unload);
    CHECK_FALSE(unload.stream);
    CHECK(parse_ollama_request(R"({"model":"chat","keep_alive":-1})", OllamaRoute::Chat).action ==
          OllamaAction::Load);
}

TEST_CASE("Ollama: options.num_ctx passes through for the handler's catalogue check") {
    const auto request = parse_ollama_request(
        R"({"model":"chat","prompt":"x","options":{"num_ctx":256}})", OllamaRoute::Generate);
    REQUIRE_MESSAGE(request.error.empty(), request.error);
    CHECK(json::parse(request.body)["options"]["num_ctx"] == 256);
    CHECK(parse_ollama_request(R"({"model":"chat","options":{"num_ctx":"big"}})", OllamaRoute::Chat)
              .error == "options.num_ctx must be an integer");
}

TEST_CASE("Ollama: requests xllama cannot honour are refused before admission") {
    const auto error = [](const char* body, OllamaRoute route = OllamaRoute::Chat) {
        return parse_ollama_request(body, route).error;
    };
    CHECK(error(R"({"model":"../chat"})") == "model must be an id from /api/tags");
    CHECK(error(R"({"model":"a/b"})") == "model must be an id from /api/tags");
    CHECK(error(R"({"model":"chat","stream":"yes"})") == "stream must be boolean");
    CHECK(error(R"({"model":"chat","options":{"num_gpu":99}})") == "unsupported options.num_gpu");
    CHECK(error(R"({"model":"chat","options":{"temperature":3}})").find("temperature") !=
          std::string::npos);
    CHECK(error(R"({"model":"chat","options":{"temperature":"hot"}})") ==
          "options.temperature must be a number");
    CHECK(error(R"({"model":"chat","options":{"stop":[1]}})") ==
          "options.stop must be a string or an array of strings");
    CHECK(error(R"({"model":"chat","options":[]})") == "options must be an object");
    CHECK(error(R"({"model":"chat","format":"json"})") == "format is unsupported");
    CHECK(error(R"({"model":"chat","prompt":"x","raw":true})", OllamaRoute::Generate) ==
          "raw is unsupported");
    CHECK(error(R"({"model":"chat","prompt":"x","images":["AA=="]})", OllamaRoute::Generate) ==
          "images is unsupported");
    CHECK(
        error(R"({"model":"chat","messages":[{"role":"user","content":"x","images":["AA=="]}]})") ==
        "messages[].images is unsupported");
    CHECK(error(R"({"model":"chat","messages":[{"role":"user","content":[]}]})") ==
          "messages[].content must be a string");
    CHECK(error(R"({"model":"chat","messages":[{"role":"user","content":"x"},)"
                R"({"role":"assistant","content":"y"}]})") == "last message must be user or tool");
    // Defaults are not refusals.
    CHECK(error(R"({"model":"chat","format":"","raw":false,"images":[],"logprobs":false,)"
                R"("top_logprobs":0,"messages":[{"role":"user","content":"x"}]})")
              .empty());
}

TEST_CASE("Ollama: think maps onto the shared reasoning decision (#648)") {
    const auto think = [](const char* model, const char* value) {
        return parse_ollama_request(std::string(R"({"model":")") + model +
                                        R"(","prompt":"x","think":)" + value + "}",
                                    OllamaRoute::Generate);
    };
    const auto decision = [](const OllamaRequest& request, const char* model) {
        REQUIRE_MESSAGE(request.error.empty(), request.error);
        return decide_reasoning(model, read_reasoning_fields(request.body));
    };
    // A boolean is chat_template_kwargs.enable_thinking, a level is reasoning_effort.
    const OllamaRequest off = think("qwen3-4b", "false");
    CHECK(json::parse(off.body)["chat_template_kwargs"] == json{{"enable_thinking", false}});
    const OllamaRequest high = think("qwen3-4b", R"("high")");
    CHECK(json::parse(high.body)["reasoning_effort"] == "high");
    CHECK_FALSE(json::parse(think("qwen3-4b", "null").body).contains("chat_template_kwargs"));

    // qwen3-4b can switch both ways; lfm25-1.2b-thinking cannot stop reasoning
    // and lfm25-350m cannot start. Restating what a model does is accepted.
    CHECK_FALSE(decision(off, "qwen3-4b").thinking);
    CHECK(decision(think("qwen3-4b", "true"), "qwen3-4b").thinking);
    CHECK(decision(high, "qwen3-4b").thinking);
    CHECK(decision(think("lfm25-350m", "false"), "lfm25-350m").error.empty());
    CHECK(decision(think("lfm25-1.2b-thinking", "true"), "lfm25-1.2b-thinking").thinking);
    const ReasoningDecision stop =
        decision(think("lfm25-1.2b-thinking", "false"), "lfm25-1.2b-thinking");
    CHECK(std::string(stop.code) == kReasoningUnsupportedCode);
    CHECK(decision(think("lfm25-350m", R"("low")"), "lfm25-350m").error.find("does not reason") !=
          std::string::npos);

    for (const char* bad : {R"("none")", R"("max")", "1", "[]"}) {
        CAPTURE(bad);
        CHECK(think("qwen3-4b", bad).error == kOllamaThinkInvalid);
    }
}

TEST_CASE("Ollama: same-name tool results bind in call order") {
    const auto request = parse_ollama_request(R"({
        "model":"chat",
        "tools":[{"type":"function","function":{"name":"weather"}}],
        "messages":[{"role":"user","content":"weather?"},
            {"role":"assistant","tool_calls":[
                {"function":{"name":"weather","arguments":{"city":"A"}}},
                {"function":{"name":"weather","arguments":{"city":"B"}}}]},
            {"role":"tool","tool_name":"weather","content":"rain"},
            {"role":"tool","tool_name":"weather","content":"sun"}]})",
                                              OllamaRoute::Chat);
    REQUIRE_MESSAGE(request.error.empty(), request.error);
    CHECK_FALSE(request.replay_without_tools);
    const auto transcript = parse_tool_request(request.body);
    REQUIRE_MESSAGE(transcript.error.empty(), transcript.error);
    REQUIRE(transcript.messages.size() == 4);
    CHECK(transcript.messages[1].tool_calls[0].id.size() == 9);
    CHECK(transcript.messages[2].tool_call_id == transcript.messages[1].tool_calls[0].id);
    CHECK(transcript.messages[2].content == "rain");
    CHECK(transcript.messages[3].tool_call_id == transcript.messages[1].tool_calls[1].id);
    CHECK(
        parse_ollama_request(
            R"({"model":"chat","messages":[{"role":"tool","tool_name":"unknown","content":"r"}]})",
            OllamaRoute::Chat)
            .error == "tool result has no matching assistant call");
    CHECK(parse_ollama_request(R"({"model":"chat","messages":[{"role":"user","content":"q"},)"
                               R"({"role":"assistant","tool_calls":[{"function":{"name":"w",)"
                               R"("arguments":"{}"}}]},{"role":"tool","tool_name":"w"}]})",
                               OllamaRoute::Chat)
              .error == "tool call function.arguments must be an object");
}

TEST_CASE("Ollama: results reorder into call order and keep a client's id") {
    const auto request = parse_ollama_request(R"({
        "model":"chat","tools":[{"type":"function","function":{"name":"weather"}}],
        "messages":[{"role":"user","content":"weather?"},
            {"role":"assistant","tool_calls":[
                {"id":"custom001","function":{"name":"weather","arguments":{}}},
                {"function":{"name":"time","arguments":{}}}]},
            {"role":"tool","tool_name":"time","content":"three"},
            {"role":"tool","tool_name":"weather","content":"rain"}]})",
                                              OllamaRoute::Chat);
    REQUIRE_MESSAGE(request.error.empty(), request.error);
    const auto transcript = parse_tool_request(request.body);
    REQUIRE_MESSAGE(transcript.error.empty(), transcript.error);
    REQUIRE(transcript.messages.size() == 4);
    CHECK(transcript.messages[1].tool_calls[0].id == "custom001");
    CHECK(transcript.messages[2].tool_call_id == "custom001");
    CHECK(transcript.messages[2].content == "rain");
    CHECK(transcript.messages[3].content == "three");
}

TEST_CASE("Ollama: a tool replay without tools still renders through the template") {
    const auto request = parse_ollama_request(R"({
        "model":"chat","messages":[{"role":"user","content":"weather?"},
            {"role":"assistant","tool_calls":[
                {"function":{"name":"weather","arguments":{"city":"A"}}}]},
            {"role":"tool","tool_name":"weather","content":"rain"}]})",
                                              OllamaRoute::Chat);
    REQUIRE_MESSAGE(request.error.empty(), request.error);
    CHECK(request.replay_without_tools);
    // Without the flag it is plain chat, which cannot render tool turns.
    CHECK(parse_tool_request(request.body).disposition == ApiToolDisposition::None);
    const auto transcript = parse_tool_request(request.body, request.replay_without_tools);
    REQUIRE_MESSAGE(transcript.error.empty(), transcript.error);
    CHECK(transcript.disposition == ApiToolDisposition::Proposal);
    CHECK(transcript.choice == ToolChoice::None);
    CHECK(transcript.tools.empty());
    REQUIRE(transcript.messages.size() == 3);
    CHECK(transcript.messages.back().tool_call_id == transcript.messages[1].tool_calls[0].id);
}

TEST_CASE("Ollama: the native routes serve GGUF text models with a plain id") {
    CHECK(ollama_chat_model("lfm25-350m", true, false));
    CHECK_FALSE(ollama_chat_model("smollm2-360m-cpu-int4", false, false)); // ONNX
    CHECK_FALSE(ollama_chat_model("bge-m3", true, true));                  // embedding
    CHECK_FALSE(ollama_chat_model("C:\\models\\x.gguf", true, false));     // a raw path
    CHECK(ollama_model_id("qwen3-4b.Q4_K_M"));
    CHECK_FALSE(ollama_model_id(""));
    CHECK_FALSE(ollama_model_id(".."));
    CHECK_FALSE(ollama_model_id("qwen3:4b"));
}
