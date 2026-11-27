// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
#include "xllama/api_discovery.h"
#include "xllama/llama_patches.h"

#include <doctest/doctest.h>
#include <nlohmann/json.hpp>

using nlohmann::json;

TEST_CASE("health_json keeps the original fields first and adds version, api, capabilities") {
    xllama::ApiCapabilities caps;
    caps.tools = true;
    caps.images = false;
    const std::string body = xllama::health_json("1.8.1.1805", caps, "d812350");
    CHECK(body.rfind("{\"status\":\"ok\",\"service\":\"xllama\",", 0) == 0);
    const json j = json::parse(body);
    CHECK(j["version"] == "1.8.1.1805");
    CHECK(j["api"] == xllama::kApiContractVersion);
    CHECK(j["capabilities"]["tools"] == true);
    CHECK(j["capabilities"]["images"] == false);
    CHECK(j["capabilities"]["reasoning_content"] == false);
    CHECK(j["capabilities"]["reasoning_tokens"] == false);
    CHECK(j["capabilities"]["cached_tokens"] == false);
    CHECK(j["capabilities"]["model_sampling"] == false);
    CHECK(j["capabilities"]["ollama_chat"] == false);
    CHECK(j["capabilities"]["ollama_keep_alive"] == false);
    CHECK(j["capabilities"]["chat_context"] == false);
    CHECK(j["capabilities"].size() == 14);
    // The accepted sampling fields, from the same table the request is checked against.
    CHECK(j["sampling"] ==
          json::array({"temperature", "top_p", "top_k", "repetition_penalty", "seed"}));
    CHECK(j["response_formats"] == json::array({"text"}));
    // The pin and every applied patch (#611), so a client can tell which upstream fixes it has.
    CHECK(j["runtime"]["llama_cpp"] == "d812350");
    REQUIRE(j["runtime"]["patches"].size() == xllama::kLlamaPatches.size());
    CHECK(j["runtime"]["patches"][0] == "0001-uwp-appcontainer-guards");
}

TEST_CASE("health_json leaves runtime out of a build without llama.cpp") {
    const json j = json::parse(xllama::health_json("1.9.0.1", xllama::ApiCapabilities{}, ""));
    CHECK_FALSE(j.contains("runtime"));
}

#ifdef XLLAMA_USE_LLAMA
TEST_CASE("build_llama_cpp is the compiled pin's commit, not a build number") {
    const std::string pin = xllama::build_llama_cpp();
    REQUIRE_FALSE(pin.empty());
    CHECK(pin.find_first_not_of("0123456789abcdef") == std::string::npos);
}
#endif

TEST_CASE("build_api_capabilities follows the compiled backends") {
    const xllama::ApiCapabilities caps = xllama::build_api_capabilities();
#ifdef XLLAMA_USE_LLAMA
    CHECK(caps.tools);
    CHECK(caps.embeddings);
#endif
#ifndef XLLAMA_USE_ORT
    CHECK_FALSE(caps.images);
#endif
    CHECK(caps.pull);
    CHECK(caps.reasoning_content);
    CHECK(caps.reasoning_tokens);
    CHECK(caps.cached_tokens);
    CHECK(caps.model_sampling);
    CHECK(caps.chat_context);
#ifdef XLLAMA_USE_LLAMA
    CHECK(caps.ollama_chat);
    CHECK(caps.ollama_keep_alive);
#endif
}

TEST_CASE("models_list_json is an OpenAI list with xllama's per-model fields") {
    xllama::ModelInfo chat;
    chat.id = "lfm25-1.2b-instruct";
    chat.display = "LFM2.5 1.2B \"Instruct\"";
    chat.kind = "gguf";
    chat.n_ctx = 4096;
    chat.active = true;
    chat.tools = true;
    chat.ollama_chat = true;
    xllama::SamplingOverrides card;
    card.temperature = 0.1f;
    card.top_k = 50;
    chat.has_sampling = true;
    chat.sampling = xllama::resolve_sampling(card);
    xllama::ModelInfo raw;
    raw.id = "C:\\models\\x.gguf"; // a model.txt path outside the catalogue
    const json j = json::parse(xllama::models_list_json({chat, raw}, 1700000000));
    CHECK(j["object"] == "list");
    REQUIRE(j["data"].size() == 2);
    const json& a = j["data"][0];
    CHECK(a["id"] == "lfm25-1.2b-instruct");
    CHECK(a["object"] == "model");
    CHECK(a["created"] == 1700000000);
    CHECK(a["owned_by"] == "xllama");
    CHECK(a["active"] == true);
    CHECK(a["display"] == "LFM2.5 1.2B \"Instruct\"");
    CHECK(a["kind"] == "gguf");
    CHECK(a["role"] == "");
    CHECK(a["n_ctx"] == 4096);
    CHECK(a["capabilities"]["tools"] == true);
    CHECK(a["capabilities"]["reasoning"] == false);
    CHECK(a["capabilities"]["ollama_chat"] == true);
    // The values a request that sets none of them gets, written as short decimals.
    const json& s = a["sampling"];
    CHECK(a.dump().find("\"temperature\":0.1,") != std::string::npos);
    CHECK(s["temperature"].get<double>() == doctest::Approx(0.1));
    CHECK(s["top_p"].get<double>() == doctest::Approx(xllama::sampling_defaults::kTopP));
    CHECK(s["top_k"] == 50);
    CHECK(s["repetition_penalty"].get<double>() ==
          doctest::Approx(xllama::sampling_defaults::kRepetitionPenalty));
    CHECK(s["source"] == json({{"temperature", "model"},
                               {"top_p", "default"},
                               {"top_k", "model"},
                               {"repetition_penalty", "default"}}));
    const json& b = j["data"][1];
    CHECK(b["id"] == "C:\\models\\x.gguf");
    CHECK_FALSE(b.contains("active"));
    CHECK_FALSE(b.contains("display"));
    CHECK_FALSE(b.contains("n_ctx"));
    CHECK_FALSE(b.contains("sampling"));
    CHECK(b["capabilities"]["tools"] == false);
    CHECK(b["capabilities"]["ollama_chat"] == false);
}
