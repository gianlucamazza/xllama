// Copyright (c) 2024 Venere Labs
// SPDX-License-Identifier: MIT

#include <doctest/doctest.h>

#include "xllama/chat_prompt.h"

using namespace xllama;

TEST_CASE("qwen detection") {
    CHECK(model_is_qwen("qwen35-0.8b"));
    CHECK(model_is_qwen("Qwen3.5-0.8B-Q4_K_M.gguf"));
    CHECK_FALSE(model_is_qwen("lfm25-350m"));
    CHECK_FALSE(model_is_qwen("smollm2-360m-cpu-int4"));
}

TEST_CASE("qwen no-think generation suffix") {
    const std::string suffix = qwen_no_think_gen_suffix("qwen35-0.8b");
    CHECK_FALSE(suffix.empty());
    CHECK(suffix.find("qwen35") == std::string::npos);
    CHECK(suffix.find('\n') != std::string::npos);
    CHECK(qwen_no_think_gen_suffix("lfm25-350m").empty());
}

TEST_CASE("strip empty thinking tags") {
    const std::string block = std::string("<think>") + "\n\n" + "</think>";
    CHECK(strip_empty_thinking_tags(block + "\n\nCiao!") == "Ciao!");
    CHECK(strip_empty_thinking_tags("  \n" + block + "  \n  Risposta") == "Risposta");
    CHECK(strip_empty_thinking_tags("plain text") == "plain text");
}

TEST_CASE("apply_stop_sequences: suffix match trims and reports") {
    // Ends with the stop -> true, trailing match trimmed.
    std::string a = "Hello there<end_of_turn>";
    CHECK(apply_stop_sequences(a, {"<end_of_turn>"}));
    CHECK(a == "Hello there");

    // ChatML stop.
    std::string b = "Ciao<|im_end|>";
    CHECK(apply_stop_sequences(b, {"<|im_end|>"}));
    CHECK(b == "Ciao");

    // Does not end with a stop -> false, unchanged.
    std::string c = "still going";
    CHECK_FALSE(apply_stop_sequences(c, {"<end_of_turn>"}));
    CHECK(c == "still going");

    // The divergence fix: a stop that appears MID-output (not at the end) must
    // NOT trigger (substring find() would have wrongly truncated here).
    std::string d = "a<end_of_turn>b";
    CHECK_FALSE(apply_stop_sequences(d, {"<end_of_turn>"}));
    CHECK(d == "a<end_of_turn>b");

    // Empty stop ignored; multiple stops, first suffix match wins.
    std::string e = "done<|im_end|>";
    CHECK(apply_stop_sequences(e, {"", "<end_of_turn>", "<|im_end|>"}));
    CHECK(e == "done");

    // No stops -> false.
    std::string f = "text";
    CHECK_FALSE(apply_stop_sequences(f, {}));
    CHECK(f == "text");
}

TEST_CASE("gemma detection") {
    CHECK(model_is_gemma("gemma3-270m"));
    CHECK(model_is_gemma("Gemma-4-E2B-it-Q4_K_M.gguf"));
    CHECK_FALSE(model_is_gemma("smollm2-360m-cpu-int4"));
    CHECK_FALSE(model_is_gemma("qwen35-0.8b"));
    CHECK_FALSE(model_is_gemma("lfm25-350m"));
}

TEST_CASE("llama detection") {
    CHECK(model_is_llama("llama32-3b"));
    CHECK(model_is_llama("Llama-3.2-3B-Instruct-Q3_K_S.gguf"));
    CHECK_FALSE(model_is_llama("smollm2-360m-cpu-int4"));
    CHECK_FALSE(model_is_llama("lfm25-350m"));
    CHECK_FALSE(model_is_llama("gemma3-270m"));
}

TEST_CASE("chat format selection") {
    CHECK(chat_format_for("smollm2-360m-cpu-int4").kind == ChatFormatKind::ChatML);
    CHECK(chat_format_for("lfm25-350m").kind == ChatFormatKind::ChatML);
    CHECK(chat_format_for("gemma3-270m").kind == ChatFormatKind::Gemma);
    CHECK(chat_format_for("llama32-3b").kind == ChatFormatKind::Llama3);

    const ChatFormat qwen = chat_format_for("qwen35-0.8b");
    CHECK(qwen.kind == ChatFormatKind::ChatML);
    CHECK_FALSE(qwen.gen_suffix.empty()); // Qwen no-think prefill

    CHECK(chat_format_for("smollm2-360m-cpu-int4").gen_suffix.empty());
    CHECK(chat_format_for("llama32-3b").gen_suffix.empty());
}

TEST_CASE("chat format stop sequences") {
    CHECK(chat_format_for("smollm2-360m-cpu-int4").stop_sequences ==
          std::vector<std::string>{"<|im_end|>"});
    CHECK(chat_format_for("gemma3-270m").stop_sequences ==
          std::vector<std::string>{"<end_of_turn>"});
    CHECK(chat_format_for("llama32-3b").stop_sequences ==
          std::vector<std::string>{"<|eot_id|>"});
}

TEST_CASE("chatml render is byte-exact with the legacy hand-built prompt") {
    const ChatFormat f = chat_format_for("smollm2-360m-cpu-int4");
    const std::string sys = "You are helpful.";
    const std::vector<ChatTurn> hist = {{"hi", "hello there"}};
    const std::string final_user = "how are you?";

    // Legacy string builder (from MainPage::BuildChatMLPrompt, no gen_suffix).
    std::string legacy = "<|im_start|>system\n" + sys + "<|im_end|>\n";
    for (const ChatTurn& t : hist) {
        legacy += "<|im_start|>user\n" + t.user + "<|im_end|>\n";
        legacy += "<|im_start|>assistant\n" + t.assistant + "<|im_end|>\n";
    }
    legacy += "<|im_start|>user\n" + final_user + "<|im_end|>\n";
    legacy += "<|im_start|>assistant\n";

    CHECK(f.render_prompt(sys, hist, final_user) == legacy);
}

TEST_CASE("chatml render appends qwen suffix only to the trailing header") {
    const ChatFormat f = chat_format_for("qwen35-0.8b");
    const std::string p = f.render_prompt("sys", {{"u1", "a1"}}, "u2");
    // Trailing header carries the suffix...
    CHECK(p.substr(p.size() - f.gen_suffix.size()) == f.gen_suffix);
    // ...but the completed history assistant turn does not.
    const std::string hist_hdr = "<|im_start|>assistant\na1<|im_end|>\n";
    CHECK(p.find(hist_hdr) != std::string::npos);
}

TEST_CASE("gemma render merges system into the first user turn, no bos/system") {
    const ChatFormat f = chat_format_for("gemma3-270m");
    const std::string p = f.render_prompt("Be nice.", {}, "ciao");
    CHECK(p == "<start_of_turn>user\nBe nice.\n\nciao<end_of_turn>\n<start_of_turn>model\n");
    CHECK(p.find("<bos>") == std::string::npos);
    CHECK(p.find("system") == std::string::npos);
    CHECK(f.gen_suffix.empty());
}

TEST_CASE("gemma multi-turn merges system only into the first user turn") {
    const ChatFormat f = chat_format_for("gemma3-270m");
    const std::string p = f.render_prompt("SYS", {{"u1", "a1"}}, "u2");
    CHECK(p == "<start_of_turn>user\nSYS\n\nu1<end_of_turn>\n"
               "<start_of_turn>model\na1<end_of_turn>\n"
               "<start_of_turn>user\nu2<end_of_turn>\n"
               "<start_of_turn>model\n");
}

TEST_CASE("gemma render with empty system leaves the first user turn unprefixed") {
    const ChatFormat f = chat_format_for("gemma3-270m");
    CHECK(f.render_prompt("", {}, "ciao") ==
          "<start_of_turn>user\nciao<end_of_turn>\n<start_of_turn>model\n");
}

TEST_CASE("render_delta for chatml, gemma, llama3 and prev_ended_with_stop") {
    const ChatFormat cm = chat_format_for("smollm2-360m-cpu-int4");
    CHECK(cm.render_delta("q", /*stop=*/true) ==
          "\n<|im_start|>user\nq<|im_end|>\n<|im_start|>assistant\n");
    CHECK(cm.render_delta("q", /*stop=*/false) ==
          "<|im_end|>\n<|im_start|>user\nq<|im_end|>\n<|im_start|>assistant\n");

    const ChatFormat gm = chat_format_for("gemma3-270m");
    CHECK(gm.render_delta("q", /*stop=*/true) ==
          "\n<start_of_turn>user\nq<end_of_turn>\n<start_of_turn>model\n");
    CHECK(gm.render_delta("q", /*stop=*/false) ==
          "<end_of_turn>\n<start_of_turn>user\nq<end_of_turn>\n<start_of_turn>model\n");

    // Llama-3: stop == turn_close, so post-stop glue is empty (no extra newline).
    const ChatFormat lm = chat_format_for("llama32-3b");
    CHECK(lm.render_delta("q", /*stop=*/true) ==
          "<|start_header_id|>user<|end_header_id|>\n\nq<|eot_id|>"
          "<|start_header_id|>assistant<|end_header_id|>\n\n");
    CHECK(lm.render_delta("q", /*stop=*/false) ==
          "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nq<|eot_id|>"
          "<|start_header_id|>assistant<|end_header_id|>\n\n");
}

TEST_CASE("llama3 render matches Meta instruct header layout") {
    const ChatFormat f = chat_format_for("llama32-3b");
    const std::string p = f.render_prompt("Be concise.", {{"hi", "hello"}}, "how are you?");
    CHECK(p == "<|start_header_id|>system<|end_header_id|>\n\nBe concise.<|eot_id|>"
               "<|start_header_id|>user<|end_header_id|>\n\nhi<|eot_id|>"
               "<|start_header_id|>assistant<|end_header_id|>\n\nhello<|eot_id|>"
               "<|start_header_id|>user<|end_header_id|>\n\nhow are you?<|eot_id|>"
               "<|start_header_id|>assistant<|end_header_id|>\n\n");
    CHECK(p.find("<|begin_of_text|>") == std::string::npos); // BOS via tokenizer
}

TEST_CASE("postprocess_output strips empty think for chatml, passthrough otherwise") {
    const std::string block = std::string("<think>") + "\n\n" + "</think>";
    CHECK(chat_format_for("qwen35-0.8b").postprocess_output(block + "\n\nCiao!") == "Ciao!");
    CHECK(chat_format_for("gemma3-270m").postprocess_output("plain gemma") == "plain gemma");
    CHECK(chat_format_for("llama32-3b").postprocess_output("plain llama") == "plain llama");
}

// The core KV-reuse safety net: the string a reused KV holds after a turn — the
// full prompt, the assistant output, plus the delta for the next turn — must
// equal a cold full re-prefill of the same conversation with that turn appended.
// Only meaningful when gen_suffix is empty: a non-empty suffix is prefilled into
// the KV but intentionally absent from stored output, a known/preserved
// discrepancy (see chat_prompt.cpp risks), so the strings would differ by design.
static void check_kv_reuse_invariant(const ChatFormat& f) {
    REQUIRE(f.gen_suffix.empty());
    const std::string sys = "sys";
    const std::vector<ChatTurn> hist = {{"u1", "a1"}};
    const std::string final_user = "u2";
    const std::string raw_out = "a2";
    const std::string next_user = "u3";
    const std::string stop = f.stop_sequences.front();

    const std::string base = f.render_prompt(sys, hist, final_user); // suffix empty

    std::vector<ChatTurn> hist2 = hist;
    hist2.push_back({final_user, raw_out});
    const std::string cold = f.render_prompt(sys, hist2, next_user);

    // Case A: generation stopped on the stop token (token resident in the KV);
    // the delta only re-opens with a newline.
    CHECK(base + raw_out + stop + f.render_delta(next_user, /*prev_ended_with_stop=*/true) == cold);

    // Case B: generation cut short by n_predict (no stop token in the KV); the
    // delta closes the turn itself.
    CHECK(base + raw_out + f.render_delta(next_user, /*prev_ended_with_stop=*/false) == cold);
}

TEST_CASE("kv-reuse invariant holds for chatml, gemma, and llama3") {
    check_kv_reuse_invariant(chat_format_for("smollm2-360m-cpu-int4"));
    check_kv_reuse_invariant(chat_format_for("gemma3-270m"));
    check_kv_reuse_invariant(chat_format_for("llama32-3b"));
}