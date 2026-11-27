// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include <doctest/doctest.h>

#include "xllama/chat_prompt.h"

using namespace xllama;

TEST_CASE("qwen detection") {
    CHECK(model_is_qwen("qwen35-0.8b"));
    CHECK(model_is_qwen("Qwen3.5-0.8B-Q4_K_M.gguf"));
    CHECK(model_is_qwen("qwen25-coder-1.5b"));
    CHECK_FALSE(model_is_qwen("lfm25-350m"));
    CHECK_FALSE(model_is_qwen("smollm2-360m-cpu-int4"));
}

TEST_CASE("qwen3 detection excludes Qwen2.5-Coder") {
    CHECK(model_is_qwen3("qwen35-0.8b"));
    CHECK(model_is_qwen3("Qwen3.5-0.8B-Q4_K_M.gguf"));
    CHECK_FALSE(model_is_qwen3("qwen25-coder-1.5b"));
    CHECK_FALSE(model_is_qwen3("Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf"));
    CHECK_FALSE(model_is_qwen3("lfm25-350m"));
}

TEST_CASE("qwen no-think generation suffix") {
    const std::string suffix = qwen_no_think_gen_suffix("qwen35-0.8b");
    CHECK_FALSE(suffix.empty());
    CHECK(suffix.find("qwen35") == std::string::npos);
    CHECK(suffix.find('\n') != std::string::npos);
    CHECK(qwen_no_think_gen_suffix("lfm25-350m").empty());
    // Qwen2.5-Coder is plain ChatML — no empty <think> prefill.
    CHECK(qwen_no_think_gen_suffix("qwen25-coder-1.5b").empty());
    CHECK(qwen_no_think_gen_suffix("Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf").empty());
}

TEST_CASE("coding models use ChatML without think suffix") {
    const auto fmt = chat_format_for("qwen25-coder-1.5b");
    CHECK(fmt.kind == ChatFormatKind::ChatML);
    CHECK(fmt.gen_suffix.empty());
    CHECK(fmt.stop_sequences.size() == 1);
    CHECK(fmt.stop_sequences[0] == "<|im_end|>");
}

TEST_CASE("strip empty thinking tags") {
    const std::string block = std::string("<think>") + "\n\n" + "</think>";
    CHECK(strip_empty_thinking_tags(block + "\n\nCiao!") == "Ciao!");
    CHECK(strip_empty_thinking_tags("  \n" + block + "  \n  Risposta") == "Risposta");
    CHECK(strip_empty_thinking_tags("plain text") == "plain text");
}

TEST_CASE("thinking model detection and postprocess") {
    CHECK(model_is_thinking("lfm25-1.2b-thinking"));
    CHECK(model_is_thinking("LFM2.5-1.2B-Thinking-Q4_K_M.gguf"));
    CHECK_FALSE(model_is_thinking("lfm25-1.2b-instruct"));
    CHECK_FALSE(model_is_thinking("qwen35-0.8b"));
    // LFM2.5-8B-A1B reasons on every turn but does not say "thinking" anywhere
    // in its name; its template emits <think> unconditionally. Detecting it by
    // family is what keeps raw CoT out of the UI and KV snapshots off a model
    // whose stripped history can never prefix-match its CoT-bearing cache.
    CHECK(model_is_thinking("lfm25-8b-a1b"));
    CHECK(model_is_thinking("LFM2.5-8B-A1B-UD-IQ3_S.gguf"));
    // The sibling dense models must stay non-thinking.
    CHECK_FALSE(model_is_thinking("lfm25-350m"));
    CHECK_FALSE(model_is_thinking("lfm2-2.6b"));

    const auto fmt = chat_format_for("lfm25-1.2b-thinking");
    CHECK(fmt.kind == ChatFormatKind::ChatML);
    CHECK(fmt.gen_suffix.empty()); // no Qwen3 no-think prefill
    CHECK(fmt.strip_thinking_content);

    const std::string raw = std::string("<think>") + " step by step\n" + "</think>" + "\n\n4";
    CHECK(fmt.postprocess_output(raw) == "4");
    // Truncated mid-thought: no answer left.
    CHECK(fmt.postprocess_output(std::string("<think>") + " still going").empty());
    // Non-thinking ChatML still only strips empty think blocks.
    const auto plain = chat_format_for("lfm25-350m");
    CHECK_FALSE(plain.strip_thinking_content);
    CHECK(plain.postprocess_output(raw).find("<think>") != std::string::npos);
}

TEST_CASE("strip_thinking_blocks complete and truncated") {
    CHECK(strip_thinking_blocks("<think>a</think>\n\nanswer") == "answer");
    CHECK(strip_thinking_blocks("<think>a</think><think>b</think>ok") == "ok");
    CHECK(strip_thinking_blocks("no tags") == "no tags");
    CHECK(strip_thinking_blocks("<think>cut off").empty());
}

TEST_CASE("strip_thinking_blocks handles a closer with no opener") {
    // The model's own template can open the reasoning, so the stream carries
    // only the closing tag: everything up to it is chain of thought.
    CHECK(strip_thinking_blocks("reasoning first</think>\n\nanswer") == "answer");
    CHECK(strip_thinking_blocks("</think>answer") == "answer");
    CHECK(strip_thinking_blocks("a</think>b<think>c</think>d") == "bd");
    // An opener before the closer is the normal balanced case, untouched.
    CHECK(strip_thinking_blocks("<think>a</think>b") == "b");
    // Reasoning only, no answer yet.
    CHECK(strip_thinking_blocks("reasoning, no closer").empty() == false);
    CHECK(strip_thinking_blocks("thoughts</think>").empty());
    // A stray closer after a balanced block is a tag, not a boundary: the answer
    // stays, the tag goes.
    CHECK(strip_thinking_blocks("<think>a</think>answer</think>") == "answer");
    CHECK(strip_thinking_blocks("<think>a<think>b</think>c</think>d") == "cd");
    CHECK(strip_thinking_blocks("answer</think>trailing").find("</think>") == std::string::npos);
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

TEST_CASE("gemma 4 detection") {
    CHECK(model_is_gemma4("gemma4-e2b"));
    CHECK(model_is_gemma4("gemma-4-E2B-it-Q3_K_S.gguf"));
    CHECK(model_is_gemma4("/models/gemma4-e2b/Gemma_4_E4B.gguf"));
    CHECK(model_is_gemma("gemma4-e2b")); // still a Gemma id for callers that only ask that
    CHECK_FALSE(model_is_gemma4("gemma3-270m"));
    CHECK_FALSE(model_is_gemma4("gemma-3-1b-it-Q4_K_M.gguf"));
    CHECK_FALSE(model_is_gemma4("/models/gemma-4/lfm25-350m.gguf")); // basename only
}

TEST_CASE("llama detection") {
    CHECK(model_is_llama("llama32-3b"));
    CHECK(model_is_llama("Llama-3.2-3B-Instruct-Q3_K_S.gguf"));
    CHECK_FALSE(model_is_llama("smollm2-360m-cpu-int4"));
    CHECK_FALSE(model_is_llama("lfm25-350m"));
    CHECK_FALSE(model_is_llama("gemma3-270m"));
}

TEST_CASE("phi detection") {
    CHECK(model_is_phi("phi35-mini"));
    CHECK(model_is_phi("Phi-3.5-mini-instruct-Q3_K_S.gguf"));
    CHECK_FALSE(model_is_phi("smollm2-360m-cpu-int4"));
    CHECK_FALSE(model_is_phi("lfm25-350m"));
    CHECK_FALSE(model_is_phi("llama32-3b"));
    CHECK_FALSE(model_is_phi("gemma3-270m"));
}

TEST_CASE("granite detection") {
    CHECK(model_is_granite("granite40-h-tiny"));
    CHECK(model_is_granite("granite-4.0-h-tiny-Q3_K_S.gguf"));
    CHECK(model_is_granite("granite-4.2-3b")); // #521 reuses this renderer
    CHECK(model_is_granite("/models/cache/granite-4.0-h-tiny-Q3_K_S.gguf"));
    CHECK_FALSE(model_is_granite("lfm25-350m"));
    CHECK_FALSE(model_is_granite("llama32-3b"));
    CHECK_FALSE(model_is_granite("phi4-mini"));
    // Basename only: a cache dir named granite must not select the template.
    CHECK_FALSE(model_is_granite("/models/granite-cache/lfm25-350m.gguf"));
}

TEST_CASE("granite wins over a llama substring in the same basename") {
    // Re-quantised Granite files are often named after the tool or runtime
    // ("…-ollama", "…-llamacpp"); "llama" is a substring there, not the family.
    CHECK(chat_format_for("granite-4.0-h-tiny-ollama-Q4_K_M.gguf").kind == ChatFormatKind::Granite);
    CHECK(chat_format_for("/m/granite-4.2-3b-llamacpp-Q4_0.gguf").kind == ChatFormatKind::Granite);
    CHECK(chat_format_for("Llama-3.2-3B-Instruct-Q3_K_S.gguf").kind == ChatFormatKind::Llama3);
}

TEST_CASE("model detection ignores directory names (full-path ids)") {
    // The CLI passes a FULL path; directory names must not select a template.
    // A cache dir literally named "xllama-gguf" contains "llama" and used to
    // force the Llama-3 template onto LFM2.5 (which then echoed <|eot_id|> as
    // text instead of answering).
    CHECK_FALSE(model_is_llama("/home/user/.cache/xllama-gguf/LFM2.5-350M-Q4_K_M.gguf"));
    CHECK(chat_format_for("/home/user/.cache/xllama-gguf/LFM2.5-350M-Q4_K_M.gguf").kind ==
          ChatFormatKind::ChatML);
    CHECK(model_is_llama("/models/lfm-dir/Llama-3.2-3B-Instruct-Q3_K_S.gguf"));
    CHECK_FALSE(model_is_phi("C:\\phi-cache\\LFM2.5-350M-Q4_K_M.gguf"));
    CHECK(model_is_phi("C:\\models\\Phi-3.5-mini-instruct-Q3_K_S.gguf"));
    CHECK_FALSE(model_is_gemma("/srv/gemma-store/lfm25-350m.gguf"));
    CHECK_FALSE(model_is_smollm3("models/smollm3-cache/lfm25-350m.gguf"));
    CHECK(chat_format_for("models/smollm3-cache/lfm25-350m.gguf").kind == ChatFormatKind::ChatML);
    CHECK_FALSE(model_is_granite("/srv/granite-store/lfm25-350m.gguf"));
    CHECK(model_is_granite("/srv/cache/granite-4.0-h-tiny-Q3_K_S.gguf"));
    CHECK(chat_format_for("models/olmoe-cache/lfm25-350m.gguf").kind == ChatFormatKind::ChatML);
    CHECK(chat_format_for("models/OLMoE-1B-7B-0125-Instruct-Q3_K_S.gguf").kind ==
          ChatFormatKind::Olmoe);
}

TEST_CASE("chat format selection") {
    CHECK(chat_format_for("smollm2-360m-cpu-int4").kind == ChatFormatKind::ChatML);
    CHECK(chat_format_for("lfm25-350m").kind == ChatFormatKind::ChatML);
    CHECK(chat_format_for("qwen25-coder-1.5b").kind == ChatFormatKind::ChatML);
    CHECK(chat_format_for("lfm25-1.2b-instruct").kind == ChatFormatKind::ChatML);
    CHECK(chat_format_for("lfm2-2.6b").kind == ChatFormatKind::ChatML);
    CHECK(chat_format_for("gemma3-270m").kind == ChatFormatKind::Gemma);
    CHECK(chat_format_for("gemma3-1b").kind == ChatFormatKind::Gemma);
    CHECK(chat_format_for("llama32-3b").kind == ChatFormatKind::Llama3);
    CHECK(chat_format_for("phi35-mini").kind == ChatFormatKind::Phi3);
    CHECK(chat_format_for("phi4-mini").kind == ChatFormatKind::Phi3);
    CHECK(chat_format_for("granite40-h-tiny").kind == ChatFormatKind::Granite);
    CHECK(chat_format_for("granite-4.2-3b").kind == ChatFormatKind::Granite);

    const ChatFormat qwen = chat_format_for("qwen35-0.8b");
    CHECK(qwen.kind == ChatFormatKind::ChatML);
    CHECK_FALSE(qwen.gen_suffix.empty()); // Qwen no-think prefill

    CHECK(chat_format_for("smollm2-360m-cpu-int4").gen_suffix.empty());
    CHECK(chat_format_for("llama32-3b").gen_suffix.empty());
    CHECK(chat_format_for("phi35-mini").gen_suffix.empty());
    CHECK(chat_format_for("granite40-h-tiny").gen_suffix.empty());
}

TEST_CASE("minicpm5 needs both renderer halves, and nothing else gets them") {
    // Phase 16 H16.1d. Measured on the vendor GGUF: without <s> the model closes
    // the turn immediately; with <s> alone it opens <think>; with both it answers
    // cleanly. So both halves are load-bearing and neither is cosmetic.
    const ChatFormat m = chat_format_for("minicpm5-1b-Q4_K_M.gguf");
    CHECK(m.kind == ChatFormatKind::ChatML); // NOT Llama3, though the arch is `llama`
    CHECK(m.bos == "<s>");
    CHECK(m.gen_suffix == "<think>\n\n</think>\n\n");
    CHECK_FALSE(m.strip_thinking_content); // would disable KV snapshots

    // The BOS must lead the prompt, and the #169 pinned head must be a byte-exact
    // prefix of it — that is the invariant a context shift relies on.
    const std::string sys = "You are helpful.";
    const std::string prompt = m.render_prompt(sys, {}, "hi");
    const std::string head = m.render_system_prefix(sys);
    CHECK(prompt.compare(0, 3, "<s>") == 0);
    CHECK(head.compare(0, 3, "<s>") == 0);
    CHECK(prompt.compare(0, head.size(), head) == 0);

    // The delta rides a KV that already holds the BOS; re-emitting it would
    // corrupt the reused cache.
    CHECK(m.render_delta("next", true).find("<s>") == std::string::npos);

    // No other catalogue id acquires a template BOS — Gemma and Llama-3 GGUF add
    // it in the tokenizer, so emitting it here would double it.
    for (const char* id : {"smollm2-360m-cpu-int4", "lfm25-350m", "qwen35-0.8b", "gemma3-270m",
                           "llama32-3b", "phi35-mini", "qwen25-coder-1.5b", "granite40-h-tiny"}) {
        CHECK(chat_format_for(id).bos.empty());
    }
    CHECK_FALSE(model_is_minicpm5("minicpm3-4b")); // MiniCPM3 is a different family
    CHECK(model_is_minicpm5("/models/MiniCPM5-1B/MiniCPM5-1B-Q4_K_M.gguf"));
}

TEST_CASE("model_is_thinking is the one reasoning decision plain chat applies (#655)") {
    // The tool path sets the template's enable_thinking from model_is_thinking,
    // and /v1/models reports it as capabilities.reasoning. Plain chat must agree:
    // a no-think prefill exactly when the family has one and the model does not
    // think, and think stripping exactly when it does.
    for (const char* id : {"qwen3-4b", "Qwen3-4B-Q4_K_M.gguf", "qwen35-0.8b", "qwen3-1.7b",
                           "minicpm5-1b-Q4_K_M.gguf"}) {
        CAPTURE(id);
        CHECK_FALSE(model_is_thinking(id));
        const ChatFormat f = chat_format_for(id);
        CHECK(f.gen_suffix == "<think>\n\n</think>\n\n");
        CHECK_FALSE(f.strip_thinking_content);
    }
    for (const char* id : {"qwen3-4b-thinking", "Qwen3-4B-Thinking-2507-Q4_K_M.gguf",
                           "minicpm5-1b-thinking", "lfm25-1.2b-thinking", "lfm25-8b-a1b"}) {
        CAPTURE(id);
        CHECK(model_is_thinking(id));
        const ChatFormat f = chat_format_for(id);
        // The MiniCPM5 template opens <think> itself when thinking (#648).
        const bool opens = model_is_minicpm5(id);
        CHECK(f.gen_suffix == (opens ? "<think>\n" : ""));
        CHECK(f.reasoning_opened == opens);
        CHECK(f.strip_thinking_content);
    }
    // MiniCPM5 keeps its template BOS whichever way it reasons.
    CHECK(chat_format_for("minicpm5-1b-thinking").bos == "<s>");
}

TEST_CASE("chat format stop sequences") {
    CHECK(chat_format_for("smollm2-360m-cpu-int4").stop_sequences ==
          std::vector<std::string>{"<|im_end|>"});
    CHECK(chat_format_for("gemma3-270m").stop_sequences ==
          std::vector<std::string>{"<end_of_turn>"});
    CHECK(chat_format_for("llama32-3b").stop_sequences == std::vector<std::string>{"<|eot_id|>"});
    CHECK(chat_format_for("phi35-mini").stop_sequences == std::vector<std::string>{"<|end|>"});
    CHECK(chat_format_for("granite40-h-tiny").stop_sequences ==
          std::vector<std::string>{"<|end_of_text|>"});
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

// Expected strings follow the canonical google/gemma-4-E2B-it chat template
// with thinking off: <|turn>role\n content <turn|>\n, system as its own turn.
TEST_CASE("gemma 4 render uses <|turn> tokens and a dedicated system turn") {
    const ChatFormat f = chat_format_for("gemma4-e2b");
    CHECK(f.kind == ChatFormatKind::Gemma);
    CHECK(f.system_style == SystemStyle::DedicatedTurn);
    const std::string p = f.render_prompt("Be nice.", {}, "ciao");
    CHECK(p == "<|turn>system\nBe nice.<turn|>\n<|turn>user\nciao<turn|>\n<|turn>model\n");
    CHECK(p.find("start_of_turn") == std::string::npos);
    CHECK(p.find("<bos>") == std::string::npos);
    CHECK(f.gen_suffix.empty());
    CHECK(f.stop_sequences == std::vector<std::string>{"<turn|>"});
}

TEST_CASE("gemma 4 multi-turn keeps the system turn out of the user turns") {
    const ChatFormat f = chat_format_for("gemma-4-E2B-it-Q3_K_S.gguf");
    const std::string p = f.render_prompt("SYS", {{"u1", "a1"}}, "u2");
    CHECK(p == "<|turn>system\nSYS<turn|>\n"
               "<|turn>user\nu1<turn|>\n"
               "<|turn>model\na1<turn|>\n"
               "<|turn>user\nu2<turn|>\n"
               "<|turn>model\n");
}

TEST_CASE("gemma 4 render_delta closes the turn only when the stop was not hit") {
    const ChatFormat f = chat_format_for("gemma4-e2b");
    CHECK(f.render_delta("q", /*stop=*/true) == "\n<|turn>user\nq<turn|>\n<|turn>model\n");
    CHECK(f.render_delta("q", /*stop=*/false) == "<turn|>\n<|turn>user\nq<turn|>\n<|turn>model\n");
}

TEST_CASE("gemma 3 keeps its template when gemma 4 is selected by id") {
    const ChatFormat f = chat_format_for("gemma-3-1b-it-Q4_K_M.gguf");
    CHECK(f.system_style == SystemStyle::MergeIntoFirstUser);
    CHECK(f.render_prompt("", {}, "ciao") ==
          "<start_of_turn>user\nciao<end_of_turn>\n<start_of_turn>model\n");
}

TEST_CASE("render_delta for chatml, gemma, llama3, phi3 and prev_ended_with_stop") {
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

    // Phi-3: stop is a prefix of turn_close ("<|end|>" vs "<|end|>\n") → post-stop glue "\n".
    const ChatFormat ph = chat_format_for("phi35-mini");
    CHECK(ph.render_delta("q", /*stop=*/true) == "\n<|user|>\nq<|end|>\n<|assistant|>\n");
    CHECK(ph.render_delta("q", /*stop=*/false) == "<|end|>\n<|user|>\nq<|end|>\n<|assistant|>\n");

    // Granite 4.x: stop is a prefix of turn_close ("<|end_of_text|>" vs
    // "<|end_of_text|>\n") → post-stop glue "\n".
    const ChatFormat gr = chat_format_for("granite40-h-tiny");
    CHECK(gr.render_delta("q", /*stop=*/true) ==
          "\n<|start_of_role|>user<|end_of_role|>q<|end_of_text|>\n"
          "<|start_of_role|>assistant<|end_of_role|>");
    CHECK(gr.render_delta("q", /*stop=*/false) ==
          "<|end_of_text|>\n<|start_of_role|>user<|end_of_role|>q<|end_of_text|>\n"
          "<|start_of_role|>assistant<|end_of_role|>");
}

TEST_CASE("phi3 render matches Microsoft instruct layout") {
    const ChatFormat f = chat_format_for("phi35-mini");
    const std::string p = f.render_prompt("Be concise.", {{"hi", "hello"}}, "how are you?");
    CHECK(p == "<|system|>\nBe concise.<|end|>\n"
               "<|user|>\nhi<|end|>\n"
               "<|assistant|>\nhello<|end|>\n"
               "<|user|>\nhow are you?<|end|>\n"
               "<|assistant|>\n");
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

TEST_CASE("kv-reuse invariant holds for chatml, gemma, llama3, phi3, and granite") {
    check_kv_reuse_invariant(chat_format_for("smollm2-360m-cpu-int4"));
    check_kv_reuse_invariant(chat_format_for("gemma3-270m"));
    check_kv_reuse_invariant(chat_format_for("gemma4-e2b"));
    check_kv_reuse_invariant(chat_format_for("llama32-3b"));
    check_kv_reuse_invariant(chat_format_for("phi35-mini"));
    check_kv_reuse_invariant(chat_format_for("granite40-h-tiny"));
    check_kv_reuse_invariant(chat_format_for("olmoe-1b-7b"));
}

TEST_CASE("render_system_prefix is the exact head of render_prompt (#169)") {
    const std::string sys = "You are terse.";
    // DedicatedTurn formats: the prefix must be byte-identical to what
    // render_prompt emits before the first user turn — a context shift pins
    // exactly its token count.
    for (const char* id : {"smollm2-360m-cpu-int4", "llama32-3b", "phi35-mini", "gemma4-e2b",
                           "granite40-h-tiny", "olmoe-1b-7b", "smollm3-3b"}) {
        const ChatFormat f = chat_format_for(id);
        const std::string prefix = f.render_system_prefix(sys);
        CHECK(!prefix.empty());
        CHECK(f.render_prompt(sys, {}, "hi").rfind(prefix, 0) == 0);
    }
    // MergeIntoFirstUser (Gemma 3): no standalone system block exists — the
    // prefix is empty and the caller pins only the BOS.
    CHECK(chat_format_for("gemma3-270m").render_system_prefix(sys).empty());
}

// Expected strings follow ibm-granite/granite-4.0-h-tiny chat_template.jinja
// (and the model-card golden) with add_generation_prompt: dedicated system
// turn, no space after <|end_of_role|>, generation header has no closer.
TEST_CASE("granite 4.x render matches the IBM instruct golden") {
    const ChatFormat f = chat_format_for("granite-4.0-h-tiny-Q3_K_S.gguf");
    CHECK(f.kind == ChatFormatKind::Granite);
    CHECK(f.system_style == SystemStyle::DedicatedTurn);
    CHECK(f.gen_suffix.empty());
    CHECK(f.stop_sequences == std::vector<std::string>{"<|end_of_text|>"});
    const std::string sys =
        "You are a helpful assistant. Please ensure responses are professional, accurate, and "
        "safe.";
    const std::string user =
        "Please list one IBM Research laboratory located in the United States. You should only "
        "output its name and location.";
    CHECK(f.render_prompt(sys, {}, user) == "<|start_of_role|>system<|end_of_role|>" + sys +
                                                "<|end_of_text|>\n"
                                                "<|start_of_role|>user<|end_of_role|>" +
                                                user +
                                                "<|end_of_text|>\n"
                                                "<|start_of_role|>assistant<|end_of_role|>");
}

TEST_CASE("granite 4.x multi-turn keeps the system turn and closes history") {
    const ChatFormat f = chat_format_for("granite-4.2-3b"); // #521 reuses this branch
    const std::string p = f.render_prompt("Be concise.", {{"hi", "hello"}}, "how are you?");
    CHECK(p == "<|start_of_role|>system<|end_of_role|>Be concise.<|end_of_text|>\n"
               "<|start_of_role|>user<|end_of_role|>hi<|end_of_text|>\n"
               "<|start_of_role|>assistant<|end_of_role|>hello<|end_of_text|>\n"
               "<|start_of_role|>user<|end_of_role|>how are you?<|end_of_text|>\n"
               "<|start_of_role|>assistant<|end_of_role|>");
    CHECK(p.find("<|im_start|>") == std::string::npos);
    CHECK(p.find("<|eot_id|>") == std::string::npos);
}

TEST_CASE("granite 4.x system prompt is a dedicated turn, never merged") {
    const ChatFormat f = chat_format_for("granite40-h-tiny");
    const std::string p = f.render_prompt("SYS", {}, "ciao");
    CHECK(p == "<|start_of_role|>system<|end_of_role|>SYS<|end_of_text|>\n"
               "<|start_of_role|>user<|end_of_role|>ciao<|end_of_text|>\n"
               "<|start_of_role|>assistant<|end_of_role|>");
    CHECK(p.find("SYS\n\nciao") == std::string::npos);
    CHECK(f.render_system_prefix("SYS") ==
          "<|start_of_role|>system<|end_of_role|>SYS<|end_of_text|>\n");
}

TEST_CASE("granite 4.x empty system still emits an empty dedicated turn") {
    // IBM's granite-4.x jinja inserts a default system message when the caller
    // omits one. This renderer emits the empty turn instead. CLI, GUI, and the
    // LAN API always pass kDefaultSystemPrompt (or the coding default) today.
    const ChatFormat f = chat_format_for("granite40-h-tiny");
    CHECK(f.render_prompt("", {}, "ciao") ==
          "<|start_of_role|>system<|end_of_role|><|end_of_text|>\n"
          "<|start_of_role|>user<|end_of_role|>ciao<|end_of_text|>\n"
          "<|start_of_role|>assistant<|end_of_role|>");
    CHECK(f.render_system_prefix("") == "<|start_of_role|>system<|end_of_role|><|end_of_text|>\n");
}

TEST_CASE("smollm3 chat golden and history prefill") {
    const ChatFormat f = chat_format_for("smollm3-3b");
    const std::string header = "<|im_start|>system\n## Metadata\n\n"
                               "Knowledge Cutoff Date: June 2025\n"
                               "Today Date: 15 June 2025\nReasoning Mode: /no_think\n\n"
                               "## Custom Instructions\n\n";
    const std::string end = "\n\n<|im_end|>\n";
    const std::string gen = "<|im_start|>assistant\n<think>\n\n</think>\n";
    CHECK(f.render_prompt("", {}, "hi") ==
          header + "You are a helpful AI assistant named SmolLM, trained by Hugging Face." + end +
              "<|im_start|>user\nhi<|im_end|>\n" + gen);
    CHECK(f.render_prompt("Be concise.", {}, "hi") ==
          header + "Be concise." + end + "<|im_start|>user\nhi<|im_end|>\n" + gen);
    CHECK(f.render_prompt("Be concise.", {{"hi", "hello"}}, "how are you?") ==
          header + "Be concise." + end + "<|im_start|>user\nhi<|im_end|>\n" + gen +
              "hello<|im_end|>\n<|im_start|>user\nhow are you?<|im_end|>\n" + gen);
    CHECK(f.gen_suffix == "<think>\n\n</think>\n");
    CHECK(f.gen_suffix != qwen_no_think_gen_suffix("qwen35-0.8b"));
    CHECK_FALSE(f.strip_thinking_content);
    CHECK(f.stop_sequences == std::vector<std::string>{"<|im_end|>"});
    CHECK(f.bos.empty());
    CHECK(f.render_system_prefix("sys").find("Today Date: 15 June 2025") != std::string::npos);
    for (const char* sys :
         {"Be concise. /think", "Be concise. /no_think", "Be concise. /system_override"}) {
        CHECK(f.render_prompt(sys, {}, "hi") ==
              header + "Be concise." + end + "<|im_start|>user\nhi<|im_end|>\n" + gen);
    }
    CHECK(f.render_prompt("/think Be concise.", {}, "hi") ==
          header + " Be concise." + end + "<|im_start|>user\nhi<|im_end|>\n" + gen);
    CHECK(f.render_system_prefix("/think") == f.render_system_prefix(""));
}

TEST_CASE("smollm3 kv-reuse includes history prefill") {
    const ChatFormat f = chat_format_for("smollm3-3b");
    const std::vector<ChatTurn> history = {{"u1", "a1"}};
    const std::string base = f.render_prompt("sys", history, "u2");
    const std::string cold = f.render_prompt("sys", {{"u1", "a1"}, {"u2", "a2"}}, "u3");
    CHECK(base + "a2" + f.stop_sequences.front() + f.render_delta("u3", true) == cold);
    CHECK(base + "a2" + f.render_delta("u3", false) == cold);
}

TEST_CASE("smollm3 detection and existing chat formats") {
    for (const char* id : {"smollm3-3b", "SmolLM3-3B-Q4_K_M.gguf", "smol-lm3", "smol_lm3"}) {
        CHECK(model_is_smollm3(id));
        CHECK(chat_format_for(id).kind == ChatFormatKind::SmolLM3);
    }
    CHECK_FALSE(model_is_smollm3("smollm2"));
    CHECK_FALSE(model_is_smollm3("smollm2-360m-cpu-int4"));
    const ChatFormat old = chat_format_for("smollm2-360m-cpu-int4");
    CHECK(old.kind == ChatFormatKind::ChatML);
    CHECK(old.gen_suffix.empty());
    CHECK_FALSE(old.strip_thinking_content);
    CHECK(chat_format_for("llama32-3b").kind == ChatFormatKind::Llama3);
    CHECK(chat_format_for("phi35-mini").kind == ChatFormatKind::Phi3);
    CHECK(chat_format_for("granite40-h-tiny").kind == ChatFormatKind::Granite);
    CHECK(chat_format_for("qwen25-coder-1.5b").kind == ChatFormatKind::ChatML);
}

TEST_CASE("olmoe detection and chat template") {
    for (const char* id :
         {"olmoe-1b-7b", "OLMoE-1B-7B-0125-Instruct-Q3_K_S.gguf", "olmo-e-1b", "olmo_e-1b",
          "models/olmoe-1b-7b", "models/OLMoE-1B-7B-0125-Instruct-Q3_K_S.gguf", "models/olmo-e-1b",
          "models/olmo_e-1b"}) {
        CHECK(model_is_olmoe(id));
    }
    for (const char* id :
         {"olmo-2-1124-7b", "olmo2", "olmo", "lfm25-350m", "models/olmoe-cache/lfm25-350m.gguf"}) {
        CHECK_FALSE(model_is_olmoe(id));
    }
    const ChatFormat f = chat_format_for("OLMoE-1B-7B-0125-Instruct-Q3_K_S.gguf");
    CHECK(f.kind == ChatFormatKind::Olmoe);
    CHECK(chat_format_for("olmoe-1b-7b-llamacpp.gguf").kind == ChatFormatKind::Olmoe);
    CHECK(chat_format_for("olmoe-1b-7b-phi.gguf").kind == ChatFormatKind::Olmoe);
    CHECK(f.stop_sequences == std::vector<std::string>{"<|endoftext|>"});
    CHECK(f.gen_suffix.empty());
    CHECK(f.render_prompt("Be concise.", {}, "Hello") ==
          "<|endoftext|><|system|>\nBe concise.\n<|user|>\nHello\n<|assistant|>\n");
    CHECK(f.render_prompt("Be concise.", {{"hi", "hello"}}, "how are you?") ==
          "<|endoftext|><|system|>\nBe concise.\n<|user|>\nhi\n"
          "<|assistant|>\nhello<|endoftext|>\n<|user|>\nhow are you?\n<|assistant|>\n");
    // The Jinja without a system message differs by design, as with Granite.
    CHECK(f.render_prompt("", {}, "How are you doing?") ==
          "<|endoftext|><|system|>\n\n<|user|>\nHow are you doing?\n<|assistant|>\n");
    CHECK(f.render_system_prefix("Be concise.") == "<|endoftext|><|system|>\nBe concise.\n");
    CHECK(f.render_delta("X", true) == "\n<|user|>\nX\n<|assistant|>\n");
    CHECK(f.render_delta("X", false) == "<|endoftext|>\n<|user|>\nX\n<|assistant|>\n");
}
