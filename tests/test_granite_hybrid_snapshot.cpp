// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// #525 / #512: hybrid prefix snapshots on the Granite-4.0-H-Tiny Mamba2
// checkpoint. CI does not fetch the ~3.1 GB official Q3_K_S GGUF; skip with
// a reason when XLLAMA_GRANITE_H_TINY is unset. Host receipt:
// bench/results/2026-11-24-granite40-h-tiny/

#include "xllama/chat_prompt.h"
#include "xllama/prefix_snapshot.h"
#include "xllama/session.h"

#include <cstdlib>
#include <doctest/doctest.h>
#include <string>

using namespace xllama;

namespace {

const char* granite_h_tiny_path() {
    return std::getenv("XLLAMA_GRANITE_H_TINY");
}

GenerateParams greedy(const std::string& prompt, const ChatFormat& fmt, int n_predict = 16) {
    GenerateParams gp;
    gp.prompt = prompt;
    gp.n_predict = n_predict;
    gp.temperature = 0.0f;
    gp.top_p = 1.0f;
    gp.top_k = 1;
    gp.stop_sequences = fmt.stop_sequences;
    return gp;
}

std::unique_ptr<Session> load_session(const char* model, std::string* error) {
    SessionParams sp;
    sp.model_path = model;
    sp.n_ctx = 512;
    sp.n_threads = 4;
    sp.n_gpu_layers = 0; // CPU only (#421); no MUL_MAT_ID / SSM offload
    sp.prefix_snapshots = true;
    return Session::create(sp, error);
}

} // namespace

TEST_CASE("Session: granitehybrid prefix snapshot restore (#512/#525, "
          "opt-in: XLLAMA_GRANITE_H_TINY)") {
    const char* model = granite_h_tiny_path();
    if (!model) {
        MESSAGE("SKIP: XLLAMA_GRANITE_H_TINY is unset — the official "
                "ibm-granite/granite-4.0-h-tiny-GGUF Q3_K_S file is not fetched "
                "in CI (~3.1 GB). Run locally and keep the receipt under "
                "bench/results/2026-11-24-granite40-h-tiny/.");
        return;
    }

    const ChatFormat fmt = chat_format_for(model);
    REQUIRE(fmt.kind == ChatFormatKind::Granite);
    const std::string sys = "You are a helpful assistant.";
    const std::string p1 = fmt.render_prompt(sys, {}, "Name a colour.");
    // Diverges inside the system turn, before any slacked head of p1 ends, so
    // the saved snapshot is not a prefix of it whatever the slack.
    const std::string p_other = fmt.render_prompt("You are a terse assistant.", {}, "Say hi.");
    // Strict extension of the saved prefix (p1) that is not an extension of a
    // generated tail. Same suffix covers KeepResident (prefill-only pin) and
    // RestoreSnapshot (after a decode that left a colour-name tail).
    const std::string p_ext = p1 +
                              "NOT_A_COLOUR_TAIL<|end_of_text|>\n"
                              "<|start_of_role|>user<|end_of_role|>And another.<|end_of_text|>\n"
                              "<|start_of_role|>assistant<|end_of_role|>";
    REQUIRE(p_ext.compare(0, p1.size(), p1) == 0);
    REQUIRE(p_ext.size() > p1.size());

    std::string error;

    // Pure extension / KeepResident: pin p1 with n_predict=0 so resident KV is
    // still the post-prefill snapshot, then extend it. A decode tail would
    // compare SSM decode-then-prefill with a cold prefill of the same tokens.
    auto keep = load_session(model, &error);
    REQUIRE_MESSAGE(keep, error);
    const auto r_pin = keep->generate(greedy(p1, fmt, 0));
    REQUIRE_MESSAGE(r_pin.success, r_pin.error_msg);
    REQUIRE(r_pin.n_p_eval > 0);
    const auto r_ext = keep->generate(greedy(p_ext, fmt));
    REQUIRE_MESSAGE(r_ext.success, r_ext.error_msg);
    CHECK(r_ext.n_p_eval < r_ext.n_prompt);
    CHECK(r_ext.n_prompt == keep->count_tokens(p_ext));
    CHECK_FALSE(r_ext.output_text.empty());
    keep.reset();

    auto cold_ext = load_session(model, &error);
    REQUIRE_MESSAGE(cold_ext, error);
    const auto c_ext = cold_ext->generate(greedy(p_ext, fmt));
    REQUIRE_MESSAGE(c_ext.success, c_ext.error_msg);
    CHECK(r_ext.output_text == c_ext.output_text);
    cold_ext.reset();

    // Non-prefix drop: unrelated prompt is not a prefix of the saved snapshot.
    auto warm_drop = load_session(model, &error);
    REQUIRE_MESSAGE(warm_drop, error);
    const auto r1 = warm_drop->generate(greedy(p1, fmt));
    REQUIRE_MESSAGE(r1.success, r1.error_msg);
    REQUIRE(r1.n_p_eval > 0);
    REQUIRE_FALSE(r1.output_text.empty());
    const auto r_drop = warm_drop->generate(greedy(p_other, fmt));
    REQUIRE_MESSAGE(r_drop.success, r_drop.error_msg);
    CHECK(r_drop.n_p_eval == r_drop.n_prompt);
    CHECK_FALSE(r_drop.output_text.empty());
    warm_drop.reset();

    auto cold_drop = load_session(model, &error);
    REQUIRE_MESSAGE(cold_drop, error);
    const auto c_drop = cold_drop->generate(greedy(p_other, fmt));
    REQUIRE_MESSAGE(c_drop.success, c_drop.error_msg);
    CHECK(r_drop.output_text == c_drop.output_text);
    cold_drop.reset();

    // Strict-prefix restore: new prompt extends p1, not the generated tail.
    auto warm_restore = load_session(model, &error);
    REQUIRE_MESSAGE(warm_restore, error);
    const auto r1b = warm_restore->generate(greedy(p1, fmt));
    REQUIRE_MESSAGE(r1b.success, r1b.error_msg);
    const auto r_restore = warm_restore->generate(greedy(p_ext, fmt));
    REQUIRE_MESSAGE(r_restore.success, r_restore.error_msg);
    CHECK(r_restore.n_p_eval < r_restore.n_prompt);
    CHECK(r_restore.n_p_eval >= 1);
    CHECK_FALSE(r_restore.output_text.empty());
    warm_restore.reset();

    auto cold_restore = load_session(model, &error);
    REQUIRE_MESSAGE(cold_restore, error);
    const auto c_restore = cold_restore->generate(greedy(p_ext, fmt));
    REQUIRE_MESSAGE(c_restore.success, c_restore.error_msg);
    CHECK(r_restore.output_text == c_restore.output_text);
}

TEST_CASE("Session: granitehybrid prefix snapshot slack and exact regenerate "
          "(opt-in: XLLAMA_GRANITE_H_TINY)") {
    const char* model = granite_h_tiny_path();
    if (!model) {
        MESSAGE("SKIP: XLLAMA_GRANITE_H_TINY is unset — slack/regenerate is "
                "not exercised in CI.");
        return;
    }

    const ChatFormat fmt = chat_format_for(model);
    REQUIRE(fmt.kind == ChatFormatKind::Granite);
    const std::string sys = "You are a helpful assistant.";
    const std::string stem = "Please reply with a single English word. "
                             "The category you must pick from is";
    const std::string p1 = fmt.render_prompt(sys, {}, stem + " colour.");
    const std::string p2 = fmt.render_prompt(sys, {}, stem + " fruit.");
    REQUIRE(p1 != p2);

    std::string error;
    auto sess = load_session(model, &error);
    REQUIRE_MESSAGE(sess, error);
    REQUIRE(sess->count_tokens(p1) > static_cast<int>(kPrefixSnapshotTailSlack));

    const auto r1 = sess->generate(greedy(p1, fmt));
    REQUIRE_MESSAGE(r1.success, r1.error_msg);
    REQUIRE(r1.n_p_eval > 0);

    const auto r_regen = sess->generate(greedy(p1, fmt));
    REQUIRE_MESSAGE(r_regen.success, r_regen.error_msg);
    CHECK(r_regen.n_p_eval == r_regen.n_prompt);
    CHECK(r_regen.n_prompt == r1.n_prompt);
    CHECK(r_regen.output_text == r1.output_text);

    auto cold1 = load_session(model, &error);
    REQUIRE_MESSAGE(cold1, error);
    const auto c1 = cold1->generate(greedy(p1, fmt));
    REQUIRE_MESSAGE(c1.success, c1.error_msg);
    CHECK(r_regen.output_text == c1.output_text);
    CHECK(r_regen.n_p_eval == c1.n_p_eval);
    cold1.reset();

    const auto r_div = sess->generate(greedy(p2, fmt));
    REQUIRE_MESSAGE(r_div.success, r_div.error_msg);
    CHECK(r_div.n_p_eval < r_div.n_prompt);
    CHECK(r_div.n_p_eval >= 1);
    CHECK_FALSE(r_div.output_text.empty());
    sess.reset();

    auto cold2 = load_session(model, &error);
    REQUIRE_MESSAGE(cold2, error);
    const auto c2 = cold2->generate(greedy(p2, fmt));
    REQUIRE_MESSAGE(c2.success, c2.error_msg);
    CHECK(r_div.output_text == c2.output_text);
}

// #527: granitehybrid logs can_shift=1. Overflow must take the #169 path
// (success, not "context full") because llama.cpp reports memory_can_shift
// from the attention half. This does *not* prove cold vs post-shift
// continuation parity — that comparison is a follow-up, not a default change.
TEST_CASE("Session: granitehybrid context shift on overflow (#527, "
          "opt-in: XLLAMA_GRANITE_H_TINY)") {
    const char* model = granite_h_tiny_path();
    if (!model) {
        MESSAGE("SKIP: XLLAMA_GRANITE_H_TINY is unset — overflow shift is "
                "not exercised in CI.");
        return;
    }

    const ChatFormat fmt = chat_format_for(model);
    REQUIRE(fmt.kind == ChatFormatKind::Granite);

    SessionParams sp;
    sp.model_path = model;
    sp.n_ctx = 96;
    sp.n_threads = 4;
    sp.n_gpu_layers = 0;
    sp.prefix_snapshots = true;
    std::string error;
    auto sess = Session::create(sp, &error);
    REQUIRE_MESSAGE(sess, error);

    const std::string sys = "You are a helpful assistant.";
    const std::string seed_prompt = fmt.render_prompt(sys, {}, "Name a colour.");
    const int n_keep = sess->count_tokens(fmt.render_system_prefix(sys));
    REQUIRE(n_keep > 0);

    GenerateParams seed = greedy(seed_prompt, fmt, 8);
    seed.reuse_kv = true;
    seed.reset_kv = true;
    seed.n_keep = n_keep;
    const auto r_seed = sess->generate(seed);
    REQUIRE_MESSAGE(r_seed.success, r_seed.error_msg);
    REQUIRE(sess->can_context_shift());

    int total_tokens = sess->count_tokens(seed_prompt) + r_seed.n_eval;
    bool saw_shift_success = false;
    for (int i = 0; i < 12; ++i) {
        GenerateParams cont =
            greedy(fmt.render_delta("And another colour, please.", /*stop=*/true), fmt, 8);
        cont.reuse_kv = true;
        cont.reset_kv = false;
        cont.n_keep = n_keep;
        const auto rc = sess->generate(cont);
        REQUIRE_MESSAGE(rc.success, rc.error_msg);
        total_tokens += rc.n_p_eval + rc.n_eval;
        if (total_tokens > sp.n_ctx)
            saw_shift_success = true;
    }
    CHECK(saw_shift_success);
}
