// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include <doctest/doctest.h>

#include "xllama/moe_streaming.h"

#include <string>
#include <vector>

using namespace xllama;

TEST_CASE("moe_arch_known") {
    for (const char* name : {"lfm2moe", "bailingmoe2", "ernie4_5-moe", "smallthinker", "gpt-oss",
                             "gemma4", "afmoe", "maple"}) {
        const MoeArch* a = moe_arch(name);
        REQUIRE(a != nullptr);
        CHECK(a->arch == name);
    }
    CHECK(moe_arch("llama") == nullptr);
    CHECK(moe_arch("lfm2") == nullptr);
    CHECK(moe_arch("") == nullptr);
    CHECK(moe_arch("bailingmoe2")->has_shared_experts);
    CHECK_FALSE(moe_arch("lfm2moe")->has_shared_experts);
    CHECK(moe_arch("gpt-oss")->has_expert_bias);
    CHECK_FALSE(moe_arch("gemma4")->has_expert_bias);
}

TEST_CASE("moe_routed_role_basic") {
    CHECK(moe_routed_role("lfm2moe", "blk.3.ffn_up_exps.weight") == MoeRole::up_exps);
    CHECK(moe_routed_role("lfm2moe", "blk.3.ffn_gate_exps.weight") == MoeRole::gate_exps);
    CHECK(moe_routed_role("afmoe", "blk.12.ffn_down_exps.weight") == MoeRole::down_exps);
}

TEST_CASE("moe_routed_role_resident") {
    for (const char* t : {"blk.3.ffn_up_shexp.weight", "blk.3.ffn_gate_inp.weight",
                          "blk.0.ffn_up_exps.bias", "blk.0.ffn_gate_exps.scale",
                          "blk.1.ffn_up.weight", "blk.1.exp_probs_b.bias", "token_embd.weight"}) {
        CAPTURE(t);
        CHECK_FALSE(moe_routed_role("gpt-oss", t).has_value());
        CHECK_FALSE(moe_routed_role("bailingmoe2", t).has_value());
    }
}

TEST_CASE("moe_routed_role_gemma4_merged") {
    CHECK(moe_routed_role("gemma4", "blk.2.ffn_gate_up_exps.weight") == MoeRole::gate_up_exps);
    CHECK_FALSE(moe_routed_role("lfm2moe", "blk.2.ffn_gate_up_exps.weight").has_value());
    CHECK_FALSE(moe_routed_role("gemma4", "blk.2.ffn_up.weight").has_value());
}

TEST_CASE("moe_routed_role_malformed") {
    for (const char* t :
         {"ffn_up_exps.weight", "blk.x.ffn_up_exps.weight", "blk..ffn_up_exps.weight",
          "blk.1.ffn_up_exps", "blk.1.ffn_up_exps.weight.extra", ""}) {
        CAPTURE(t);
        CHECK_FALSE(moe_routed_role("lfm2moe", t).has_value());
    }
    CHECK_FALSE(moe_routed_role("llama", "blk.1.ffn_up_exps.weight").has_value());
}

TEST_CASE("moe_unrouted_expert_tensor") {
    // A per-expert matrix the table does not route for this architecture must be
    // detectable, so the loader refuses instead of streaming a partial set.
    CHECK(moe_is_unrouted_expert_tensor("lfm2moe", "blk.2.ffn_gate_up_exps.weight"));
    CHECK_FALSE(moe_is_unrouted_expert_tensor("gemma4", "blk.2.ffn_gate_up_exps.weight"));
    CHECK_FALSE(moe_is_unrouted_expert_tensor("lfm2moe", "blk.2.ffn_up_exps.weight"));
    CHECK_FALSE(moe_is_unrouted_expert_tensor("lfm2moe", "blk.2.ffn_up_shexp.weight"));
    CHECK_FALSE(moe_is_unrouted_expert_tensor("gpt-oss", "blk.2.ffn_up_exps.bias"));
}

namespace {
const std::string kSha(64, 'a');
const std::string kTpl(64, 'b');

MoePin pin24() {
    MoePin p;
    p.id = "lfm25-24b-a2b-q4-k-m";
    p.filename = "LFM2.5-24B-A2B-Q4_K_M.gguf";
    p.bytes = 14526258464ull;
    p.sha256 = kSha;
    p.arch = "lfm2moe";
    p.template_sha256 = kTpl;
    return p;
}

MoeObserved seen24() {
    MoeObserved o;
    o.filename = "LFM2.5-24B-A2B-Q4_K_M.gguf";
    o.bytes = 14526258464ull;
    o.sha256 = kSha;
    o.arch = "lfm2moe";
    o.template_sha256 = kTpl;
    return o;
}
} // namespace

TEST_CASE("moe_pin_admits_exact") {
    const MoeAdmission a = check_moe_checkpoint(pin24(), seen24());
    CHECK(a.ok);
    CHECK(a.reason.empty());
}

TEST_CASE("moe_pin_field_mismatch") {
    struct Case {
        const char* field;
        void (*mutate)(MoeObserved&);
    };
    const Case cases[] = {
        {"filename", [](MoeObserved& o) { o.filename = "other.gguf"; }},
        {"bytes", [](MoeObserved& o) { o.bytes += 1; }},
        {"sha256", [](MoeObserved& o) { o.sha256 = std::string(64, 'c'); }},
        {"arch", [](MoeObserved& o) { o.arch = "bailingmoe2"; }},
        {"template_sha256", [](MoeObserved& o) { o.template_sha256 = std::string(64, 'd'); }},
    };
    for (const auto& c : cases) {
        CAPTURE(c.field);
        MoeObserved o = seen24();
        c.mutate(o);
        const MoeAdmission a = check_moe_checkpoint(pin24(), o);
        CHECK_FALSE(a.ok);
        CHECK(a.reason.find(c.field) != std::string::npos);
    }
}

TEST_CASE("moe_pin_sha_mismatch_names_both_digests") {
    MoeObserved o = seen24();
    o.sha256 = std::string(64, 'c');
    const MoeAdmission a = check_moe_checkpoint(pin24(), o);
    CHECK(a.reason.find(kSha) != std::string::npos);
    CHECK(a.reason.find(o.sha256) != std::string::npos);
}

TEST_CASE("moe_pin_unsupported_arch") {
    MoePin p = pin24();
    p.arch = "llama";
    MoeObserved o = seen24();
    o.arch = "llama";
    const MoeAdmission a = check_moe_checkpoint(p, o);
    CHECK_FALSE(a.ok);
    CHECK(a.reason.find("arch") != std::string::npos);
}

TEST_CASE("moe_pin_from_catalogue") {
    const std::vector<MoeCatalogueFile> files = {
        {"LFM2.5-24B-A2B-Q4_K_M.gguf", kSha, 14526258464ull},
        {"LICENSE", std::string(64, 'e'), 10596},
    };
    const auto p = moe_pin_from_catalogue("lfm25-24b-a2b-q4-k-m", "lfm2moe", kTpl, files, false);
    REQUIRE(p.has_value());
    CHECK(p->id == "lfm25-24b-a2b-q4-k-m");
    CHECK(p->filename == "LFM2.5-24B-A2B-Q4_K_M.gguf");
    CHECK(p->bytes == 14526258464ull);
    CHECK(p->sha256 == kSha);
    CHECK(p->arch == "lfm2moe");
    CHECK(p->template_sha256 == kTpl);
}

TEST_CASE("moe_pin_override_ignored") {
    const std::vector<MoeCatalogueFile> files = {{"m.gguf", kSha, 10}};
    CHECK_FALSE(moe_pin_from_catalogue("x", "lfm2moe", kTpl, files, true).has_value());
}

TEST_CASE("moe_pin_needs_single_gguf") {
    const std::vector<MoeCatalogueFile> two = {{"a.gguf", kSha, 10}, {"b.gguf", kSha, 10}};
    CHECK_FALSE(moe_pin_from_catalogue("x", "lfm2moe", kTpl, two, false).has_value());
    const std::vector<MoeCatalogueFile> none = {{"LICENSE", kSha, 10}};
    CHECK_FALSE(moe_pin_from_catalogue("x", "lfm2moe", kTpl, none, false).has_value());
}

TEST_CASE("moe_pin_rejects_bad_fields") {
    const std::vector<MoeCatalogueFile> files = {{"m.gguf", kSha, 10}};
    CHECK_FALSE(moe_pin_from_catalogue("x", "", kTpl, files, false).has_value());
    CHECK_FALSE(moe_pin_from_catalogue("x", "lfm2moe", "", files, false).has_value());
    CHECK_FALSE(
        moe_pin_from_catalogue("x", "lfm2moe", std::string(64, 'B'), files, false).has_value());
    CHECK_FALSE(
        moe_pin_from_catalogue("x", "lfm2moe", std::string(63, 'b'), files, false).has_value());
    CHECK_FALSE(moe_pin_from_catalogue("", "lfm2moe", kTpl, files, false).has_value());
    const std::vector<MoeCatalogueFile> bad_sha = {{"m.gguf", "xyz", 10}};
    CHECK_FALSE(moe_pin_from_catalogue("x", "lfm2moe", kTpl, bad_sha, false).has_value());
}
