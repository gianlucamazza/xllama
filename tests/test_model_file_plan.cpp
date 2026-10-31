// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include <doctest/doctest.h>

#include "xllama/model_file_plan.h"
#include "xllama/routing_policy.h"

using namespace xllama;

using Pins = std::vector<ManifestFilePin>;
using Have = std::vector<PresentFile>;

TEST_CASE("pin_check: hash-unavailable retry defaults are documented") {
    CHECK(kHashVerifyRetries == 2);
    CHECK(kHashVerifyRetryPauseS == 1);
}

TEST_CASE("sidecar: licence and notice names, with or without a text extension") {
    CHECK(is_catalogue_sidecar(L"LICENSE"));
    CHECK(is_catalogue_sidecar(L"license.txt"));
    CHECK(is_catalogue_sidecar(L"NOTICE"));
    CHECK(is_catalogue_sidecar(L"Notice.MD"));
    CHECK(is_catalogue_sidecar(L"README"));
    CHECK(is_catalogue_sidecar(L"docs/COPYING"));
    CHECK_FALSE(is_catalogue_sidecar(L"model.onnx"));
    CHECK_FALSE(is_catalogue_sidecar(L"LFM2.5-350M-QAD-Q4_0.gguf"));
    CHECK_FALSE(is_catalogue_sidecar(L"genai_config.json"));
    CHECK_FALSE(is_catalogue_sidecar(L"tokenizer.json"));
}

TEST_CASE("sidecar: files[].role overrides the documented filename default") {
    CHECK(parse_catalogue_file_role("") == CatalogueFileRole::Unspecified);
    CHECK(parse_catalogue_file_role("weight") == CatalogueFileRole::Weight);
    CHECK(parse_catalogue_file_role("DATA") == CatalogueFileRole::Weight);
    CHECK(parse_catalogue_file_role("licence") == CatalogueFileRole::Licence);
    CHECK(parse_catalogue_file_role("License") == CatalogueFileRole::Licence);
    CHECK(parse_catalogue_file_role("notice") == CatalogueFileRole::Notice);
    CHECK(parse_catalogue_file_role("sidecar") == CatalogueFileRole::Sidecar);
    CHECK(parse_catalogue_file_role("unknown") == CatalogueFileRole::Weight);

    ManifestFilePin terms = make_file_pin(L"TERMS", 80, "abc", "licence");
    CHECK(is_catalogue_sidecar(terms));
    CHECK_FALSE(is_weight_pin(terms));
    CHECK_FALSE(is_catalogue_sidecar(L"TERMS")); // default would treat TERMS as a weight

    ManifestFilePin named_license = make_file_pin(L"LICENSE", 80, "abc", "weight");
    CHECK_FALSE(is_catalogue_sidecar(named_license));
    CHECK(is_weight_pin(named_license));

    ManifestFilePin generic = make_file_pin(L"COPYRIGHT.notes", 12, "ee", "sidecar");
    CHECK(generic.role == CatalogueFileRole::Sidecar);
    CHECK(is_catalogue_sidecar(generic));
}

TEST_CASE("file_pin_satisfied: size match is enough when the listing has no hash") {
    PresentFile have{L"model.onnx", 725021421, {}};
    ManifestFilePin pin{L"model.onnx", 725021421,
                        "c417dda34ebf9c0c4ab9f0057c0841121581296789f8fa156f2899195434133f"};
    CHECK(file_pin_satisfied(have, pin));
    have.size = 1;
    CHECK_FALSE(file_pin_satisfied(have, pin));
    have.size = 0;
    CHECK_FALSE(file_pin_satisfied(have, pin));
}

TEST_CASE("file_pin_satisfied: provided hash mismatch still rejects") {
    PresentFile have{L"model.onnx", 100, "aa"};
    ManifestFilePin pin{L"model.onnx", 100, "bb"};
    CHECK_FALSE(file_pin_satisfied(have, pin));
    have.sha256 = "BB";
    pin.sha256 = "bb";
    CHECK(file_pin_satisfied(have, pin));
}

TEST_CASE("plan: old install plus new LICENSE/NOTICE fetches sidecars only") {
    // #370: a pre-licence tree whose weights still match their pins must not
    // re-download those weights when the catalogue grows sidecar rows.
    Have present{
        {L"genai_config.json", 1599, {}},
        {L"tokenizer.json", 3522871, {}},
        {L"tokenizer_config.json", 452, {}},
        {L"model.onnx", 725021421, {}},
    };
    Pins expected{
        {L"LICENSE", 11357, "abc"},
        {L"NOTICE", 232, "def"},
        {L"genai_config.json", 1599, "cfg"},
        {L"tokenizer.json", 3522871, "tok"},
        {L"tokenizer_config.json", 452, "tc"},
        {L"model.onnx", 725021421, "onnx"},
    };
    const auto fetch = files_needing_fetch(present, expected);
    REQUIRE(fetch.size() == 2);
    CHECK(fetch[0] == L"LICENSE");
    CHECK(fetch[1] == L"NOTICE");
    CHECK(weight_files_needing_fetch(present, expected).empty());
    CHECK(weight_pins_verified(present, expected));
    CHECK(only_sidecars_missing(present, expected));
}

TEST_CASE("plan: #370 (a) old install plus role-tagged sidecars leaves weights alone") {
    // Generic names only — the rule is the pin's role, not a catalogue id.
    Have present{
        {L"weights.bin", 1000, "aa"},
        {L"tokenizer.json", 50, "bb"},
    };
    Pins expected{
        make_file_pin(L"TERMS", 20, "cc", "licence"),
        make_file_pin(L"ATTRIBUTION", 8, "dd", "notice"),
        make_file_pin(L"weights.bin", 1000, "aa", "weight"),
        make_file_pin(L"tokenizer.json", 50, "bb", "data"),
    };
    CHECK(files_needing_fetch(present, expected) ==
          std::vector<std::wstring>({L"TERMS", L"ATTRIBUTION"}));
    CHECK(weight_files_needing_fetch(present, expected).empty());
    CHECK(weight_pins_verified(present, expected));
    CHECK(only_sidecars_missing(present, expected));
    CHECK(gpu_tree_ready_for_routing(weight_pins_verified(present, expected), true));
    CHECK_FALSE(gpu_tree_ready_for_routing(weight_pins_verified(present, expected), false));
}

TEST_CASE("plan: #370 (b) changed weight hash is caught at download/promotion") {
    // Both sides have a hash: this is the DownloadAsync / promotion path.
    // A size-only listing (empty sha, as routing and EnsureModel use) would
    // still classify Ok — routing does not see this mismatch.
    Have present{
        {L"weights.bin", 1000, "oldhash"},
        {L"TERMS", 20, "cc"},
    };
    Pins expected{
        make_file_pin(L"TERMS", 20, "cc", "licence"),
        make_file_pin(L"weights.bin", 1000, "newhash", "weight"),
    };
    CHECK(classify_file_pin(present[0], expected[1]) == PinCheck::Refetch);
    CHECK(weight_files_needing_fetch(present, expected) ==
          std::vector<std::wstring>{L"weights.bin"});
    CHECK_FALSE(weight_pins_verified(present, expected));
    CHECK_FALSE(only_sidecars_missing(present, expected));

    PresentFile size_only{L"weights.bin", 1000, {}};
    CHECK(classify_file_pin(size_only, expected[1]) == PinCheck::Ok);
}

TEST_CASE("plan: GGUF weights plus new licence sidecar leaves the quant untouched") {
    Have present{{L"LFM2.5-350M-QAD-Q4_0.gguf", 248512512, {}}};
    Pins expected{
        {L"LFM2.5-350M-QAD-Q4_0.gguf", 248512512, "ggufhash"},
        {L"LICENSE", 10574, "lic"},
    };
    CHECK(files_needing_fetch(present, expected) == std::vector<std::wstring>{L"LICENSE"});
    CHECK(weight_files_needing_fetch(present, expected).empty());
}

TEST_CASE("plan: missing or size-mismatched weight is fetched; matching siblings are not") {
    Have present{{L"model.onnx", 10, {}}, {L"LICENSE", 100, {}}};
    Pins expected{
        {L"LICENSE", 100, {}},
        {L"NOTICE", 20, {}},
        {L"model.onnx", 99, {}},
    };
    const auto fetch = files_needing_fetch(present, expected);
    REQUIRE(fetch.size() == 2);
    CHECK(fetch[0] == L"NOTICE");
    CHECK(fetch[1] == L"model.onnx");
    CHECK(weight_files_needing_fetch(present, expected) ==
          std::vector<std::wstring>{L"model.onnx"});
}

TEST_CASE("plan: stale quant is still a weight fetch") {
    Have present{{L"old.gguf", 100, {}}};
    Pins expected{{L"new.gguf", 200, "x"}, {L"LICENSE", 10, "y"}};
    const auto weights = weight_files_needing_fetch(present, expected);
    REQUIRE(weights.size() == 1);
    CHECK(weights[0] == L"new.gguf");
}

TEST_CASE("plan: empty expected fetches nothing") {
    Have present{{L"model.onnx", 1, {}}};
    CHECK(files_needing_fetch(present, {}).empty());
}

TEST_CASE("plan: separator and case on the filename do not force a refetch") {
    Have present{{L"unet\\model.onnx", 50, {}}};
    Pins expected{{L"unet/model.onnx", 50, "deadbeef"}};
    CHECK(files_needing_fetch(present, expected).empty());
    CHECK(weight_files_needing_fetch(present, expected).empty());
}

TEST_CASE("pin_check: hash ok") {
    PresentFile have{L"model.onnx", 100, "aabb"};
    ManifestFilePin pin{L"model.onnx", 100, "AABB"};
    CHECK(classify_file_pin(have, pin) == PinCheck::Ok);
    CHECK(classify_file_pin(&have, pin) == PinCheck::Ok);
}

TEST_CASE("pin_check: hash mismatch") {
    PresentFile have{L"model.onnx", 100, "aa"};
    ManifestFilePin pin{L"model.onnx", 100, "bb"};
    CHECK(classify_file_pin(have, pin) == PinCheck::Refetch);
    CHECK(files_needing_fetch({have}, {pin}) == std::vector<std::wstring>{L"model.onnx"});
}

TEST_CASE("pin_check: hash unreadable") {
    PresentFile have{L"model.onnx", 100, {}, true};
    ManifestFilePin pin{L"model.onnx", 100, "deadbeef"};
    CHECK(classify_file_pin(have, pin) == PinCheck::Unverified);
    CHECK(files_needing_fetch({have}, {pin}).empty()); // keep, do not refetch
    CHECK_FALSE(file_pin_satisfied(have, pin));
}

TEST_CASE("pin_check: size mismatch or truncated") {
    ManifestFilePin pin{L"model.onnx", 100, "aa"};
    PresentFile truncated{L"model.onnx", 40, "aa"};
    CHECK(classify_file_pin(truncated, pin) == PinCheck::Refetch);
    PresentFile missing{L"model.onnx", 0, {}};
    CHECK(classify_file_pin(missing, pin) == PinCheck::Refetch);
    CHECK(classify_file_pin(nullptr, pin) == PinCheck::Refetch);
    PresentFile oversized{L"model.onnx", 200, {}};
    CHECK(classify_file_pin(oversized, pin) == PinCheck::Refetch);
}

TEST_CASE("only_sidecars_missing: weights present, licence and notice absent") {
    Have present{
        {L"genai_config.json", 1599, {}},
        {L"tokenizer.json", 3522871, {}},
        {L"model.onnx", 725021421, {}},
    };
    Pins expected{
        {L"LICENSE", 11357, "abc"},          {L"NOTICE", 232, "def"},
        {L"genai_config.json", 1599, "cfg"}, {L"tokenizer.json", 3522871, "tok"},
        {L"model.onnx", 725021421, "onnx"},
    };
    CHECK(only_sidecars_missing(present, expected));
    CHECK(weight_files_needing_fetch(present, expected).empty());
}

TEST_CASE("only_sidecars_missing: a weight gap is not sidecar-only") {
    Have present{{L"LICENSE", 10, {}}};
    Pins expected{{L"LICENSE", 10, {}}, {L"model.onnx", 99, {}}};
    CHECK_FALSE(only_sidecars_missing(present, expected));
}

TEST_CASE("only_sidecars_missing: unverified weight is not ready") {
    Have present{{L"model.onnx", 99, {}, true}, {L"LICENSE", 10, {}}};
    Pins expected{{L"LICENSE", 10, "x"}, {L"NOTICE", 4, "y"}, {L"model.onnx", 99, "z"}};
    CHECK_FALSE(only_sidecars_missing(present, expected));
}

TEST_CASE("only_sidecars_missing: empty expected is false") {
    CHECK_FALSE(only_sidecars_missing({{L"model.onnx", 1, {}}}, {}));
}

TEST_CASE("plan: #370 (c) incomplete GPU weights are not routing-ready") {
    Have present{
        {L"TERMS", 20, "cc"},
        {L"ATTRIBUTION", 8, "dd"},
        {L"tokenizer.json", 50, "bb"},
    };
    Pins expected{
        make_file_pin(L"TERMS", 20, "cc", "licence"),
        make_file_pin(L"ATTRIBUTION", 8, "dd", "notice"),
        make_file_pin(L"tokenizer.json", 50, "bb", "data"),
        make_file_pin(L"weights.bin", 1000, "aa", "weight"),
    };
    CHECK_FALSE(weight_pins_verified(present, expected));
    CHECK_FALSE(gpu_tree_ready_for_routing(weight_pins_verified(present, expected), true));
    CHECK_FALSE(gpu_ready_for_routing(false, true));

    RoutingSettings s;
    s.mode = RoutingMode::Auto;
    s.token_threshold = 1;
    s.cpu_model = "cpu-model";
    s.gpu_model = "smollm2-360m-dml-fp16-v2";
    const bool gpu_ok = gpu_ready_for_routing(
        gpu_tree_ready_for_routing(weight_pins_verified(present, expected), true), false);
    auto d = decide_routing(s, 100, false, gpu_ok);
    CHECK_FALSE(d.use_gpu);
    CHECK(d.active_model == "cpu-model");

    s.mode = RoutingMode::GpuOnly;
    auto gpu_only = decide_routing(s, 10, false, gpu_ok);
    CHECK_FALSE(gpu_only.use_gpu);
    CHECK(gpu_only.active_model == "cpu-model");
}

TEST_CASE("weight_pins_verified: missing licence does not invalidate weights") {
    Have present{{L"weights.bin", 1000, "aa"}};
    Pins expected{
        make_file_pin(L"TERMS", 20, "cc", "licence"),
        make_file_pin(L"weights.bin", 1000, "aa", "weight"),
    };
    CHECK(weight_pins_verified(present, expected));
    CHECK(gpu_tree_ready_for_routing(true, true));
    CHECK_FALSE(weight_pins_verified(present, {}));
}

TEST_CASE("plan: #370 size-only weights without .complete are not routing-ready") {
    // Routing / EnsureModel listings have no hash. Size match alone must not
    // make an unmarked tree GPU-ready (failed write, Unverified pin, mid-write
    // exit, or a mixed tree whose sizes happen to match).
    Have present{{L"weights.bin", 1000, {}}};
    Pins expected{make_file_pin(L"weights.bin", 1000, "aa", "weight")};
    CHECK(weight_pins_verified(present, expected));
    const bool complete_marker = false;
    const bool expected_verified =
        gpu_tree_ready_for_routing(weight_pins_verified(present, expected), complete_marker);
    CHECK_FALSE(expected_verified);
    CHECK_FALSE(gpu_ready_for_routing(expected_verified, false));

    RoutingSettings s;
    s.mode = RoutingMode::Auto;
    s.token_threshold = 1;
    s.cpu_model = "cpu-model";
    s.gpu_model = "smollm2-360m-dml-fp16-v2";
    auto d = decide_routing(s, 100, false, gpu_ready_for_routing(expected_verified, false));
    CHECK_FALSE(d.use_gpu);
    CHECK(d.active_model == "cpu-model");
}

TEST_CASE("gpu_needs_provision: names alone never heal a missing marker") {
    CHECK(gpu_needs_provision(false, false));
    CHECK(gpu_needs_provision(false, true));
    CHECK(gpu_needs_provision(true, false));
    CHECK_FALSE(gpu_needs_provision(true, true));
}

TEST_CASE("skip_provision_early_exits: GPU without marker falls through") {
    CHECK(skip_provision_early_exits(false, false));
    CHECK_FALSE(skip_provision_early_exits(false, true));
    CHECK_FALSE(skip_provision_early_exits(true, false));
    CHECK_FALSE(skip_provision_early_exits(true, true));
}

TEST_CASE("sidecar write: no prior marker hashes every file; wrong hash is not Ok") {
    // #370 B5: a same-size weight with the wrong hash must never publish
    // `.complete` or become GPU-ready. Without a prior marker the background
    // write includes the weight so classify_file_pin sees the hash.
    Pins expected{
        {L"LICENSE", 10, "lic"},
        {L"NOTICE", 4, "not"},
        {L"model.onnx", 100, "goodhash"},
    };
    CHECK(sidecar_background_write_files(expected, true) ==
          std::vector<std::wstring>({L"LICENSE", L"NOTICE"}));
    const auto full = sidecar_background_write_files(expected, false);
    REQUIRE(full.size() == 3);
    CHECK(full[0] == L"LICENSE");
    CHECK(full[2] == L"model.onnx");
    CHECK(sidecar_write_needs_full_hash(false));
    CHECK_FALSE(sidecar_write_needs_full_hash(true));

    PresentFile same_size_wrong{L"model.onnx", 100, "badhash"};
    CHECK(classify_file_pin(same_size_wrong, expected[2]) == PinCheck::Refetch);
    CHECK_FALSE(file_pin_satisfied(same_size_wrong, expected[2]));

    const bool names_present = true;
    const bool had_marker = false;
    CHECK(gpu_needs_provision(names_present, had_marker));
    const bool expected_verified = names_present && had_marker;
    CHECK_FALSE(gpu_ready_for_routing(expected_verified, false));
    CHECK_FALSE(gpu_ready_for_routing(expected_verified, true));
}
