// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include <doctest/doctest.h>

#include "xllama/model_file_plan.h"

using namespace xllama;

using Pins = std::vector<ManifestFilePin>;
using Have = std::vector<PresentFile>;

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
