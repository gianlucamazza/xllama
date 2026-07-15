// Copyright (c) 2024 Venere Labs
// SPDX-License-Identifier: MIT

#include <doctest/doctest.h>

#include "xllama/model_provision.h"

#include <string>
#include <vector>

using namespace xllama;

using WS = std::vector<std::wstring>;

TEST_CASE("provision: gguf current quant present -> true") {
    WS present{L"gemma-4-E2B-it-Q3_K_S.gguf"};
    WS expected{L"gemma-4-E2B-it-Q3_K_S.gguf"};
    CHECK(dir_satisfies_expected_files(present, expected));
}

TEST_CASE("provision: stale quant present, current expected -> false (the bug)") {
    // The reported on-console case: dir holds the old IQ2_M, manifest wants Q3_K_S.
    WS present{L"gemma-4-E2B-it-UD-IQ2_M.gguf"};
    WS expected{L"gemma-4-E2B-it-Q3_K_S.gguf"};
    CHECK_FALSE(dir_satisfies_expected_files(present, expected));
}

TEST_CASE("provision: old + new both present, expects new -> true (extra ignored)") {
    WS present{L"gemma-4-E2B-it-UD-IQ2_M.gguf", L"gemma-4-E2B-it-Q3_K_S.gguf"};
    WS expected{L"gemma-4-E2B-it-Q3_K_S.gguf"};
    CHECK(dir_satisfies_expected_files(present, expected));
}

TEST_CASE("provision: ort-genai full set present -> true") {
    WS present{L"genai_config.json", L"tokenizer.json", L"tokenizer_config.json", L"model.onnx"};
    WS expected{L"genai_config.json", L"tokenizer.json", L"tokenizer_config.json", L"model.onnx"};
    CHECK(dir_satisfies_expected_files(present, expected));
}

TEST_CASE("provision: ort-genai missing model.onnx -> false") {
    WS present{L"genai_config.json", L"tokenizer.json", L"tokenizer_config.json"};
    WS expected{L"genai_config.json", L"tokenizer.json", L"tokenizer_config.json", L"model.onnx"};
    CHECK_FALSE(dir_satisfies_expected_files(present, expected));
}

TEST_CASE("provision: diffusion multi-subdir set present -> true") {
    WS present{L"text_encoder/model.onnx", L"unet/model.onnx", L"vae_decoder/model.onnx"};
    WS expected{L"text_encoder/model.onnx", L"unet/model.onnx", L"vae_decoder/model.onnx"};
    CHECK(dir_satisfies_expected_files(present, expected));
}

TEST_CASE("provision: native backslash present vs forward-slash expected -> true") {
    WS present{L"unet\\model.onnx", L"vae_decoder\\model.onnx"};
    WS expected{L"unet/model.onnx", L"vae_decoder/model.onnx"};
    CHECK(dir_satisfies_expected_files(present, expected));
}

TEST_CASE("provision: case-insensitive filename match -> true") {
    WS present{L"model.onnx"};
    WS expected{L"Model.ONNX"};
    CHECK(dir_satisfies_expected_files(present, expected));
}

TEST_CASE("provision: empty expected -> false (caller does loose fallback)") {
    WS present{L"anything.gguf"};
    CHECK_FALSE(dir_satisfies_expected_files(present, WS{}));
}

TEST_CASE("provision: empty present, non-empty expected -> false") {
    WS expected{L"model.gguf"};
    CHECK_FALSE(dir_satisfies_expected_files(WS{}, expected));
}

TEST_CASE("provision: present is a strict superset of expected -> true") {
    WS present{L"model.gguf", L"README.md", L".complete", L"extra.bin"};
    WS expected{L"model.gguf"};
    CHECK(dir_satisfies_expected_files(present, expected));
}

TEST_CASE("provision: diffusion with one required subdir file missing -> false") {
    WS present{L"text_encoder/model.onnx", L"unet/model.onnx"};
    WS expected{L"text_encoder/model.onnx", L"unet/model.onnx", L"vae_decoder/model.onnx"};
    CHECK_FALSE(dir_satisfies_expected_files(present, expected));
}

TEST_CASE("provision: normalize_model_path lowercases and unifies separators") {
    CHECK(normalize_model_path(L"UNet\\Model.ONNX") == L"unet/model.onnx");
}
