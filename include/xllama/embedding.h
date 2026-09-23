// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace xllama {

struct EmbeddingParams {
    std::string input;
    int dimensions = 0; // 0 = model-native dimension
    bool truncate = true;
};

struct EmbeddingResult {
    bool success = false;
    std::vector<float> embedding;
    int n_tokens = 0;
    std::string error_msg;
};

// llama_pooling_type values (from llama.h, reproduced to avoid header dependency).
enum class PoolingType {
    Unspecified = -1,
    None = 0,
    Mean = 1,
    Cls = 2,
    Last = 3,
    Rank = 4,
};

// Trim a token sequence to fit max_tokens, keeping the tokens that matter for
// the given pooling type. MEAN/CLS pool the full sequence and need the prefix.
// LAST/NONE read the final token and need the suffix. Used by Session::embed to
// truncate over-long inputs before inference.
void trim_tokens_for_pooling(std::vector<int32_t>& tokens, PoolingType pooling, int max_tokens);

// Truncate to the requested dimensions (when nonzero) and L2-normalize.
// Returns false for invalid dimensions, an empty vector, or a non-finite/zero norm.
bool normalize_embedding(std::vector<float>& values, int dimensions, std::string* err = nullptr);

// Encode float32 values as little-endian bytes in standard Base64 format.
std::string embedding_base64(const std::vector<float>& values);

} // namespace xllama
