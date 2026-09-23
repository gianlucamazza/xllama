// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
#include "xllama/embedding.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace xllama {

void trim_tokens_for_pooling(std::vector<int32_t>& tokens, PoolingType pooling, int max_tokens) {
    if (static_cast<int>(tokens.size()) <= max_tokens)
        return;

    // MEAN and CLS pool the full sequence: the embedding is computed from all
    // token embeddings, so we keep the prefix (the start of the sequence matters).
    // LAST and NONE read the final token: the embedding is the last position,
    // so we keep the suffix (the end of the sequence matters).
    switch (pooling) {
    case PoolingType::Mean:
    case PoolingType::Cls:
        tokens.resize(static_cast<size_t>(max_tokens));
        break;
    case PoolingType::Last:
    case PoolingType::None: {
        const size_t keep = static_cast<size_t>(max_tokens);
        const size_t drop = tokens.size() - keep;
        tokens.erase(tokens.begin(), tokens.begin() + static_cast<std::ptrdiff_t>(drop));
        break;
    }
    case PoolingType::Unspecified:
    case PoolingType::Rank:
        // Unspecified: leave unchanged (caller should fail elsewhere).
        // Rank: reranking models don't use embeddings, caller should reject.
        break;
    }
}

bool normalize_embedding(std::vector<float>& values, int dimensions, std::string* err) {
    const auto fail = [err](const char* message) {
        if (err)
            *err = message;
        return false;
    };
    if (values.empty())
        return fail("model returned an empty embedding");
    if (dimensions < 0 || (dimensions > 0 && static_cast<size_t>(dimensions) > values.size()))
        return fail("requested dimensions exceed the model embedding size");

    const size_t n = dimensions == 0 ? values.size() : static_cast<size_t>(dimensions);
    double norm2 = 0.0;
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(values[i]))
            return fail("model returned a non-finite embedding value");
        norm2 += static_cast<double>(values[i]) * values[i];
    }
    if (!(norm2 > 0.0) || !std::isfinite(norm2))
        return fail("model returned a zero or invalid embedding norm");

    values.resize(n);
    const float inv_norm = static_cast<float>(1.0 / std::sqrt(norm2));
    for (float& value : values)
        value *= inv_norm;
    if (err)
        err->clear();
    return true;
}

std::string embedding_base64(const std::vector<float>& values) {
    static constexpr char alphabet[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve(((values.size() * sizeof(float) + 2) / 3) * 4);
    uint32_t accumulator = 0;
    int bits = -6;
    const auto append_byte = [&](uint8_t byte) {
        accumulator = (accumulator << 8) | byte;
        bits += 8;
        while (bits >= 0) {
            out.push_back(alphabet[(accumulator >> bits) & 0x3f]);
            bits -= 6;
        }
    };
    for (float value : values) {
        uint32_t word = 0;
        static_assert(sizeof(word) == sizeof(value), "float32 required");
        std::memcpy(&word, &value, sizeof(word));
        append_byte(static_cast<uint8_t>(word & 0xff));
        append_byte(static_cast<uint8_t>((word >> 8) & 0xff));
        append_byte(static_cast<uint8_t>((word >> 16) & 0xff));
        append_byte(static_cast<uint8_t>((word >> 24) & 0xff));
    }
    if (bits > -6)
        out.push_back(alphabet[(accumulator << 8 >> (bits + 8)) & 0x3f]);
    while (out.size() % 4)
        out.push_back('=');
    return out;
}

} // namespace xllama
