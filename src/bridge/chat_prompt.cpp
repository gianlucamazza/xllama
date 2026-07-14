// Copyright (c) 2024 Venere Labs
// SPDX-License-Identifier: MIT

#include "xllama/chat_prompt.h"

#include <algorithm>
#include <cctype>

namespace xllama {
namespace {

constexpr char kThinkOpen[] = "<think>";
constexpr char kThinkClose[] = "</think>";

std::string empty_think_block() {
    return std::string(kThinkOpen) + "\n\n" + kThinkClose;
}

std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

void ltrim_inplace(std::string& s) {
    size_t i = 0;
    while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i])))
        ++i;
    if (i > 0)
        s.erase(0, i);
}

bool starts_with(const std::string& s, const std::string& prefix) {
    return s.size() >= prefix.size() && s.compare(0, prefix.size(), prefix) == 0;
}

} // namespace

bool model_is_qwen(const std::string& model_id) {
    return to_lower(model_id).find("qwen") != std::string::npos;
}

std::string qwen_no_think_gen_suffix(const std::string& model_id) {
    if (!model_is_qwen(model_id))
        return {};
    return empty_think_block() + "\n\n";
}

std::string strip_empty_thinking_tags(std::string text) {
    const std::string kEmpty = empty_think_block();
    ltrim_inplace(text);
    while (starts_with(text, kEmpty)) {
        text.erase(0, kEmpty.size());
        ltrim_inplace(text);
    }
    return text;
}

} // namespace xllama