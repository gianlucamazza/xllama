// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include "xllama/preference_capture.h"
#include "xllama/json_utils.h"

#include <chrono>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <sstream>

namespace xllama {
namespace {

std::string now_iso_utc() {
    using clock = std::chrono::system_clock;
    const auto t = clock::to_time_t(clock::now());
    std::tm tm{};
#if defined(_WIN32)
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm);
    return buf;
}

} // namespace

bool preference_label_valid(const std::string& label) {
    return label == "like" || label == "dislike" || label == "correction" || label == "implicit";
}

std::string
format_preference_sample_jsonl(const std::string& label,
                               const std::vector<std::pair<std::string, std::string>>& messages,
                               const std::string& preferred_assistant, const std::string& ts_iso) {
    if (!preference_label_valid(label) || messages.empty() ||
        (label == "correction" &&
         preferred_assistant.find_first_not_of(" \t\r\n") == std::string::npos))
        return {};
    std::ostringstream os;
    os << "{\"ts\":\"" << json_escape(ts_iso.empty() ? now_iso_utc() : ts_iso) << "\","
       << "\"label\":\"" << json_escape(label) << "\",\"messages\":[";
    for (size_t i = 0; i < messages.size(); ++i) {
        if (i)
            os << ',';
        os << "{\"role\":\"" << json_escape(messages[i].first) << "\",\"content\":\""
           << json_escape(messages[i].second) << "\"}";
    }
    os << ']';
    if (!preferred_assistant.empty())
        os << ",\"preferred_assistant\":\"" << json_escape(preferred_assistant) << '"';
    os << '}';
    return os.str();
}

bool append_preference_sample_file(const std::string& path, const std::string& jsonl_line) {
    if (jsonl_line.empty())
        return false;
    std::ofstream out(path, std::ios::app | std::ios::binary);
    if (!out)
        return false;
    out << jsonl_line;
    if (jsonl_line.back() != '\n')
        out << '\n';
    return static_cast<bool>(out);
}

} // namespace xllama
