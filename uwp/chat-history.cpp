// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#ifdef XLLAMA_UWP

// clang-format off
#include "pch.h"
    // clang-format on
    #include <objbase.h>
#endif

#include "chat-history.h"
#include "xllama/atomic_file.h"
#include "xllama/json_utils.h"
#include "xllama/platform.h"
#include "xllama/sse.h"

#include <algorithm>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <nlohmann/json.hpp>

namespace xllama::ui {

namespace {
nlohmann::json read_chat_json(std::ifstream& in) {
    auto json = nlohmann::json::parse(in, nullptr, false);
    if (!json.is_discarded() || in.bad())
        return json;

    // Older TitleFrom could split a multi-byte character at the title limit.
    // Repair UTF-8 in memory; a structurally incomplete document still fails.
    in.clear();
    in.seekg(0);
    const std::string bytes(std::istreambuf_iterator<char>(in), {});
    std::string_view remaining(bytes);
    std::string repaired;
    while (!remaining.empty()) {
        const auto step = sse_utf8_step(remaining, true);
        if (step.replace)
            repaired += "\xef\xbf\xbd";
        else
            repaired.append(remaining.substr(0, step.take));
        remaining.remove_prefix(step.take);
    }
    return nlohmann::json::parse(repaired, nullptr, false);
}
} // namespace

ChatHistory::ChatHistory(const std::string& chats_dir) : m_dir(chats_dir) {}

#ifdef XLLAMA_UWP
std::string ChatHistory::NewId() {
    GUID g{};
    CoCreateGuid(&g);
    char buf[37];
    snprintf(buf, sizeof(buf), "%08lx-%04x-%04x-%02x%02x-%02x%02x%02x%02x%02x%02x",
             static_cast<unsigned long>(g.Data1), g.Data2, g.Data3, g.Data4[0], g.Data4[1],
             g.Data4[2], g.Data4[3], g.Data4[4], g.Data4[5], g.Data4[6], g.Data4[7]);
    return buf;
}

#endif

std::string ChatHistory::TitleFrom(const std::string& text) {
    // First non-empty line, truncated at 60 chars
    size_t pos = 0;
    while (pos < text.size() && (text[pos] == '\n' || text[pos] == '\r' || text[pos] == ' '))
        ++pos;
    size_t end = pos;
    while (end < text.size() && text[end] != '\n' && text[end] != '\r')
        ++end;
    std::string title = text.substr(pos, end - pos);
    if (title.size() > 60) {
        size_t cut = 57;
        while (cut && (static_cast<unsigned char>(title[cut]) & 0xc0) == 0x80)
            --cut;
        title.resize(cut);
        title += "...";
    }
    return title.empty() ? "Conversation" : title;
}

std::filesystem::path ChatHistory::ConvPath(const std::string& id) const {
    return (std::filesystem::u8path(m_dir) / std::filesystem::u8path(id + ".json"));
}

bool ChatHistory::ReadIndex() {
    const auto path = ConvPath("index");
    std::error_code ec;
    if (!std::filesystem::exists(path, ec)) {
        if (ec)
            return false;
        m_index.clear();
        return true;
    }
    if (!std::filesystem::is_regular_file(path, ec))
        return false;
    std::ifstream in(path, std::ios::binary);
    if (!in)
        return false;
    try {
        const auto json = read_chat_json(in);
        if (!json.is_array() || in.bad())
            return false;
        std::vector<ConversationMeta> index;
        for (const auto& item : json) {
            ConversationMeta meta;
            meta.id = item.at("id").get<std::string>();
            if (meta.id.empty())
                return false;
            meta.title = item.value("title", std::string{});
            meta.last_modified = item.value("last_modified", int64_t{0});
            meta.n_messages = item.value("n_messages", 0);
            index.push_back(std::move(meta));
        }
        std::sort(index.begin(), index.end(),
                  [](const auto& a, const auto& b) { return a.last_modified > b.last_modified; });
        m_index = std::move(index);
        return true;
    } catch (const nlohmann::json::exception&) {
        return false;
    }
}

bool ChatHistory::LoadIndex() {
    m_index_ready = false;
    if (!ReadIndex())
        return false;
    std::error_code ec;
    std::vector<std::filesystem::path> pending;
    std::vector<Conversation> recovered;
    std::filesystem::directory_iterator it(std::filesystem::u8path(m_dir), ec), end;
    if (ec == std::errc::no_such_file_or_directory)
        ec.clear();
    for (; !ec && it != end; it.increment(ec)) {
        const auto marker = it->path();
        const auto source = marker.parent_path() / marker.stem();
        if (marker.extension() != ".pending" || source.extension() != ".json")
            continue;
        if (std::filesystem::exists(source, ec)) {
            auto conv = Load(source.stem().string());
            if (conv.id.empty())
                return false;
            recovered.push_back(std::move(conv));
        } else if (ec) {
            return false;
        }
        pending.push_back(marker);
    }
    if (ec)
        return false;
    for (const auto& conv : recovered)
        UpdateMeta(conv);
    if (!pending.empty()) {
        if (!SaveIndex())
            return false;
        for (const auto& marker : pending)
            std::filesystem::remove(marker, ec);
    }
    m_index_ready = true;
    return true;
}

bool ChatHistory::SaveIndex() {
    auto idx_path = ConvPath("index");
    return replace_file_checked(idx_path, [&](FILE* f) {
        fputs("[\n", f);
        for (size_t i = 0; i < m_index.size(); ++i) {
            const auto& m = m_index[i];
            fprintf(
                f,
                "  {\"id\":\"%s\",\"title\":\"%s\",\"last_modified\":%lld,\"n_messages\":%d}%s\n",
                xllama::json_escape(m.id).c_str(), xllama::json_escape(m.title).c_str(),
                static_cast<long long>(m.last_modified), m.n_messages,
                (i + 1 < m_index.size()) ? "," : "");
        }
        fputs("]\n", f);
        return !ferror(f);
    });
}

void ChatHistory::Delete(const std::string& id) {
    if (!m_index_ready && !LoadIndex())
        return;
    // Remove JSON file
    auto path = ConvPath(id);
    std::error_code ec;
    std::filesystem::remove(path, ec);
    // Remove from index
    m_index.erase(std::remove_if(m_index.begin(), m_index.end(),
                                 [&](const ConversationMeta& m) { return m.id == id; }),
                  m_index.end());
    SaveIndex();
}

void ChatHistory::Clear() {
    if (!m_index_ready && !LoadIndex())
        return;
    // Delete all conversation JSON files
    for (const auto& meta : m_index) {
        auto path = ConvPath(meta.id);
        std::error_code ec;
        std::filesystem::remove(path, ec);
    }
    m_index.clear();
    SaveIndex();
}

bool ChatHistory::Save(const Conversation& conv) {
    if (!m_index_ready && !LoadIndex())
        return false;
    // Ensure directory exists
    std::error_code ec;
    std::filesystem::create_directories(std::filesystem::u8path(m_dir), ec);
    if (ec)
        return false;

    auto path = ConvPath(conv.id);
    const auto pending = std::filesystem::path(path).concat(".pending");
    if (!replace_file_bytes(pending, ""))
        return false;
    if (!replace_file_checked(path, [&](FILE* f) {
            fputs("{\"id\":\"", f);
            fputs(xllama::json_escape(conv.id).c_str(), f);
            fputs("\",\"title\":\"", f);
            fputs(xllama::json_escape(conv.title).c_str(), f);
            fputs("\",\"messages\":[\n", f);
            for (size_t i = 0; i < conv.messages.size(); ++i) {
                const auto& msg = conv.messages[i];
                const char* role_str = (msg.role == MessageRole::System) ? "system"
                                       : (msg.role == MessageRole::User) ? "user"
                                                                         : "assistant";
                fprintf(f,
                        "  {\"role\":\"%s\",\"content\":\"%s\",\"ts\":%lld,\"partial\":%s,"
                        "\"feedback_label\":\"%s\"}%s\n",
                        role_str, xllama::json_escape(msg.content).c_str(),
                        static_cast<long long>(msg.ts_unix), msg.partial ? "true" : "false",
                        xllama::json_escape(msg.feedback_label).c_str(),
                        (i + 1 < conv.messages.size()) ? "," : "");
            }
            fputs("]}\n", f);
            return !ferror(f);
        }))
        return false;

    UpdateMeta(conv);
    if (!SaveIndex()) {
        m_index_ready = false;
        return false;
    }
    std::filesystem::remove(pending, ec);
    return true;
}

void ChatHistory::UpdateMeta(const Conversation& conv) {
    int64_t now_ts = static_cast<int64_t>(std::time(nullptr));
    auto it = std::find_if(m_index.begin(), m_index.end(),
                           [&](const ConversationMeta& m) { return m.id == conv.id; });
    ConversationMeta meta;
    meta.id = conv.id;
    meta.title = conv.title;
    meta.last_modified = now_ts;
    meta.n_messages = static_cast<int>(conv.messages.size());
    if (it != m_index.end()) {
        *it = std::move(meta);
    } else {
        m_index.insert(m_index.begin(), std::move(meta));
    }
    // Keep index sorted
    std::sort(m_index.begin(), m_index.end(),
              [](const ConversationMeta& a, const ConversationMeta& b) {
                  return a.last_modified > b.last_modified;
              });
}

Conversation ChatHistory::Load(const std::string& id) const {
    const auto path = ConvPath(id);
    std::error_code ec;
    if (!std::filesystem::is_regular_file(path, ec))
        return {};
    std::ifstream in(path, std::ios::binary);
    if (!in)
        return {};
    try {
        const auto json = read_chat_json(in);
        if (in.bad() || !json.is_object() || !json.contains("messages") ||
            !json.at("messages").is_array() || json.value("id", std::string{}) != id)
            return {};
        Conversation conv{id, json.value("title", std::string{}), {}};
        for (const auto& item : json.at("messages")) {
            ChatMessage msg;
            const auto role = item.value("role", std::string{"user"});
            msg.role = role == "system" ? MessageRole::System
                       : role == "user" ? MessageRole::User
                                        : MessageRole::Assistant;
            msg.content = item.value("content", std::string{});
            msg.ts_unix = item.value("ts", int64_t{0});
            msg.partial = item.value("partial", false);
            msg.feedback_label = item.value("feedback_label", std::string{});
            conv.messages.push_back(std::move(msg));
        }
        return conv;
    } catch (const nlohmann::json::exception&) {
        return {};
    }
}

} // namespace xllama::ui
