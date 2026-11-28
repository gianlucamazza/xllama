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
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <utility>

namespace xllama::ui {

// ---------------------------------------------------------------------------
// Minimal JSON helpers (no external dependency)
// ---------------------------------------------------------------------------

// json_escape is in xllama::json_utils.h — included above.

// Parse a JSON string value (after opening "). Returns empty string on failure.
// Advances `pos` past the closing ".
static std::string json_parse_string(const std::string& s, size_t& pos) {
    std::string out;
    if (!json_read_string(s, pos, out))
        return {};
    return out;
}

// Advance pos past whitespace.
static void skip_ws(const std::string& s, size_t& pos) {
    while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\t' || s[pos] == '\n' || s[pos] == '\r'))
        ++pos;
}

// Advance pos to next occurrence of ch; return false if not found.
static bool seek_char(const std::string& s, size_t& pos, char ch) {
    while (pos < s.size() && s[pos] != ch)
        ++pos;
    return pos < s.size();
}

// Read a JSON key: seek '"', parse string, seek ':', return key.
static std::string json_read_key(const std::string& s, size_t& pos) {
    if (!seek_char(s, pos, '"'))
        return {};
    ++pos;
    return json_parse_string(s, pos);
}

// Read a JSON value (string or number only).
static std::string json_read_value(const std::string& s, size_t& pos) {
    skip_ws(s, pos);
    if (pos >= s.size())
        return {};
    if (s[pos] == '"') {
        ++pos;
        return json_parse_string(s, pos);
    }
    // number or boolean/null — read until delimiter
    size_t start = pos;
    while (pos < s.size() && s[pos] != ',' && s[pos] != '}' && s[pos] != ']' && s[pos] != '\n')
        ++pos;
    std::string val = s.substr(start, pos - start);
    // trim trailing whitespace
    while (!val.empty() && (val.back() == ' ' || val.back() == '\t' || val.back() == '\r'))
        val.pop_back();
    return val;
}

// ---------------------------------------------------------------------------
// ChatHistory implementation
// ---------------------------------------------------------------------------

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

void ChatHistory::ReadIndex() {
    m_index.clear();
    auto idx_path = ConvPath("index");
    std::ifstream input(idx_path, std::ios::binary);
    if (!input)
        return;
    const std::string json((std::istreambuf_iterator<char>(input)), {});

    // Parse array of objects
    size_t pos = 0;
    seek_char(json, pos, '[');
    ++pos;
    while (pos < json.size()) {
        skip_ws(json, pos);
        if (pos < json.size() && json[pos] == ']')
            break;
        if (pos < json.size() && json[pos] != '{') {
            ++pos;
            continue;
        }
        ++pos; // skip '{'
        ConversationMeta meta;
        // parse up to 4 keys per object
        for (int k = 0; k < 8 && pos < json.size(); ++k) {
            skip_ws(json, pos);
            if (pos < json.size() && json[pos] == '}') {
                ++pos;
                break;
            }
            if (pos < json.size() && json[pos] == ',') {
                ++pos;
                continue;
            }
            std::string key = json_read_key(json, pos);
            if (!seek_char(json, pos, ':'))
                break;
            ++pos;
            std::string val = json_read_value(json, pos);
            if (key == "id")
                meta.id = val;
            else if (key == "title")
                meta.title = val;
            else if (key == "last_modified")
                meta.last_modified = std::stoll(val.empty() ? "0" : val);
            else if (key == "n_messages")
                meta.n_messages = std::stoi(val.empty() ? "0" : val);
        }
        if (!meta.id.empty())
            m_index.push_back(std::move(meta));
    }
    // Sort most-recent first
    std::sort(m_index.begin(), m_index.end(),
              [](const ConversationMeta& a, const ConversationMeta& b) {
                  return a.last_modified > b.last_modified;
              });
}

void ChatHistory::LoadIndex() {
    ReadIndex();
    std::error_code ec;
    std::vector<std::filesystem::path> pending;
    std::filesystem::directory_iterator it(std::filesystem::u8path(m_dir), ec), end;
    for (; !ec && it != end; it.increment(ec)) {
        const auto marker = it->path();
        const auto source = marker.parent_path() / marker.stem();
        if (marker.extension() != ".pending" || source.extension() != ".json")
            continue;
        const auto status = std::filesystem::status(source, ec);
        if (ec && ec != std::errc::no_such_file_or_directory)
            return;
        if (std::filesystem::is_regular_file(status)) {
            const auto chat = Load(source.stem().u8string());
            if (chat.id.empty())
                continue; // keep the committed index entry and retry this marker later
            UpdateMeta(chat);
        } else if (std::filesystem::exists(status)) {
            continue;
        }
        ec.clear();
        pending.push_back(marker);
    }
    if (!ec && !pending.empty() && SaveIndex()) {
        for (const auto& marker : pending)
            std::filesystem::remove(marker, ec);
    }
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
    if (!SaveIndex())
        return false;
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

// Older saves could split a UTF-8 character before the title's ellipsis. Only
// retry the exact header emitted by that writer; message bytes remain untouched.
static nlohmann::json parse_chat(const std::string& id, std::ifstream& input) {
    using Json = nlohmann::json;
    auto root = Json::parse(input, nullptr, false);
    if (!root.is_discarded() || input.bad())
        return root;
    input.clear();
    input.seekg(0);
    char header[512]; // GUID, an escaped 60-byte title and the messages key
    if (!input.getline(header, sizeof(header)))
        return root;
    std::string line(header);
    if (!line.empty() && line.back() == '\r')
        line.pop_back(); // historical Windows text-mode saves used CRLF
    const std::string prefix = "{\"id\":\"" + json_escape(id) + "\",\"title\":\"";
    size_t end = prefix.size();
    std::string title;
    if (line.compare(0, end, prefix) != 0 || !json_read_string(line, end, title) ||
        line.substr(end) != ",\"messages\":[" || title.size() != 60 || title.substr(57) != "...")
        return root;
    const std::string escaped = Json(title).dump(-1, ' ', false, Json::error_handler_t::ignore);
    if (escaped == "\"" + json_escape(title) + "\"")
        return root;
    line.replace(prefix.size() - 1, end - prefix.size() + 1, escaped);
    line += '\n';
    line.append(std::istreambuf_iterator<char>(input), {});
    return Json::parse(line, nullptr, false);
}

Conversation ChatHistory::Load(const std::string& id) const {
    std::ifstream input(ConvPath(id), std::ios::binary);
    if (!input || id.empty())
        return {};
    try {
        auto root = parse_chat(id, input);
        if (input.bad() || !root.is_object() || root.at("id") != id ||
            !root.at("messages").is_array())
            return {};
        Conversation conv{id, root.value("title", std::string{}), {}};
        auto& messages = root.at("messages");
        conv.messages.reserve(messages.size());
        for (auto& item : messages) {
            ChatMessage msg;
            const auto& role = item.at("role");
            if (role == "system")
                msg.role = MessageRole::System;
            else if (role == "assistant")
                msg.role = MessageRole::Assistant;
            else if (role != "user")
                return {};
            msg.content = std::move(item.at("content").get_ref<std::string&>());
            if (const auto ts = item.find("ts"); ts != item.end()) {
                if (!ts->is_number_integer() ||
                    (ts->is_number_unsigned() && ts->get<uint64_t>() > INT64_MAX))
                    return {};
                msg.ts_unix = ts->get<int64_t>();
            }
            msg.partial = item.value("partial", false);
            msg.feedback_label = item.value("feedback_label", std::string{});
            conv.messages.push_back(std::move(msg));
        }
        return conv;
    } catch (const nlohmann::json::exception&) {
        return {};
    } catch (const std::ios_base::failure&) {
        return {};
    }
}

} // namespace xllama::ui
