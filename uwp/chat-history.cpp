// Copyright (c) 2024 Venere Labs
// SPDX-License-Identifier: MIT

#ifdef XLLAMA_UWP

// clang-format off
#include "pch.h"
// clang-format on

    #include "chat-history.h"
    #include "xllama/platform.h"
    #include "xllama/utf8_utils.h"

    #include <algorithm>
    #include <cstdio>
    #include <ctime>
    #include <sstream>

    // Windows headers for UUID generation
    #include <objbase.h>

namespace xllama::ui {

// ---------------------------------------------------------------------------
// Minimal JSON helpers (no external dependency)
// ---------------------------------------------------------------------------

// Escape a UTF-8 string for JSON output: replace " → \", \ → \\, control chars.
static std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (unsigned char c : s) {
        if (c == '"') {
            out += "\\\"";
        } else if (c == '\\') {
            out += "\\\\";
        } else if (c == '\n') {
            out += "\\n";
        } else if (c == '\r') {
            out += "\\r";
        } else if (c == '\t') {
            out += "\\t";
        } else if (c < 0x20) { /* skip other control chars */
        } else {
            out += static_cast<char>(c);
        }
    }
    return out;
}

// Parse a JSON string value (after opening "). Returns empty string on failure.
// Advances `pos` past the closing ".
static std::string json_parse_string(const std::string& s, size_t& pos) {
    std::string out;
    while (pos < s.size()) {
        char c = s[pos++];
        if (c == '"')
            return out;
        if (c == '\\' && pos < s.size()) {
            char esc = s[pos++];
            switch (esc) {
            case '"':
                out += '"';
                break;
            case '\\':
                out += '\\';
                break;
            case 'n':
                out += '\n';
                break;
            case 'r':
                out += '\r';
                break;
            case 't':
                out += '\t';
                break;
            default:
                out += esc;
                break;
            }
        } else {
            out += c;
        }
    }
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

std::string ChatHistory::NewId() {
    GUID g{};
    CoCreateGuid(&g);
    char buf[37];
    snprintf(buf, sizeof(buf), "%08lx-%04x-%04x-%02x%02x-%02x%02x%02x%02x%02x%02x",
             static_cast<unsigned long>(g.Data1), g.Data2, g.Data3, g.Data4[0], g.Data4[1],
             g.Data4[2], g.Data4[3], g.Data4[4], g.Data4[5], g.Data4[6], g.Data4[7]);
    return buf;
}

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
        title.resize(57);
        title += "...";
    }
    return title.empty() ? "Conversation" : title;
}

std::string ChatHistory::ConvPath(const std::string& id) const {
    return m_dir + "\\" + id + ".json";
}

void ChatHistory::LoadIndex() {
    m_index.clear();
    std::string idx_path = m_dir + "\\index.json";
    FILE* f = fopen(idx_path.c_str(), "r");
    if (!f)
        return;
    std::string json;
    char buf[4096];
    while (size_t n = fread(buf, 1, sizeof(buf), f))
        json.append(buf, n);
    fclose(f);

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

void ChatHistory::SaveIndex() {
    std::string idx_path = m_dir + "\\index.json";
    FILE* f = fopen(idx_path.c_str(), "w");
    if (!f)
        return;
    fputs("[\n", f);
    for (size_t i = 0; i < m_index.size(); ++i) {
        const auto& m = m_index[i];
        fprintf(f,
                "  {\"id\":\"%s\",\"title\":\"%s\",\"last_modified\":%lld,\"n_messages\":%d}%s\n",
                json_escape(m.id).c_str(), json_escape(m.title).c_str(),
                static_cast<long long>(m.last_modified), m.n_messages,
                (i + 1 < m_index.size()) ? "," : "");
    }
    fputs("]\n", f);
    fclose(f);
}

void ChatHistory::Delete(const std::string& id) {
    // Remove JSON file
    std::string path = ConvPath(id);
    ::remove(path.c_str());
    // Remove from index
    m_index.erase(std::remove_if(m_index.begin(), m_index.end(),
                                 [&](const ConversationMeta& m) { return m.id == id; }),
                  m_index.end());
    SaveIndex();
}

void ChatHistory::Clear() {
    // Delete all conversation JSON files
    for (const auto& meta : m_index) {
        std::string path = ConvPath(meta.id);
        ::remove(path.c_str());
    }
    m_index.clear();
    SaveIndex();
}

void ChatHistory::Save(const Conversation& conv) {
    // Ensure directory exists
    std::wstring wdir = ::xllama::utf8_to_wstring(m_dir);
    CreateDirectoryW(wdir.c_str(), nullptr); // no-op if exists

    std::string path = ConvPath(conv.id);
    FILE* f = fopen(path.c_str(), "w");
    if (!f) {
        log_output("[xllama][chat] failed to save conversation " + conv.id + "\n");
        return;
    }
    fputs("{\"id\":\"", f);
    fputs(json_escape(conv.id).c_str(), f);
    fputs("\",\"title\":\"", f);
    fputs(json_escape(conv.title).c_str(), f);
    fputs("\",\"messages\":[\n", f);
    for (size_t i = 0; i < conv.messages.size(); ++i) {
        const auto& msg = conv.messages[i];
        const char* role_str = (msg.role == MessageRole::System) ? "system"
                               : (msg.role == MessageRole::User) ? "user"
                                                                 : "assistant";
        fprintf(f,
                "  {\"role\":\"%s\",\"content\":\"%s\",\"ts\":%lld,\"partial\":%s,"
                "\"feedback_label\":\"%s\"}%s\n",
                role_str, json_escape(msg.content).c_str(), static_cast<long long>(msg.ts_unix),
                msg.partial ? "true" : "false", json_escape(msg.feedback_label).c_str(),
                (i + 1 < conv.messages.size()) ? "," : "");
    }
    fputs("]}\n", f);
    fclose(f);

    // Update index
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
    SaveIndex();
}

Conversation ChatHistory::Load(const std::string& id) const {
    Conversation conv;
    conv.id = id;
    std::string path = ConvPath(id);
    FILE* f = fopen(path.c_str(), "r");
    if (!f)
        return conv;
    std::string json;
    char buf[4096];
    while (size_t n = fread(buf, 1, sizeof(buf), f))
        json.append(buf, n);
    fclose(f);

    // Parse top-level fields
    size_t pos = 0;
    seek_char(json, pos, '{');
    ++pos;
    // Find "messages" array start
    size_t msgs_pos = json.find("\"messages\"");
    if (msgs_pos == std::string::npos)
        return conv;

    // Parse id and title first
    {
        size_t p2 = 0;
        while (p2 < msgs_pos) {
            skip_ws(json, p2);
            if (p2 >= msgs_pos)
                break;
            if (json[p2] == ',') {
                ++p2;
                continue;
            }
            if (json[p2] != '"') {
                ++p2;
                continue;
            }
            std::string key = json_read_key(json, p2);
            if (!seek_char(json, p2, ':'))
                break;
            ++p2;
            std::string val = json_read_value(json, p2);
            if (key == "id")
                conv.id = val;
            if (key == "title")
                conv.title = val;
        }
    }

    // Parse messages array
    pos = msgs_pos + 10; // skip "messages"
    if (!seek_char(json, pos, '['))
        return conv;
    ++pos;
    while (pos < json.size()) {
        skip_ws(json, pos);
        if (pos < json.size() && json[pos] == ']')
            break;
        if (pos < json.size() && json[pos] != '{') {
            ++pos;
            continue;
        }
        ++pos;
        ChatMessage msg;
        for (int k = 0; k < 10 && pos < json.size(); ++k) {
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
            if (key == "role") {
                if (val == "system")
                    msg.role = MessageRole::System;
                else if (val == "user")
                    msg.role = MessageRole::User;
                else
                    msg.role = MessageRole::Assistant;
            } else if (key == "content") {
                msg.content = val;
            } else if (key == "ts") {
                msg.ts_unix = val.empty() ? 0 : std::stoll(val);
            } else if (key == "partial") {
                msg.partial = (val == "true");
            } else if (key == "feedback_label") {
                msg.feedback_label = val;
            }
        }
        conv.messages.push_back(std::move(msg));
    }
    return conv;
}

} // namespace xllama::ui

#endif // XLLAMA_UWP
