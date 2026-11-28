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
#include "xllama/kv_store.h"
#include "xllama/sse.h"

#include <algorithm>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>

namespace xllama::ui {

namespace {
// A restored history may use a UTF-8 id. It must still name one conversation
// on both host and Windows, without aliasing the index or a DOS device.
bool valid_history_id(const std::string& id) {
    if (id.empty() || id == "." || id == ".." || id.find_first_of("/\\:<>\"|?*") != id.npos)
        return false;
    const auto utf8 = sse_utf8_step(id, true);
    if (utf8.replace || utf8.take != id.size())
        return false;
    auto name = id;
    for (char& c : name) {
        if (static_cast<unsigned char>(c) < 32)
            return false;
        if (c >= 'A' && c <= 'Z')
            c = static_cast<char>(c - 'A' + 'a');
    }
    if (name == "index")
        return false;
    auto stem = std::string_view(name).substr(0, name.find('.'));
    while (!stem.empty() && stem.back() == ' ')
        stem.remove_suffix(1);
    if (stem == "con" || stem == "prn" || stem == "aux" || stem == "nul" || stem == "conin$" ||
        stem == "conout$")
        return false;
    if (stem.substr(0, 3) == "com" || stem.substr(0, 3) == "lpt") {
        const auto digit = stem.substr(3);
        if ((digit.size() == 1 && digit[0] >= '1' && digit[0] <= '9') || digit == "\xc2\xb9" ||
            digit == "\xc2\xb2" || digit == "\xc2\xb3")
            return false;
    }
    return true;
}

nlohmann::json read_index_json(std::ifstream& in) {
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
// nlohmann converts numbers to the destination type on get<T>(), including
// truncation of fractions and unsigned values outside the signed range.
bool read_history_integer(const nlohmann::json& row, const char* key, int64_t& value) {
    const auto field = row.find(key);
    if (field == row.end())
        return true; // optional in older histories
    if (!field->is_number_integer() ||
        (field->is_number_unsigned() && field->get<uint64_t>() > INT64_MAX))
        return false;
    value = field->get<int64_t>();
    return true;
}

constexpr auto max_message_count = std::numeric_limits<int>::max();

ConversationMeta metadata_for(const Conversation& conv) {
    return {conv.id, conv.title, static_cast<int64_t>(std::time(nullptr)),
            static_cast<int>(conv.messages.size())};
}
} // namespace

bool ChatDeletion::ApplyClear(Conversation& active, const KvStore& snapshots) const {
    // The active conversation may exist only in memory after its first save failed.
    const bool removed_active =
        std::find(removed_ids.begin(), removed_ids.end(), active.id) != removed_ids.end();
    if (all_removed && (removed_active || active.id.empty())) {
        snapshots.prune(/*max_files=*/0, /*max_bytes=*/0);
    } else {
        for (const auto& id : removed_ids)
            snapshots.erase(id);
    }
    if (removed_active)
        active = {}; // NewChat saves the previous active conversation.
    return removed_active;
}

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
        const auto json = read_index_json(in);
        if (!json.is_array() || in.bad())
            return false;
        std::vector<ConversationMeta> index;
        for (const auto& item : json) {
            ConversationMeta meta;
            meta.id = item.at("id").get<std::string>();
            if (!valid_history_id(meta.id))
                return false;
            meta.title = item.value("title", std::string{});
            int64_t count = 0;
            if (!read_history_integer(item, "last_modified", meta.last_modified) ||
                !read_history_integer(item, "n_messages", count) || count < 0 ||
                count > max_message_count)
                return false;
            meta.n_messages = static_cast<int>(count);
            index.push_back(std::move(meta));
        }
        std::sort(index.begin(), index.end(),
                  [](const auto& a, const auto& b) { return a.last_modified > b.last_modified; });
        m_index = std::move(index);
        return true;
    } catch (const nlohmann::json::exception&) {
        return false;
    } catch (const std::ios_base::failure&) {
        return false;
    }
}

bool ChatHistory::LoadIndex() {
    m_index_ready = false;
    if (!ReadIndex())
        return false;
    std::error_code ec;
    std::vector<std::filesystem::path> pending;
    std::vector<ConversationMeta> recovered;
    std::vector<std::string> removed;
    std::filesystem::directory_iterator it(std::filesystem::u8path(m_dir), ec), end;
    if (ec == std::errc::no_such_file_or_directory)
        ec.clear();
    for (; !ec && it != end; it.increment(ec)) {
        const auto marker = it->path();
        const auto source = marker.parent_path() / marker.stem();
        const bool deleting = marker.extension() == ".deleting";
        if ((!deleting && marker.extension() != ".pending") || source.extension() != ".json")
            continue;
        const auto name = source.stem().u8string();
        const std::string id(name.begin(), name.end());
        if (!valid_history_id(id))
            continue;
        const bool present = std::filesystem::exists(source, ec);
        if (ec)
            return false;
        if (deleting) {
            // A crash before removing the body leaves its index entry intact.
            if (!present)
                removed.push_back(id);
        } else if (present) {
            const auto conv = Load(id);
            if (conv.id.empty())
                return false;
            recovered.push_back(metadata_for(conv));
        }
        pending.push_back(marker);
    }
    if (ec)
        return false;
    for (auto& meta : recovered)
        UpdateMeta(std::move(meta));
    RemoveMetadata(removed);
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

void ChatHistory::RemoveMetadata(std::vector<std::string>& ids) {
    if (ids.empty())
        return;
    std::sort(ids.begin(), ids.end());
    m_index.erase(std::remove_if(m_index.begin(), m_index.end(),
                                 [&](const auto& meta) {
                                     return std::binary_search(ids.begin(), ids.end(), meta.id);
                                 }),
                  m_index.end());
}

ChatDeletion ChatHistory::RemoveChats(std::vector<std::string> ids) {
    ChatDeletion result;
    result.all_removed = true;
    result.removed_ids.reserve(ids.size());
    for (auto& id : ids) {
        const auto path = ConvPath(id);
        const auto marker = std::filesystem::path(path).concat(".deleting");
        std::error_code ec;
        const auto status = std::filesystem::symlink_status(path, ec);
        if ((ec && ec != std::errc::no_such_file_or_directory) ||
            (std::filesystem::exists(status) && !std::filesystem::is_regular_file(status)) ||
            !replace_file_bytes(marker, "")) {
            result.all_removed = false;
            continue;
        }
        std::filesystem::remove(path, ec); // a missing body already satisfies the request
        if (ec) {
            result.all_removed = false;
            std::filesystem::remove(marker, ec);
            continue;
        }
        result.removed_ids.push_back(std::move(id));
    }
    if (result.removed_ids.empty())
        return result;
    RemoveMetadata(result.removed_ids);
    if (!SaveIndex()) {
        m_index_ready = false;
        result.index_pending = true;
        return result;
    }
    for (const auto& id : result.removed_ids) {
        std::error_code ec;
        std::filesystem::remove(ConvPath(id).concat(".deleting"), ec);
    }
    return result;
}

ChatDeletion ChatHistory::Delete(const std::string& id) {
    if (!valid_history_id(id) || (!m_index_ready && !LoadIndex()))
        return {};
    return RemoveChats({id});
}

ChatDeletion ChatHistory::Clear() {
    if (!m_index_ready && !LoadIndex())
        return {};
    std::vector<std::string> ids;
    ids.reserve(m_index.size());
    for (const auto& meta : m_index)
        ids.push_back(meta.id);
    return RemoveChats(std::move(ids));
}

bool ChatHistory::Save(const Conversation& conv) {
    if (!valid_history_id(conv.id) || conv.messages.size() > max_message_count ||
        (!m_index_ready && !LoadIndex()))
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

    UpdateMeta(metadata_for(conv));
    if (!SaveIndex()) {
        m_index_ready = false;
        return false;
    }
    std::filesystem::remove(pending, ec);
    return true;
}

void ChatHistory::UpdateMeta(ConversationMeta meta) {
    auto it = std::find_if(m_index.begin(), m_index.end(),
                           [&](const ConversationMeta& m) { return m.id == meta.id; });
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
    if (!valid_history_id(id))
        return {};
    const auto path = ConvPath(id);
    std::error_code ec;
    if (!std::filesystem::is_regular_file(path, ec))
        return {};
    std::ifstream input(path, std::ios::binary);
    if (!input)
        return {};
    try {
        auto root = parse_chat(id, input);
        if (input.bad() || !root.is_object() || root.at("id") != id ||
            !root.at("messages").is_array())
            return {};
        Conversation conv{id, root.value("title", std::string{}), {}};
        auto& messages = root.at("messages");
        if (messages.size() > max_message_count)
            return {};
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
            if (!read_history_integer(item, "ts", msg.ts_unix))
                return {};
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
