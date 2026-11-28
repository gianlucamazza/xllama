// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace xllama::ui {

enum class MessageRole { System, User, Assistant };

struct ChatMessage {
    MessageRole role = MessageRole::User;
    std::string content;
    int64_t ts_unix = 0;  // seconds since epoch
    bool partial = false; // true if generation was cancelled or failed mid-stream
    // Empty until the user records one immutable preference for this assistant response.
    std::string feedback_label;
};

struct ConversationMeta {
    std::string id;    // UUID string (e.g. "550e8400-e29b-41d4-a716-446655440000")
    std::string title; // first 60 chars of first user message
    int64_t last_modified = 0;
    int n_messages = 0;
};

struct Conversation {
    std::string id;
    std::string title;
    std::vector<ChatMessage> messages;
};

struct ChatDeletion {
    std::vector<std::string> removed_ids; // bodies confirmed absent
    bool all_removed = false;
    bool index_pending = false; // a journal will retry the index update
};

// ---------------------------------------------------------------------------
// ChatHistory — persists conversations under LocalState/chats/
// ---------------------------------------------------------------------------

class ChatHistory {
  public:
    ChatHistory() = default;
    explicit ChatHistory(const std::string& chats_dir); // absolute path to LocalState/chats
    void SetDir(const std::string& chats_dir) {
        m_dir = chats_dir;
        m_index_ready = false;
    }

    // Load the index and reconcile pending saves. Failed reads leave the files available for retry.
    bool LoadIndex();

    // Save conversation to disk and update index.
    bool Save(const Conversation& conv);

    // A read, format or identity failure returns an empty id. A valid chat may have no messages.
    Conversation Load(const std::string& id) const;

    // Sorted by last_modified descending (most recent first).
    const std::vector<ConversationMeta>& Index() const {
        return m_index;
    }

    // Delete a single conversation (removes JSON file + updates index).
    ChatDeletion Delete(const std::string& id);

    // Delete all conversations (removes all JSON files + clears index).
    ChatDeletion Clear();

    // Derive conversation id from title (for new convos: generate UUID).
#ifdef XLLAMA_UWP
    static std::string NewId();
#endif

    // Build title from first user message (up to 60 chars, no newlines).
    static std::string TitleFrom(const std::string& text);

  private:
    bool ReadIndex();
    void UpdateMeta(ConversationMeta meta);
    ChatDeletion RemoveChats(std::vector<std::string> ids);
    void RemoveMetadata(std::vector<std::string>& ids);
    bool SaveIndex();
    std::filesystem::path ConvPath(const std::string& id) const;

    bool m_index_ready = false;
    std::string m_dir;
    std::vector<ConversationMeta> m_index;
};

} // namespace xllama::ui
