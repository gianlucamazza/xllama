// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#pragma once

#ifdef XLLAMA_UWP

    #include <cstdint>
    #include <string>
    #include <vector>

namespace xllama::ui {

enum class MessageRole { System, User, Assistant };

struct ChatMessage {
    MessageRole role = MessageRole::User;
    std::string content;
    int64_t ts_unix = 0;  // seconds since epoch
    bool partial = false; // true if generation was cancelled mid-stream
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

// ---------------------------------------------------------------------------
// ChatHistory — persists conversations under LocalState/chats/
// ---------------------------------------------------------------------------

class ChatHistory {
  public:
    ChatHistory() = default;
    explicit ChatHistory(const std::string& chats_dir); // absolute path to LocalState/chats
    void SetDir(const std::string& chats_dir) {
        m_dir = chats_dir;
    }

    // Load index from disk. Call once at startup (best-effort; empty on failure).
    void LoadIndex();

    // Save conversation to disk and update index.
    void Save(const Conversation& conv);

    // Load a specific conversation by id. Returns empty on failure.
    Conversation Load(const std::string& id) const;

    // Sorted by last_modified descending (most recent first).
    const std::vector<ConversationMeta>& Index() const {
        return m_index;
    }

    // Delete a single conversation (removes JSON file + updates index).
    void Delete(const std::string& id);

    // Delete all conversations (removes all JSON files + clears index).
    void Clear();

    // Derive conversation id from title (for new convos: generate UUID).
    static std::string NewId();

    // Build title from first user message (up to 60 chars, no newlines).
    static std::string TitleFrom(const std::string& text);

  private:
    void SaveIndex();
    std::string ConvPath(const std::string& id) const;

    std::string m_dir;
    std::vector<ConversationMeta> m_index;
};

} // namespace xllama::ui

#endif // XLLAMA_UWP
