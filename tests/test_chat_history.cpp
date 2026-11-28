// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include "../uwp/chat-history.h"
#include "xllama/atomic_file.h"
#include <chrono>
#include <doctest/doctest.h>
#include <filesystem>
#include <fstream>

TEST_CASE("ChatHistory::TitleFrom — basic") {
    CHECK(xllama::ui::ChatHistory::TitleFrom("Hello world") == "Hello world");
    CHECK(xllama::ui::ChatHistory::TitleFrom("") == "Conversation");
    CHECK(xllama::ui::ChatHistory::TitleFrom("   ") == "Conversation");
    CHECK(xllama::ui::ChatHistory::TitleFrom("\n\nActual title") == "Actual title");
}

TEST_CASE("ChatHistory::TitleFrom — truncates at 60 chars") {
    std::string long_str(70, 'x');
    std::string result = xllama::ui::ChatHistory::TitleFrom(long_str);
    CHECK(result.size() == 60);
    CHECK(result.substr(57) == "...");
}

TEST_CASE("ChatHistory::TitleFrom — stops at first newline") {
    std::string result = xllama::ui::ChatHistory::TitleFrom("line one\nline two");
    CHECK(result == "line one");
}

TEST_CASE("ChatHistory::TitleFrom — exact 60 chars not truncated") {
    std::string exact(60, 'a');
    CHECK(xllama::ui::ChatHistory::TitleFrom(exact) == exact);
}

struct ChatDirectory {
    std::filesystem::path path =
        std::filesystem::temp_directory_path() /
        ("xllama-chat-test-" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    ~ChatDirectory() {
        std::error_code ec;
        std::filesystem::remove_all(path, ec);
    }
};

TEST_CASE("ChatHistory::Save + Load roundtrip") {
    ChatDirectory temp;
    std::string dir = temp.path.string();
    xllama::ui::ChatHistory h(dir);
    h.LoadIndex();

    xllama::ui::Conversation conv;
    conv.id = "chat";
    conv.title = "Test conversation";
    xllama::ui::ChatMessage msg;
    msg.role = xllama::ui::MessageRole::User;
    msg.content = "Hello!";
    conv.messages.push_back(msg);
    xllama::ui::ChatMessage assistant;
    assistant.role = xllama::ui::MessageRole::Assistant;
    assistant.content = "Hi!";
    assistant.feedback_label = "like";
    conv.messages.push_back(assistant);
    h.Save(conv);

    auto loaded = h.Load(conv.id);
    CHECK(loaded.id == conv.id);
    CHECK(loaded.title == conv.title);
    CHECK(loaded.messages.size() == 2);
    CHECK(loaded.messages[0].content == "Hello!");
    CHECK(loaded.messages[1].feedback_label == "like");
}

TEST_CASE("ChatHistory::Delete removes entry from index and disk") {
    ChatDirectory temp;
    std::string dir = temp.path.string();
    xllama::ui::ChatHistory h(dir);
    h.LoadIndex();

    xllama::ui::Conversation conv;
    conv.id = "chat";
    conv.title = "To be deleted";
    h.Save(conv);

    CHECK(h.Index().size() >= 1);
    h.Delete(conv.id);

    // Not in index
    bool found = false;
    for (const auto& m : h.Index())
        if (m.id == conv.id)
            found = true;
    CHECK_FALSE(found);

    // File gone
    std::string path = (std::filesystem::path(dir) / (conv.id + ".json")).string();
    CHECK_FALSE(std::filesystem::exists(path));
}

TEST_CASE("ChatHistory::Clear wipes all entries") {
    ChatDirectory temp;
    std::string dir = temp.path.string();
    xllama::ui::ChatHistory h(dir);
    h.LoadIndex();

    for (int i = 0; i < 3; ++i) {
        xllama::ui::Conversation c;
        c.id = "chat" + std::to_string(i);
        c.title = "Conv " + std::to_string(i);
        h.Save(c);
    }
    CHECK(h.Index().size() >= 3);

    h.Clear();
    CHECK(h.Index().empty());
}

TEST_CASE("ChatHistory::TitleFrom") {
    CHECK(xllama::ui::ChatHistory::TitleFrom("Hello world") == "Hello world");
    CHECK(xllama::ui::ChatHistory::TitleFrom("") == "Conversation");
    CHECK(xllama::ui::ChatHistory::TitleFrom(std::string(70, 'x')).size() == 60);
}

TEST_CASE("ChatHistory preserves committed bytes and repairs an interrupted index update") {
    ChatDirectory temp;
    xllama::ui::ChatHistory history(temp.path.string());
    xllama::ui::Conversation chat{"test", "previous", {}};
    REQUIRE(history.Save(chat));
    chat.title = "updated";
    // Prevent opening the sibling staging file. An existing chat must survive.
    std::filesystem::create_directory(temp.path / "test.json.tmp");
    CHECK_FALSE(history.Save(chat));
    CHECK(history.Load(chat.id).title == "previous");
    std::filesystem::remove(temp.path / "test.json.tmp");
    // The conversation succeeds but replacing the index fails.
    std::filesystem::create_directory(temp.path / "index.json.tmp");
    CHECK_FALSE(history.Save(chat));
    std::filesystem::remove(temp.path / "index.json.tmp");
    xllama::ui::ChatHistory reopened(temp.path.string());
    reopened.LoadIndex();
    REQUIRE(reopened.Index().size() == 1);
    CHECK(reopened.Index()[0].title == "updated");
    CHECK_FALSE(std::filesystem::exists(temp.path / "test.json.pending"));
}

TEST_CASE("aborted staged write preserves the entire prior chat") {
    ChatDirectory temp;
    xllama::ui::ChatHistory history(temp.path.string());
    xllama::ui::Conversation chat{"interrupted", "last good chat", {}};
    chat.messages.push_back({xllama::ui::MessageRole::User, "existing message", 10, false, {}});
    REQUIRE(history.Save(chat));
    CHECK_FALSE(xllama::replace_file_checked(temp.path / "interrupted.json", [](FILE* file) {
        std::fputs("{partial", file);
        return false;
    }));
    const auto recovered = history.Load(chat.id);
    REQUIRE(recovered.messages.size() == 1);
    CHECK(recovered.messages.front().content == "existing message");
}
