// Copyright (c) 2024 Venere Labs
// SPDX-License-Identifier: MIT

// ChatHistory is compiled only for the UWP target (XLLAMA_UWP).
// On Linux CI we test the platform-neutral helpers inline.
// On UWP the #else branch exercises Delete/Clear via the real class.

#include <cstdint>
#include <doctest/doctest.h>
#include <string>

// ---------------------------------------------------------------------------
// Linux CI: test pure string logic (mirrors ChatHistory static helpers)
// ---------------------------------------------------------------------------
#ifndef XLLAMA_UWP

// Mirror of ChatHistory::TitleFrom (no WinRT deps)
static std::string title_from(const std::string& text) {
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

TEST_CASE("ChatHistory::TitleFrom — basic") {
    CHECK(title_from("Hello world") == "Hello world");
    CHECK(title_from("") == "Conversation");
    CHECK(title_from("   ") == "Conversation");
    CHECK(title_from("\n\nActual title") == "Actual title");
}

TEST_CASE("ChatHistory::TitleFrom — truncates at 60 chars") {
    std::string long_str(70, 'x');
    std::string result = title_from(long_str);
    CHECK(result.size() == 60);
    CHECK(result.substr(57) == "...");
}

TEST_CASE("ChatHistory::TitleFrom — stops at first newline") {
    std::string result = title_from("line one\nline two");
    CHECK(result == "line one");
}

TEST_CASE("ChatHistory::TitleFrom — exact 60 chars not truncated") {
    std::string exact(60, 'a');
    CHECK(title_from(exact) == exact);
}

#else // XLLAMA_UWP

    // ---------------------------------------------------------------------------
    // UWP smoke tests: require Windows filesystem and CoCreateGuid
    // ---------------------------------------------------------------------------
    #include "uwp/chat-history.h"
    #include <cstdio>
    #include <filesystem>

static std::string make_tmpdir() {
    auto base = std::filesystem::temp_directory_path() / "xllama_test_chats";
    std::filesystem::create_directories(base);
    return base.string();
}

TEST_CASE("ChatHistory::Save + Load roundtrip") {
    std::string dir = make_tmpdir();
    xllama::ui::ChatHistory h(dir);
    h.LoadIndex();

    xllama::ui::Conversation conv;
    conv.id = xllama::ui::ChatHistory::NewId();
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
    std::string dir = make_tmpdir();
    xllama::ui::ChatHistory h(dir);
    h.LoadIndex();

    xllama::ui::Conversation conv;
    conv.id = xllama::ui::ChatHistory::NewId();
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
    std::string path = dir + "\\" + conv.id + ".json";
    CHECK_FALSE(std::filesystem::exists(path));
}

TEST_CASE("ChatHistory::Clear wipes all entries") {
    std::string dir = make_tmpdir();
    xllama::ui::ChatHistory h(dir);
    h.LoadIndex();

    for (int i = 0; i < 3; ++i) {
        xllama::ui::Conversation c;
        c.id = xllama::ui::ChatHistory::NewId();
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

#endif // XLLAMA_UWP
