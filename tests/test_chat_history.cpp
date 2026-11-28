// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include "../uwp/chat-history.h"
#include "xllama/atomic_file.h"
#include <chrono>
#include <doctest/doctest.h>
#include <filesystem>
#include <fstream>
#ifdef __linux__
    #include <csignal>
    #include <sys/resource.h>
    #include <sys/wait.h>
    #include <unistd.h>
#endif

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

#ifdef __linux__
TEST_CASE("ChatHistory keeps the committed conversation after a partial write") {
    ChatDirectory temp;
    xllama::ui::ChatHistory history(temp.path.string());
    xllama::ui::Conversation chat{"limited", "saved", {}};
    xllama::ui::ChatMessage msg;
    msg.role = xllama::ui::MessageRole::Assistant;
    msg.content = "original reply";
    chat.messages.push_back(msg);
    REQUIRE(history.Save(chat));

    const auto child = fork();
    REQUIRE(child >= 0);
    if (child == 0) {
        const struct rlimit limit = {600, 600};
        if (signal(SIGXFSZ, SIG_IGN) == SIG_ERR || setrlimit(RLIMIT_FSIZE, &limit) != 0)
            _exit(2);
        chat.title = "interrupted";
        chat.messages.front().content.assign(65536, 'x');
        _exit(history.Save(chat) ? 1 : 0);
    }
    int status = 0;
    REQUIRE(waitpid(child, &status, 0) == child);
    REQUIRE(WIFEXITED(status));
    REQUIRE(WEXITSTATUS(status) == 0);
    xllama::ui::ChatHistory reopened(temp.path.string());
    reopened.LoadIndex();
    REQUIRE(reopened.Index().size() == 1);
    CHECK(reopened.Index().front().title == "saved");
    const auto loaded = reopened.Load(chat.id);
    REQUIRE(loaded.messages.size() == 1);
    CHECK(loaded.messages.front().content == "original reply");
}
#endif

TEST_CASE("ChatHistory retries reconciliation without replacing an unreadable index or chat") {
    ChatDirectory temp;
    xllama::ui::ChatHistory history(temp.path.string());
    REQUIRE(history.Save({"first", "keep", {}}));
    REQUIRE(history.Save({"second", "pending", {}}));
    const auto pending = temp.path / "second.json.pending";
    std::ofstream(pending).close();
    const auto index = temp.path / "index.json";
    const auto chat = temp.path / "second.json";
    for (const auto& input : {index, chat}) {
        auto backup = input;
        backup += ".backup";
        std::filesystem::rename(input, backup);
        std::ofstream(input) << "{truncated";
        xllama::ui::ChatHistory reopened(temp.path.string());
        CHECK_FALSE(reopened.LoadIndex());
        CHECK_FALSE(reopened.Save({"third", "wait", {}}));
        CHECK(std::filesystem::exists(pending));
        CHECK_FALSE(std::filesystem::exists(temp.path / "third.json"));
        std::filesystem::remove(input);
        std::filesystem::rename(backup, input);
    }
    REQUIRE(history.LoadIndex());
    CHECK(history.Index().size() == 2);
    CHECK_FALSE(std::filesystem::exists(pending));
}

TEST_CASE("ChatHistory reads titles truncated in older UTF-8 indexes") {
    ChatDirectory temp;
    std::filesystem::create_directories(temp.path);
    std::ofstream(temp.path / "index.json")
        << "[{\"id\":\"a\",\"title\":\"" << std::string(56, 'x') << char(0xc3) << "...\"}]";
    xllama::ui::ChatHistory history(temp.path.string());
    REQUIRE(history.LoadIndex());
    REQUIRE(history.Index().size() == 1);
    CHECK(history.Index().front().title == std::string(56, 'x') + "\xef\xbf\xbd...");
    CHECK(history.TitleFrom(std::string(56, 'x') + "\xc3\xa9" + "long") ==
          std::string(56, 'x') + "...");
}

TEST_CASE("ChatHistory rejects incomplete saves without replacing their committed index entries") {
    ChatDirectory temp;
    xllama::ui::ChatHistory history(temp.path.string());
    xllama::ui::Conversation chat{"interrupted", "saved title", {}};
    chat.messages.push_back({xllama::ui::MessageRole::User, "saved turn", 10, false, {}});
    REQUIRE(history.Save(chat));
    const auto path = temp.path / "interrupted.json";
    const auto marker = temp.path / "interrupted.json.pending";
    std::ifstream input(path, std::ios::binary);
    const std::string original((std::istreambuf_iterator<char>(input)), {});
    input.close();
    REQUIRE_FALSE(original.empty());
    std::ofstream(marker).close();
    REQUIRE(xllama::replace_file_bytes(path, original.substr(0, original.find("\"messages\""))));
    CHECK(history.Load(chat.id).id.empty());
    xllama::ui::ChatHistory reopened(temp.path.string());
    reopened.LoadIndex();
    REQUIRE(reopened.Index().size() == 1);
    CHECK(reopened.Index().front().title == "saved title");
    CHECK(reopened.Index().front().n_messages == 1);
    CHECK(std::filesystem::exists(marker));
    REQUIRE(xllama::replace_file_bytes(path, original));
    reopened.LoadIndex();
    CHECK_FALSE(std::filesystem::exists(marker));
    CHECK(reopened.Load(chat.id).messages.size() == 1);
    CHECK(reopened.Load("missing").id.empty());
}

TEST_CASE("ChatHistory preserves legacy messages, ignores extension fields and checks identity") {
    ChatDirectory temp;
    std::filesystem::create_directories(temp.path);
    xllama::ui::ChatHistory history(temp.path.string());
    const auto path = temp.path / "saved.json";
    const std::string legacy =
        R"({"messages":[{"role":"user","content":"a\u0020turn"}],"extra":{"revision":2},"title":"messages","id":"saved"})";
    REQUIRE(xllama::replace_file_bytes(path, legacy));
    const auto loaded = history.Load("saved");
    REQUIRE(loaded.id == "saved");
    CHECK(loaded.title == "messages");
    REQUIRE(loaded.messages.size() == 1);
    CHECK(loaded.messages.front().content == "a turn");
    CHECK_FALSE(loaded.messages.front().partial);
    CHECK(loaded.messages.front().feedback_label.empty());
    REQUIRE(xllama::replace_file_bytes(path, R"({"id":"different","title":"a","messages":[]})"));
    CHECK(history.Load("saved").id.empty());
    REQUIRE(xllama::replace_file_bytes(path, R"({"id":"saved","title":"a","messages":[]})"));
    CHECK(history.Load("saved").id == "saved");
}

TEST_CASE("ChatHistory reads titles that older releases cut inside a UTF-8 character") {
    ChatDirectory temp;
    xllama::ui::ChatHistory history(temp.path.string());
    const std::string prefix(56, 'a');
    const std::string text = prefix + "\xe2\x82\xac" + "padding";
    xllama::ui::Conversation chat{"legacy", prefix + "\xe2" + "...", {}};
    chat.messages.push_back({xllama::ui::MessageRole::User, text, 10, false, {}});
    REQUIRE(history.Save(chat));
    const auto path = temp.path / "legacy.json";
    std::ifstream input(path, std::ios::binary);
    std::string old((std::istreambuf_iterator<char>(input)), {});
    input.close();
    old.insert(old.find('\n'), "\r"); // the legacy Windows writer used text mode
    REQUIRE(xllama::replace_file_bytes(path, old));
    const auto loaded = history.Load(chat.id);
    REQUIRE(loaded.messages.size() == 1);
    CHECK(loaded.messages.front().content == text);
    CHECK(loaded.title == prefix + "...");
    CHECK(history.TitleFrom(text) == prefix + "...");
}

TEST_CASE("ChatHistory refuses unrepresentable metadata without replacing the existing index") {
    ChatDirectory temp;
    xllama::ui::ChatHistory history(temp.path.string());
    REQUIRE(history.Save({"current", "keep", {}}));
    const auto path = temp.path / "index.json";
    const auto marker = temp.path / "current.json.pending";
    for (const char* field :
         {R"("n_messages":-1)", R"("n_messages":1.5)", R"("n_messages":2147483648)",
          R"("last_modified":18446744073709551615)"}) {
        const std::string bad = std::string(R"([{"id":"other",)") + field + "}]";
        REQUIRE(xllama::replace_file_bytes(path, bad));
        std::ofstream(marker).close();
        CHECK_FALSE(history.LoadIndex());
        CHECK_FALSE(history.Save({"new", "wait", {}}));
        CHECK(std::filesystem::exists(marker));
        std::ifstream input(path, std::ios::binary);
        CHECK(std::string(std::istreambuf_iterator<char>(input), {}) == bad);
    }
    REQUIRE(xllama::replace_file_bytes(
        path, R"([{"id":"other","n_messages":2147483647,"last_modified":9223372036854775807}])"));
    REQUIRE(history.LoadIndex());
    REQUIRE(history.Index().size() == 2);
    CHECK(history.Index().front().n_messages == 2147483647);
    CHECK(history.Index().front().last_modified == INT64_MAX);
}

TEST_CASE("ChatHistory keeps failed deletions and removes only confirmed bodies on clear") {
    ChatDirectory temp;
    xllama::ui::ChatHistory history(temp.path.string());
    REQUIRE(history.Save({"keep", "original title", {}}));
    REQUIRE(history.Save({"remove", "delete me", {}}));
    const auto refusal = temp.path / "keep.json.deleting.tmp";
    std::filesystem::create_directory(refusal); // forbid staging the deletion marker on any OS
    const auto failed = history.Delete("keep");
    CHECK_FALSE(failed.all_removed);
    CHECK_FALSE(failed.index_pending);
    CHECK(failed.removed_ids.empty());
    CHECK(history.Index().size() == 2);
    const auto partial = history.Clear();
    CHECK_FALSE(partial.all_removed);
    CHECK_FALSE(partial.index_pending);
    REQUIRE(partial.removed_ids == std::vector<std::string>{"remove"});
    CHECK_FALSE(std::filesystem::exists(temp.path / "remove.json"));
    xllama::ui::ChatHistory reopened(temp.path.string());
    REQUIRE(reopened.LoadIndex());
    REQUIRE(reopened.Index().size() == 1);
    CHECK(reopened.Index()[0].id == "keep");
    CHECK(reopened.Load("keep").title == "original title");
    std::filesystem::remove(refusal);
    const auto retried = reopened.Clear();
    CHECK(retried.all_removed);
    CHECK_FALSE(retried.index_pending);
    CHECK(retried.removed_ids == std::vector<std::string>{"keep"});
}

TEST_CASE("ChatHistory replays a deleted body across failed index publication and missing files") {
    ChatDirectory temp;
    xllama::ui::ChatHistory history(temp.path.string());
    REQUIRE(history.Save({"survivor", "retain", {}}));
    REQUIRE(history.Save({"drop", "remove", {}}));
    const auto blocked = temp.path / "index.json.tmp";
    const auto journal = temp.path / "drop.json.deleting";
    std::filesystem::create_directory(blocked);
    const auto result = history.Delete("drop");
    REQUIRE(result.all_removed);
    CHECK(result.index_pending);
    CHECK(result.removed_ids == std::vector<std::string>{"drop"});
    REQUIRE(std::filesystem::exists(journal));
    CHECK_FALSE(std::filesystem::exists(temp.path / "drop.json"));
    xllama::ui::ChatHistory reopened(temp.path.string());
    CHECK_FALSE(reopened.LoadIndex());
    CHECK(std::filesystem::exists(journal));
    std::filesystem::remove(blocked);
    REQUIRE(reopened.LoadIndex());
    REQUIRE(reopened.Index().size() == 1);
    CHECK(reopened.Index().front().id == "survivor");
    CHECK_FALSE(std::filesystem::exists(journal));
    REQUIRE(reopened.Delete("drop").all_removed); // retry an already absent body
    REQUIRE(reopened.LoadIndex());
    CHECK(reopened.Index().size() == 1);
}

TEST_CASE("ChatHistory rolls back a deletion interrupted before removing its live body") {
    ChatDirectory temp;
    xllama::ui::ChatHistory history(temp.path.string());
    REQUIRE(history.Save({"keep", "keep", {}}));
    const auto journal = temp.path / "keep.json.deleting";
    std::ofstream(journal).close();
    REQUIRE(xllama::replace_file_bytes(
        temp.path / "index.json",
        R"([{"id":"keep","title":"keep","last_modified":123456,"n_messages":0}])"));
    REQUIRE(history.LoadIndex());
    REQUIRE(history.Index().size() == 1);
    CHECK(history.Index().front().last_modified == 123456);
    CHECK(history.Load("keep").id == "keep");
    CHECK_FALSE(std::filesystem::exists(journal));
}

TEST_CASE("ChatHistory keeps restored Unicode IDs and reserves its index and Windows filenames") {
    ChatDirectory temp;
    xllama::ui::ChatHistory history(temp.path.string());
    const std::string restored = "conversazione-\xc3\xa8.2";
    REQUIRE(history.Save({restored, "saved", {}}));
    REQUIRE(history.Load(restored).id == restored);
    for (const auto& id : {"INDEX", "Nul", "COM1", "LPT2.notes"}) {
        CHECK_FALSE(history.Save({id, "not a chat file", {}}));
        CHECK(history.Load(id).id.empty());
        CHECK_FALSE(history.Delete(id).all_removed);
    }
    REQUIRE(history.LoadIndex());
    REQUIRE(history.Index().size() == 1);
    CHECK(history.Index().front().id == restored);
    CHECK(history.Delete(restored).all_removed);
}
