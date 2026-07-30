// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// The autopilot script contract. Every console gate is written in this
// language, so a rule that is wrong here is wrong for all nine of them — and
// until this file existed none of those rules were covered by anything: the
// driver is UWP-only, so the checks could only be exercised by deploying to a
// console and watching what broke.

#include "xllama/autopilot.h"

#include <doctest/doctest.h>

using xllama::autopilot_op_known;
using xllama::AutopilotAction;
using xllama::validate_autopilot_script;

namespace {

AutopilotAction op(const char* name) {
    AutopilotAction a;
    a.op = name;
    return a;
}

bool ok(const std::vector<AutopilotAction>& actions, bool store_sku = false) {
    std::string err;
    return validate_autopilot_script(actions, store_sku, err);
}

std::string why(const std::vector<AutopilotAction>& actions, bool store_sku = false) {
    std::string err;
    validate_autopilot_script(actions, store_sku, err);
    return err;
}

AutopilotAction send(const wchar_t* text) {
    AutopilotAction a = op("send");
    a.arg = text;
    a.has_text = true;
    return a;
}

} // namespace

TEST_CASE("autopilot: the op table matches what the driver implements") {
    for (const char* name :
         {"send", "new_chat", "load_chat", "set_model", "set_api", "set_routing", "set_sampling",
          "set_kv_reuse", "set_taesd", "set_system_prompt", "generate_image", "mark", "show_pane",
          "rate", "start_train", "train_status", "quit"})
        CHECK(autopilot_op_known(name));

    CHECK_FALSE(autopilot_op_known("sned"));  // typo
    CHECK_FALSE(autopilot_op_known("SEND"));  // case matters
    CHECK_FALSE(autopilot_op_known("send ")); // trailing space
    CHECK_FALSE(autopilot_op_known(""));
}

TEST_CASE("autopilot: an empty script is rejected") {
    CHECK_FALSE(ok({}));
    CHECK(why({}) == "empty 'actions'");
}

TEST_CASE("autopilot: an unknown op is caught before anything runs") {
    // The point of the whole file: this used to surface only when the driver
    // reached the action, by which time the earlier ones had already changed
    // settings.json, the chats folder and the selected model.
    std::vector<AutopilotAction> script = {op("new_chat"), send(L"hello"), op("teleport")};
    CHECK_FALSE(ok(script));
    CHECK(why(script) == "action 2: unknown op 'teleport'");
}

TEST_CASE("autopilot: payload-carrying ops reject an empty payload") {
    for (const char* name : {"send", "load_chat", "set_model", "generate_image", "rate"}) {
        CAPTURE(name);
        CHECK_FALSE(ok({op(name)}));
    }
    // ...and accept a non-empty one.
    AutopilotAction m = op("set_model");
    m.arg = L"lfm25-350m";
    CHECK(ok({m}));
}

TEST_CASE("autopilot: mark requires a label") {
    // Without one the host is waiting for a name that never arrives: the run
    // stalls until the mark times out rather than failing. A silent hang is the
    // worst of the available outcomes.
    CHECK_FALSE(ok({op("mark")}));
    CHECK(why({op("mark")}) == "action 0 mark: 'label' is required");

    AutopilotAction m = op("mark");
    m.label = L"01-chat-answer";
    CHECK(ok({m}));

    // The label lives in its own slot, not in the shared payload: putting it in
    // `arg` is what show_pane would have done by accident, and it must not pass.
    AutopilotAction wrong = op("mark");
    wrong.arg = L"01-chat-answer";
    CHECK_FALSE(ok({wrong}));
}

TEST_CASE("autopilot: show_pane needs a known pane and a label") {
    auto pane = [](const wchar_t* name, const wchar_t* label) {
        AutopilotAction a = op("show_pane");
        a.arg = name;
        a.label = label;
        return a;
    };
    for (const wchar_t* name : {L"settings", L"history", L"image"}) {
        CAPTURE(name);
        CHECK(ok({pane(name, L"03-pane")}));
    }
    // A pane with no coroutine behind it would open nothing and then time out
    // waiting for a dialog that is never coming.
    CHECK_FALSE(ok({pane(L"chat", L"03-pane")}));
    CHECK_FALSE(ok({pane(L"Settings", L"03-pane")})); // case matters
    CHECK_FALSE(ok({pane(L"", L"03-pane")}));
    CHECK(why({pane(L"chat", L"03-pane")}) ==
          "action 0 show_pane: 'name' must be one of settings|history|image");

    // Both are required, and the label is the half that would fail silently:
    // the pane would open, nobody would be waiting for it, and the action would
    // sit there until its timeout.
    CHECK_FALSE(ok({pane(L"settings", L"")}));
    CHECK(why({pane(L"settings", L"")}) == "action 0 show_pane: 'label' is required");
}

TEST_CASE("autopilot: set_routing accepts only 0..2") {
    for (int r : {0, 1, 2}) {
        AutopilotAction a = op("set_routing");
        a.routing = r;
        CAPTURE(r);
        CHECK(ok({a}));
    }
    for (int r : {-1, 3, 99}) {
        AutopilotAction a = op("set_routing");
        a.routing = r;
        CAPTURE(r);
        CHECK_FALSE(ok({a}));
    }
}

TEST_CASE("autopilot: set_sampling needs at least one key") {
    CHECK_FALSE(ok({op("set_sampling")}));

    AutopilotAction a = op("set_sampling");
    a.temperature = 0.7;
    CHECK(ok({a}));

    // Negative means "leave unchanged", so n_predict alone is a real request.
    AutopilotAction b = op("set_sampling");
    b.n_predict = 80;
    CHECK(ok({b}));
}

TEST_CASE("autopilot: top_p must be in (0, 1]") {
    auto with_top_p = [](double v) {
        AutopilotAction a = op("set_sampling");
        a.top_p = v;
        return std::vector<AutopilotAction>{a};
    };
    CHECK(ok(with_top_p(1.0)));   // the closed end
    CHECK(ok(with_top_p(0.001))); // just inside the open end
    CHECK_FALSE(ok(with_top_p(0.0)));
    CHECK_FALSE(ok(with_top_p(1.0001)));
    // Unset (negative) is not a value, so it must not trip the range check —
    // but on its own it is also not a sampling key.
    AutopilotAction unset = op("set_sampling");
    unset.top_k = 40;
    CHECK(ok({unset}));
}

TEST_CASE("autopilot: toggles require 'enabled' rather than defaulting to off") {
    for (const char* name : {"set_kv_reuse", "set_taesd"}) {
        CAPTURE(name);
        // Without the key, the default false would mean "disable" — an op that
        // did nothing would still be reported as applied.
        CHECK_FALSE(ok({op(name)}));

        AutopilotAction on = op(name);
        on.has_enabled = true;
        on.enabled = true;
        CHECK(ok({on}));

        AutopilotAction off = op(name);
        off.has_enabled = true;
        off.enabled = false;
        CHECK(ok({off}));
    }
}

TEST_CASE("autopilot: set_system_prompt requires the key, not a non-empty value") {
    CHECK_FALSE(ok({op("set_system_prompt")}));

    AutopilotAction clear = op("set_system_prompt");
    clear.has_text = true;
    clear.arg = L""; // clearing the prompt is a legitimate request
    CHECK(ok({clear}));
}

TEST_CASE("autopilot: set_api is rejected in the Store SKU, accepted otherwise") {
    AutopilotAction a = op("set_api");
    a.has_enabled = true;
    a.enabled = true;
    CHECK(ok({a}, /*store_sku=*/false));
    CHECK_FALSE(ok({a}, /*store_sku=*/true));
    CHECK(why({a}, true) == "action 0 set_api: LAN API not available in Store SKU");
}

TEST_CASE("autopilot: generate_image needs at least one step") {
    AutopilotAction a = op("generate_image");
    a.arg = L"pixel art robot";
    a.steps = 1;
    CHECK(ok({a}));
    a.steps = 0;
    CHECK_FALSE(ok({a}));
    a.steps = -3;
    CHECK_FALSE(ok({a}));
}

TEST_CASE("autopilot: a negative timeout is rejected") {
    AutopilotAction a = send(L"hi");
    a.timeout = std::chrono::seconds{-1};
    CHECK_FALSE(ok({a}));
    a.timeout = std::chrono::seconds{0}; // 0 means "per-op default"
    CHECK(ok({a}));
}

TEST_CASE("autopilot: the error names the offending action, not just the problem") {
    // Gate scripts are generated, sometimes long. "action 3" is the difference
    // between a fix and a hunt.
    std::vector<AutopilotAction> script = {op("new_chat"), send(L"a"), send(L"b"),
                                           op("set_sampling")};
    CHECK(why(script).rfind("action 3 set_sampling:", 0) == 0);
}

TEST_CASE("autopilot: a realistic gate script validates") {
    AutopilotAction model = op("set_model");
    model.arg = L"lfm25-350m";
    AutopilotAction routing = op("set_routing");
    routing.routing = 0;
    AutopilotAction kv = op("set_kv_reuse");
    kv.has_enabled = true;
    kv.enabled = true;
    AutopilotAction mark = op("mark");
    mark.label = L"01-chat-answer";

    CHECK(ok({model, routing, kv, op("new_chat"), send(L"What can you do?"), mark, op("quit")}));
}
