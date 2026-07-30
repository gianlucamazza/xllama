// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include "xllama/autopilot.h"

#include <algorithm>
#include <array>

namespace xllama {

namespace {

// Ops the driver implements. Adding one here without adding a branch to ApRun
// makes a script that validates and then fails at run time, so the two lists
// belong together; check-coherence.py asserts the documentation side.
constexpr std::array<const char*, 17> kOps = {
    "send",           "new_chat",     "load_chat",    "set_model", "set_api",
    "set_routing",    "set_sampling", "set_kv_reuse", "set_taesd", "set_system_prompt",
    "generate_image", "mark",         "show_pane",    "rate",      "start_train",
    "train_status",   "quit"};

// Panes show_pane can open. Each is a ContentDialog with an existing coroutine
// behind it; the op reuses those rather than rebuilding any UI.
constexpr std::array<const char*, 3> kPanes = {"settings", "history", "image"};

// The pane name is an ASCII keyword, so a byte-wise narrowing is enough and
// keeps this header-free of platform string helpers.
std::string wide_to_narrow_ascii(const std::wstring& w) {
    std::string s;
    s.reserve(w.size());
    for (wchar_t c : w)
        s.push_back(c < 128 ? static_cast<char>(c) : '?');
    return s;
}

std::string where(size_t i, const std::string& op) {
    return "action " + std::to_string(i) + " " + op + ": ";
}

} // namespace

bool autopilot_op_known(const std::string& op) {
    return std::find_if(kOps.begin(), kOps.end(), [&](const char* k) { return op == k; }) !=
           kOps.end();
}

bool validate_autopilot_script(const std::vector<AutopilotAction>& actions, bool store_sku,
                               std::string& err) {
    if (actions.empty()) {
        err = "empty 'actions'";
        return false;
    }

    for (size_t i = 0; i < actions.size(); ++i) {
        const AutopilotAction& a = actions[i];

        if (!autopilot_op_known(a.op)) {
            err = "action " + std::to_string(i) + ": unknown op '" + a.op + "'";
            return false;
        }

        if (a.op == "send" || a.op == "load_chat" || a.op == "set_model" ||
            a.op == "generate_image" || a.op == "rate") {
            // Every one of these carries its payload in the same slot, and an
            // empty one is never meaningful: an empty send generates from
            // nothing, an empty model name clears the selection, an empty chat
            // id cannot resolve to a file.
            if (a.arg.empty()) {
                err = where(i, a.op) + "missing or empty payload";
                return false;
            }
        }

        // Both rendez-vous ops need a label: the host matches it to decide which
        // state a screenshot belongs to, and waiting for one that never arrives
        // is a silent hang rather than an error.
        if ((a.op == "mark" || a.op == "show_pane") && a.label.empty()) {
            err = where(i, a.op) + "'label' is required";
            return false;
        }

        if (a.op == "show_pane") {
            const std::string pane = wide_to_narrow_ascii(a.arg);
            if (std::find_if(kPanes.begin(), kPanes.end(),
                             [&](const char* k) { return pane == k; }) == kPanes.end()) {
                err = where(i, a.op) + "'name' must be one of settings|history|image";
                return false;
            }
        }

        if (a.op == "set_routing" && (a.routing < 0 || a.routing > 2)) {
            err = where(i, a.op) + "'routing' must be 0..2";
            return false;
        }

        if (a.op == "set_sampling") {
            if (a.temperature < 0 && a.top_p < 0 && a.top_k < 0 && a.repetition_penalty < 0 &&
                a.n_predict < 0) {
                err = where(i, a.op) +
                      "no sampling key (temperature/top_p/top_k/repetition_penalty/n_predict)";
                return false;
            }
            // The others degrade safely downstream (temperature <= 0 falls back
            // to greedy, top_k <= 0 disables filtering, repetition_penalty <= 0
            // is skipped, n_predict is capped), but top_p outside (0, 1]
            // violates the GenAI contract.
            if (a.top_p >= 0 && !(a.top_p > 0.0 && a.top_p <= 1.0)) {
                err = where(i, a.op) + "'top_p' must be in (0, 1]";
                return false;
            }
        }

        // 'enabled' defaulting to false would make a missing key mean "disable",
        // so an op that silently did nothing would still be reported as applied.
        if ((a.op == "set_kv_reuse" || a.op == "set_taesd") && !a.has_enabled) {
            err = where(i, a.op) + "'enabled' is required";
            return false;
        }

        // Clearing the system prompt to "" is legitimate, so presence of the key
        // is what is required, not a non-empty value.
        if (a.op == "set_system_prompt" && !a.has_text) {
            err = where(i, a.op) + "'text' is required";
            return false;
        }

        if (a.op == "set_api" && store_sku) {
            err = where(i, a.op) + "LAN API not available in Store SKU";
            return false;
        }

        if (a.op == "generate_image" && a.steps < 1) {
            err = where(i, a.op) + "'steps' must be >= 1";
            return false;
        }

        if (a.timeout.count() < 0) {
            err = where(i, a.op) + "'timeout_s' must not be negative";
            return false;
        }
    }
    return true;
}

} // namespace xllama
