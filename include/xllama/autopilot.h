// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// The autopilot script contract: what a valid action list is, checked before
// any of it runs.
//
// The driver itself (MainPageController::ApRun) can only live on UWP — it
// dispatches to XAML on the UI thread. The *contract* does not need XAML, and
// keeping it here makes it host-testable, which is the only way this subsystem
// gets tested at all: every console gate is scripted through it.
#pragma once

#include <chrono>
#include <string>
#include <vector>

namespace xllama {

// One parsed autopilot action. Filled by the JSON parser on the UWP side; kept
// WinRT-free so validation and tests need neither XAML nor a console.
struct AutopilotAction {
    std::string op;   // send|new_chat|load_chat|set_model|set_api|set_routing|set_sampling|
                      // set_kv_reuse|set_taesd|set_system_prompt|generate_image|mark|rate|
                      // start_train|train_status|quit
    std::wstring arg; // text / id / model name / image prompt / mark label / rate label
    int steps{1};     // generate_image
    unsigned seed{42};
    bool enabled{false};             // set_api / set_kv_reuse / set_taesd
    bool has_enabled{false};         // 'enabled' was present (set_kv_reuse and set_taesd
                                     // require it: the default would silently mean "disable")
    bool has_text{false};            // 'text' was present (set_system_prompt requires it;
                                     // clearing the prompt to "" is a legitimate value)
    int port{11434};                 // set_api
    int routing{-1};                 // set_routing (0=CPU, 1=GPU, 2=auto)
    double temperature{-1};          // set_sampling; negative = leave unchanged
    double top_p{-1};                //
    int top_k{-1};                   //
    double repetition_penalty{-1};   //
    int n_predict{-1};               //
    std::chrono::seconds timeout{0}; // 0 = per-op default
};

// Every op the driver implements. Also the list check-coherence.py holds the
// documentation side of.
bool autopilot_op_known(const std::string& op);

// Check the whole list before the first action runs. Returns false + err.
//
// Why up front rather than per action, which is where these checks used to
// live: the driver mutates real, persistent state — settings.json, the chats
// folder, the selected model. A bad op name or an out-of-range value in action
// 7 was previously discovered after actions 0..6 had already been applied, so a
// typo in a gate script left the console in a half-scripted state and reported
// a failure that reads exactly like a product failure. Nothing here touches the
// device, so there is no reason to find out late.
//
// |store_sku| exists because set_api is rejected in the Store SKU, and taking
// it as a parameter rather than reading a macro is what makes both branches
// testable on the host.
//
// Runtime conditions stay in the driver on purpose: whether a chat file exists,
// whether a port binds, whether the UI is busy. Those are not properties of the
// script.
bool validate_autopilot_script(const std::vector<AutopilotAction>& actions, bool store_sku,
                               std::string& err);

} // namespace xllama
