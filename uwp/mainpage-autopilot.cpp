// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// MainPageController: autopilot interpreter. Split out of MainPage.cpp (#549); shared
// includes and helpers are in mainpage-internal.h.

#include "mainpage-internal.h"

#ifdef XLLAMA_UWP

namespace xllama {

// ---------------------------------------------------------------------------
// Autopilot — scripted validation of the real XAML UI.
//
// Dev Mode gives the console no working text-input path (the Xbox companion
// app's remote keyboard does not connect to a Dev-Mode console), so the §2
// routing A/B, §7c TAESD and GGUF-chat validations could only be done by hand.
// This driver replays a JSON action list against the same controller methods
// the buttons call (StartInference / NewChat / LoadConversation / StartDiffusion),
// on the UI thread, from a background MTA thread — no UI code is duplicated.
//
// Trigger: LocalState\autopilot.flag (consumed). Script: LocalState\autopilot.json.
// Result: LocalState\autopilot-done.txt = "ok" | "error: <detail>". Progress is
// logged with an [autopilot] prefix (flushed per line) so a hard crash still
// leaves the in-flight action identifiable.
// ---------------------------------------------------------------------------

// Parse autopilot.json into an action list. Returns false + err on shape error.
bool MainPageController::ApParseScript(const std::string& json_utf8, std::vector<ApAction>& out,
                                       std::chrono::seconds& total_cap, std::string& err) {
    using winrt::Windows::Data::Json::JsonObject;
    JsonObject root{nullptr};
    if (!JsonObject::TryParse(::xllama::utf8_to_wstring(json_utf8), root)) {
        err = "not valid JSON";
        return false;
    }
    total_cap = std::chrono::seconds((int64_t)root.GetNamedNumber(L"total_timeout_s", 1800));
    if (!root.HasKey(L"actions")) {
        err = "no 'actions' array";
        return false;
    }
    for (auto const& item : root.GetNamedArray(L"actions")) {
        auto obj = item.GetObject();
        ApAction a;
        a.op = ::xllama::wstring_to_utf8(std::wstring(obj.GetNamedString(L"op", L"").c_str()));
        if (a.op.empty()) {
            err = "action without 'op'";
            return false;
        }
        // The rendez-vous label has its own slot: show_pane carries a pane
        // "name" AND a "label", and the loop below stops at the first key it
        // finds — "name" first — so sharing one slot would have dropped the
        // label in silence and left the host waiting for a mark that never
        // matched.
        a.label = std::wstring(obj.GetNamedString(L"label", L"").c_str());
        // Single payload slot: text (send) / id (load_chat) / name (set_model,
        // show_pane) / prompt (generate_image).
        for (auto key : {L"text", L"id", L"name", L"prompt"}) {
            if (obj.HasKey(key)) {
                a.arg = std::wstring(obj.GetNamedString(key, L"").c_str());
                break;
            }
        }
        // Same contract as the Image dialog and the LAN API; clamped before any
        // double->int conversion (xllama/diffusion/request.h).
        a.steps = ::xllama::diffusion::interactive_steps(
            obj.GetNamedNumber(L"steps", ::xllama::diffusion::kDefaultSteps));
        a.seed = ::xllama::diffusion::request_seed(
            obj.GetNamedNumber(L"seed", ::xllama::diffusion::kDefaultSeed));
        a.has_enabled = obj.HasKey(L"enabled");
        a.has_text = obj.HasKey(L"text");
        a.enabled = obj.GetNamedBoolean(L"enabled", false);
        a.port = (int)obj.GetNamedNumber(L"port", 11434);
        a.routing = (int)obj.GetNamedNumber(L"routing", -1);
        a.temperature = obj.GetNamedNumber(L"temperature", -1);
        a.top_p = obj.GetNamedNumber(L"top_p", -1);
        a.top_k = (int)obj.GetNamedNumber(L"top_k", -1);
        a.repetition_penalty = obj.GetNamedNumber(L"repetition_penalty", -1);
        a.n_predict = (int)obj.GetNamedNumber(L"n_predict", -1);
        int t = (int)obj.GetNamedNumber(L"timeout_s", 0);
        a.timeout = std::chrono::seconds(t);
        out.push_back(std::move(a));
    }
    // Everything the script can be wrong about that does not require the device,
    // decided here — before ApRun applies the first action. The driver mutates
    // persistent state (settings.json, the chats folder, the selected model), so
    // a bad op name or an out-of-range value found at action 7 used to leave the
    // console half-scripted and report it as a product failure.
    // Runtime conditions stay in ApRun: whether a chat file exists, whether a
    // port binds, whether the UI is busy. Those are not properties of a script.
    #ifdef XLLAMA_STORE_SKU
    constexpr bool kStoreSku = true;
    #else
    constexpr bool kStoreSku = false;
    #endif
    return ::xllama::validate_autopilot_script(out, kStoreSku, err);
}

// Record the dialog that is going on screen, and clear it when it leaves.
//
// Two things depend on this. The autopilot needs a handle to Hide() a pane it
// opened for a screenshot — the Show* coroutines keep their ContentDialog in a
// local, so nothing outside the coroutine frame can reach it. And m_pane_open
// is the guard against opening a second one: XAML permits exactly one, and the
// throw lands inside a fire_and_forget, whose unhandled_exception() calls
// std::terminate() — a silent process death with no autopilot-done.txt written.
//
// The handlers hold a weak_ptr, not the controller: the dialog is stored in a
// member, so a strong capture would close a cycle between them.
void MainPageController::ApTrackDialog(
    winrt::Windows::UI::Xaml::Controls::ContentDialog const& dlg) {
    std::weak_ptr<MainPageController> weak = weak_from_this();
    dlg.Opened([weak](winrt::Windows::UI::Xaml::Controls::ContentDialog const& sender,
                      winrt::Windows::UI::Xaml::Controls::ContentDialogOpenedEventArgs const&) {
        if (auto self = weak.lock()) {
            self->m_ap_dialog = sender;
            self->m_pane_open.store(true);
        }
    });
    dlg.Closed([weak](winrt::Windows::UI::Xaml::Controls::ContentDialog const&,
                      winrt::Windows::UI::Xaml::Controls::ContentDialogClosedEventArgs const&) {
        if (auto self = weak.lock()) {
            self->m_ap_dialog = nullptr;
            self->m_pane_open.store(false);
        }
    });
}

void MainPageController::ApDispatchSync(std::function<void()> fn) {
    // RunAsync(...).get() blocks the calling MTA thread until the lambda has run
    // on the UI thread. Legal from MTA (would deadlock/assert on an STA pump).
    m_root.Dispatcher()
        .RunAsync(winrt::Windows::UI::Core::CoreDispatcherPriority::Normal,
                  [fn = std::move(fn)]() { fn(); })
        .get();
}

bool MainPageController::ApWaitAtomic(std::atomic<bool>& flag, bool want,
                                      std::chrono::seconds timeout) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
        if (flag.load() == want)
            return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
    }
    return flag.load() == want;
}

void MainPageController::StartAutopilotIfRequested() {
    std::wstring flag = local_wpath(L"autopilot.flag");
    if (GetFileAttributesW(flag.c_str()) == INVALID_FILE_ATTRIBUTES)
        return;
    _wremove(flag.c_str()); // consume before run (same as bench/diffuse flags)
    log_output("[autopilot] flag detected\n");

    auto self = shared_from_this(); // strong: worker-thread pattern (see StartDiffusion)
    std::thread([self]() {
        winrt::init_apartment(); // MTA (same as the diffuse worker)
        std::string result = "ok";
        try {
            std::string json = ::xllama::read_local_text("autopilot.json");
            if (json.empty())
                throw std::runtime_error("autopilot.json missing or empty");
            std::vector<ApAction> actions;
            std::chrono::seconds total_cap{1800};
            std::string perr;
            if (!ApParseScript(json, actions, total_cap, perr))
                throw std::runtime_error("bad autopilot.json: " + perr);
            self->ApRun(std::move(actions), total_cap);
        } catch (const winrt::hresult_error& e) {
            result = "error: hresult 0x" + std::to_string((unsigned)e.code().value) + " " +
                     ::xllama::wstring_to_utf8(std::wstring(e.message().c_str()));
        } catch (const std::exception& e) {
            result = std::string("error: ") + e.what();
        } catch (...) {
            result = "error: unknown exception";
        }
        // ApRun writes the marker itself on a normal finish (so 'quit' can exit
        // before we get here); only write it if it didn't.
        if (GetFileAttributesW(local_wpath(L"autopilot-done.txt").c_str()) ==
            INVALID_FILE_ATTRIBUTES)
            ::xllama::write_local_text("autopilot-done.txt", result);
        log_output("[autopilot] driver exit: " + result + "\n");
    }).detach();
}

void MainPageController::ApRun(std::vector<ApAction> actions, std::chrono::seconds total_cap) {
    const auto total_deadline = std::chrono::steady_clock::now() + total_cap;
    const std::chrono::seconds kGrace{30};

    // Gate on the model being ready (first-launch download may be in flight).
    if (!ApWaitAtomic(m_model_ready, true, std::chrono::seconds{600}))
        throw std::runtime_error("model not ready after 600s");

    auto not_running = [&]() {
        return ApWaitAtomic(m_is_running, false, kGrace) &&
               ApWaitAtomic(m_diffuse_running, false, kGrace);
    };

    // Park on a named UI state until the host has grabbed it, then carry on.
    //
    // The host cannot see this process's UI and the app cannot reach the Device
    // Portal, so "screenshot the Settings pane" used to be a race between an
    // autopilot action and a host-side sleep. Here the app publishes the label
    // and blocks; the host polls for the file, takes its shot, and deletes the
    // file to release us. Nothing is timed, so nothing is guessed.
    //
    // A timeout releases the action rather than failing the script: an
    // unattended capture run should still finish, and the missing screenshot is
    // visible on the host anyway. Shared by `mark` and `show_pane` rather than
    // written twice — every copy in this codebase has eventually disagreed with
    // the other, and always silently.
    auto rendezvous = [&](const std::string& label, std::chrono::seconds timeout) {
        const std::wstring mark_path = local_wpath(L"autopilot-mark.txt");
        ::xllama::write_local_text("autopilot-mark.txt", label);
        // The wait below reads "file gone" as "the host has taken its shot". A
        // write that failed produces the same absence, so without this check the
        // action would sail through and the log would claim a release that never
        // happened — a false record of evidence that does not exist.
        if (GetFileAttributesW(mark_path.c_str()) == INVALID_FILE_ATTRIBUTES)
            throw std::runtime_error("mark '" + label + "': could not write autopilot-mark.txt");
        log_output("[autopilot] mark '" + label + "' waiting for host\n");
        const auto t = timeout.count() > 0 ? timeout : std::chrono::seconds{120};
        const auto deadline = std::chrono::steady_clock::now() + t;
        while (std::chrono::steady_clock::now() < deadline) {
            if (GetFileAttributesW(mark_path.c_str()) == INVALID_FILE_ATTRIBUTES) {
                log_output("[autopilot] mark '" + label + "' released\n");
                return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
        }
        _wremove(mark_path.c_str());
        log_output("[autopilot] mark '" + label + "' timed out, continuing\n");
    };

    for (size_t i = 0; i < actions.size(); ++i) {
        const ApAction& a = actions[i];
        if (std::chrono::steady_clock::now() > total_deadline)
            throw std::runtime_error("total timeout");
        log_output("[autopilot] action " + std::to_string(i) + " " + a.op + " start\n");

        if (a.op == "send") {
            if (!not_running())
                throw std::runtime_error("action " + std::to_string(i) + " send: busy");
            std::wstring text = a.arg;
            bool started = false;
            ApDispatchSync([this, text, &started]() { started = StartInference(text); });
            if (!started)
                throw std::runtime_error("action " + std::to_string(i) + " send: busy");
            auto t = a.timeout.count() > 0 ? a.timeout : std::chrono::seconds{300};
            if (!ApWaitAtomic(m_is_running, false, t)) {
                m_abort.store(true);
                ApWaitAtomic(m_is_running, false, kGrace);
                throw std::runtime_error("action " + std::to_string(i) + " send: timeout");
            }
            // Fence: a status probe dispatched now runs AFTER the completion
            // lambda (which does SetRunning(false) then SaveCurrentConversation),
            // so the chat JSON is on disk and the status text is final.
            std::wstring status;
            ApDispatchSync(
                [this, &status]() { status = std::wstring(m_statusText.Text().c_str()); });
            if (status.rfind(L"! ", 0) == 0)
                throw std::runtime_error("action " + std::to_string(i) +
                                         " send: " + ::xllama::wstring_to_utf8(status.substr(2)));
        } else if (a.op == "new_chat") {
            if (!not_running())
                throw std::runtime_error("action " + std::to_string(i) + " new_chat: busy");
            ApDispatchSync([this]() { NewChat(); });
        } else if (a.op == "load_chat") {
            if (!not_running())
                throw std::runtime_error("action " + std::to_string(i) + " load_chat: busy");
            std::string id = ::xllama::wstring_to_utf8(a.arg);
            std::wstring path = local_wpath((L"chats\\" + a.arg + L".json").c_str());
            if (GetFileAttributesW(path.c_str()) == INVALID_FILE_ATTRIBUTES)
                throw std::runtime_error("action " + std::to_string(i) + " load_chat: chat '" + id +
                                         "' not found");
            ApDispatchSync([this, id]() { LoadConversation(id); });
        } else if (a.op == "set_model") {
            std::wstring name = a.arg;
            // Same as ShowSettings' Save path, plus clear the sticky routed model
            // so the next send re-decides the EP for the new model. Catalogue
            // n_predict (thinking 1024) applies here so gates need not restate it.
            bool changed = false;
            ApDispatchSync([this, name, &changed]() {
                changed = SelectModel(name);
                m_active_model.clear();
                ApplyCatalogueModelKnobs(name);
                SaveSettings();
            });
            if (changed) {
                const auto deadline =
                    std::chrono::steady_clock::now() +
                    (a.timeout.count() > 0 ? a.timeout : std::chrono::seconds{300});
                while (!m_model_ready.load()) {
                    std::wstring status;
                    ApDispatchSync([this, &status] { status = m_statusText.Text().c_str(); });
                    if (status.rfind(L"! ", 0) == 0)
                        throw std::runtime_error("set_model: " +
                                                 ::xllama::wstring_to_utf8(status.substr(2)));
                    if (std::chrono::steady_clock::now() >= deadline)
                        throw std::runtime_error("set_model: model readiness timeout");
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                }
            }
        } else if (a.op == "set_api") {
    #ifndef XLLAMA_STORE_SKU
            if (a.enabled && !::xllama::api::port_bindable(a.port))
                throw std::runtime_error("action " + std::to_string(i) + " set_api: invalid port");
            ApDispatchSync(
                [this, enabled = a.enabled, port = a.port]() { ApplyApiSettings(enabled, port); });
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(15);
            while (std::chrono::steady_clock::now() < deadline) {
                const auto state = ::xllama::api::server_status().state;
                if ((a.enabled && state == ::xllama::api::ServerState::Running) ||
                    (!a.enabled && state == ::xllama::api::ServerState::Stopped))
                    break;
                if (state == ::xllama::api::ServerState::Error)
                    throw std::runtime_error("action " + std::to_string(i) +
                                             " set_api: listener error");
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            const auto state = ::xllama::api::server_status().state;
            if ((a.enabled && state != ::xllama::api::ServerState::Running) ||
                (!a.enabled && state != ::xllama::api::ServerState::Stopped))
                throw std::runtime_error("action " + std::to_string(i) +
                                         " set_api: lifecycle timeout");
    #endif
        } else if (a.op == "set_routing") {
            // Same semantics as the Settings panel: per-conversation, applies from
            // the next new/loaded chat (m_active_model stays sticky meanwhile).
            ApDispatchSync([this, r = a.routing]() {
                m_routing = r;
                SaveSettings();
            });
        } else if (a.op == "set_sampling") {
            ApDispatchSync([this, &a]() {
                // The user's layer, as the Settings dialog sets it.
                if (a.temperature >= 0)
                    m_sampling.temperature = (float)a.temperature;
                if (a.top_p >= 0)
                    m_sampling.top_p = (float)a.top_p;
                if (a.top_k >= 0)
                    m_sampling.top_k = a.top_k;
                if (a.repetition_penalty >= 0)
                    m_sampling.repetition_penalty = (float)a.repetition_penalty;
                if (a.n_predict >= 0)
                    m_n_predict = a.n_predict;
                SaveSettings();
            });
        } else if (a.op == "set_kv_reuse") {
            // Require the key: 'enabled' defaults to false, so a typo'd or missing
            // key would silently assert "KV reuse off" and pass.
            ApDispatchSync([this, on = a.enabled]() {
                m_kv_reuse = on;
                SaveSettings();
            });
        } else if (a.op == "set_taesd") {
            ApDispatchSync([this, on = a.enabled]() {
                m_diffuse_taesd = on;
                SaveSettings();
            });
        } else if (a.op == "set_system_prompt") {
            ApDispatchSync([this, text = ::xllama::wstring_to_utf8(a.arg)]() {
                m_system_prompt = text;
                SaveSettings();
            });
        } else if (a.op == "wait_idle") {
            // Session preload after launch holds the process-wide compute slot
            // without setting m_is_running. generate_image still fails busy if
            // it arrives during that window; this parks until the atomics stay
            // idle across three 250 ms polls. Do not try_acquire here: a probe
            // can win the slot and make PreloadSessionAsync skip the load.
            // The busy guard on the next action is unchanged.
            auto t = a.timeout.count() > 0 ? a.timeout : std::chrono::seconds{120};
            const auto deadline = std::chrono::steady_clock::now() + t;
            constexpr int kIdlePolls = 3;
            int idle_streak = 0;
            auto flags_idle = [this]() {
                return !m_is_running.load() && !m_diffuse_running.load() &&
                       !m_train_running.load() &&
                       !::xllama::session_hub().preloading.load(std::memory_order_acquire);
            };
            while (std::chrono::steady_clock::now() < deadline) {
                if (flags_idle() && flags_idle())
                    ++idle_streak;
                else
                    idle_streak = 0;
                if (idle_streak >= kIdlePolls)
                    break;
                std::this_thread::sleep_for(std::chrono::milliseconds(250));
            }
            if (idle_streak < kIdlePolls)
                throw std::runtime_error("action " + std::to_string(i) + " wait_idle: timeout");
        } else if (a.op == "generate_image") {
            if (!not_running())
                throw std::runtime_error("action " + std::to_string(i) + " generate_image: busy");
            auto compute_permit = ::xllama::try_acquire_compute_job();
            if (!compute_permit)
                throw std::runtime_error("action " + std::to_string(i) + " generate_image: busy");
            // No default prompt. Inventing one when the script did not supply
            // it is the defect class already paid for in the bench (#205): the
            // run succeeds and produces a real-looking image nobody asked for.
            // validate_autopilot_script guarantees a non-empty prompt.
            ::xllama::write_local_text("prompt.txt", ::xllama::wstring_to_utf8(a.arg));
            ::xllama::write_local_text("diffuse-steps.txt", std::to_string(a.steps));
            ::xllama::write_local_text("diffuse-seed.txt", std::to_string(a.seed));
            ::xllama::write_local_text("diffuse-model.txt", "sd-turbo-fp16");
            ApDispatchSync([this, seed = a.seed, compute_permit]() {
                m_last_diffuse_seed = seed;
                StartDiffusion(compute_permit);
            });
            auto t = a.timeout.count() > 0 ? a.timeout : std::chrono::seconds{600};
            if (!ApWaitAtomic(m_diffuse_running, false, t)) {
                ::xllama::write_local_text("diffuse-cancel.flag", "cancel");
                ApWaitAtomic(m_diffuse_running, false, kGrace);
                throw std::runtime_error("action " + std::to_string(i) +
                                         " generate_image: timeout");
            }
            std::string stage = ::xllama::read_local_text("diffuse-progress.txt");
            if (stage != "done")
                throw std::runtime_error("action " + std::to_string(i) +
                                         " generate_image: stage=" + stage);
        } else if (a.op == "gpu_selftest" || a.op == "dump_logits") {
            if (!not_running())
                throw std::runtime_error("action " + std::to_string(i) + " " + a.op + ": busy");
            auto compute_permit = ::xllama::try_acquire_compute_job();
            if (!compute_permit)
                throw std::runtime_error("action " + std::to_string(i) + " " + a.op + ": busy");
            // Keep XAML/compositor in the measurement process, but never retain
            // a second model beside the diagnostic's temporary inference model.
            {
                auto& hub = ::xllama::session_hub();
                std::lock_guard<std::mutex> lock(hub.mtx);
                hub.reset_locked();
            }
            std::string probe_model;
            ApDispatchSync([this, &probe_model]() {
                probe_model = ::xllama::wstring_to_utf8(m_model_filename);
                m_kv_valid = false;
                SetRunning(true);
            });
            log_output("[autopilot] " + a.op + ": ui-process start\n");
            try {
                if (a.op == "gpu_selftest") {
                    _wremove(local_wpath(L"d3d12be-result.csv.done").c_str());
                    _wremove(local_wpath(L"d3d12be-result.csv").c_str());
                    ::xllama::bridge::run_d3d12_selftest();
                } else {
                    ::xllama::write_local_text("model.txt", probe_model);
                    ::xllama::write_local_text("prompt.txt", ::xllama::wstring_to_utf8(a.arg));
                    _wremove(local_wpath(L"logits.done").c_str());
                    _wremove(local_wpath(L"logits.bin").c_str());
                    _wremove(local_wpath(L"logits.bin.json").c_str());
                    ::xllama::bridge::run_logits();
                    if (::xllama::read_local_text("logits.done") != "ok")
                        throw std::runtime_error("GPU logit diagnostic failed");
                }
            } catch (...) {
                ApDispatchSync([this]() { SetRunning(false); });
                throw;
            }
            ApDispatchSync([this]() { SetRunning(false); });
            // Completion means execution finished; the host must still check
            // every numerical CSV row / compare the fresh binary distributions.
            log_output("[autopilot] " + a.op + ": ui-process complete\n");
        } else if (a.op == "rate") {
            // Preference capture (training hybrid loop): append last user+assistant
            // turn to LocalState\training\samples.jsonl with label like|dislike|...
            std::string label = ::xllama::wstring_to_utf8(a.arg);
            if (label.empty())
                label = "like";
            if (!::xllama::preference_label_valid(label))
                throw std::runtime_error("action " + std::to_string(i) + " rate: bad label '" +
                                         label + "'");
            bool saved = false;
            std::string rate_err;
            ApDispatchSync([this, &label, &saved, &rate_err]() {
                for (size_t index = m_current.messages.size(); index-- > 0;) {
                    if (m_current.messages[index].role == xllama::ui::MessageRole::Assistant &&
                        !m_current.messages[index].partial) {
                        saved = SubmitFeedback(index, label, {}, &rate_err);
                        return;
                    }
                }
                rate_err = "no user/assistant turn to capture";
            });
            if (!saved)
                throw std::runtime_error("action " + std::to_string(i) + " rate: " + rate_err);
            log_output("[autopilot] rate label=" + label + " appended preference sample\n");
        } else if (a.op == "start_train") {
            // Phase 11 (#116): in-process personalize; non-blocking start, then wait.
            if (!not_running() || m_train_running.load())
                throw std::runtime_error("action " + std::to_string(i) + " start_train: busy");
            ApDispatchSync([this]() { StartPersonalizeTrain(); });
            if (!m_train_running.load()) {
                // StartPersonalizeTrain failed preflight synchronously on UI thread.
                const std::string done = ::xllama::read_local_text("training\\result.done");
                throw std::runtime_error("action " + std::to_string(i) +
                                         " start_train: did not start (preflight)");
            }
            auto t = a.timeout.count() > 0 ? a.timeout : std::chrono::seconds{7200};
            if (!ApWaitAtomic(m_train_running, false, t)) {
                m_train_abort.store(true);
                ApWaitAtomic(m_train_running, false, kGrace);
                throw std::runtime_error("action " + std::to_string(i) + " start_train: timeout");
            }
            const std::string done = ::xllama::read_local_text("training\\result.done");
            if (::xllama::parse_train_result_done(done) != "ok")
                throw std::runtime_error("action " + std::to_string(i) +
                                         " start_train: result.done=" + done);
        } else if (a.op == "train_status") {
            const std::string done = ::xllama::read_local_text("training\\result.done");
            const std::string prog = ::xllama::read_local_text("training\\progress.json");
            const std::string state = m_train_running.load()
                                          ? "running"
                                          : (::xllama::parse_train_result_done(done).empty()
                                                 ? "idle"
                                                 : ::xllama::parse_train_result_done(done));
            ::xllama::write_local_text("autopilot-train-status.txt",
                                       "state=" + state + "\nresult_done=" + done +
                                           "\nprogress=" + prog + "\n");
            log_output("[autopilot] train_status state=" + state + "\n");
        } else if (a.op == "mark") {
            if (!not_running())
                throw std::runtime_error("action " + std::to_string(i) + " mark: busy");
            rendezvous(::xllama::wstring_to_utf8(a.label), a.timeout);
        } else if (a.op == "show_pane") {
            // Screenshot a pane that is not the chat view. Settings, History and
            // the image viewer are all ContentDialogs, so `mark` alone cannot
            // reach them — generate_image completes and the chat view only says
            // "Image ready", which is what a frame taken there shows.
            //
            // Open and close in ONE action, deliberately: a dialog left up would
            // break every gate that runs afterwards, and no separate close op can
            // be relied on to run after a failure.
            if (!not_running())
                throw std::runtime_error("action " + std::to_string(i) + " show_pane: busy");
            if (m_pane_open.load())
                throw std::runtime_error(
                    "action " + std::to_string(i) +
                    " show_pane: a dialog is already open — XAML allows exactly one, and "
                    "opening a second throws inside a fire_and_forget, which terminates the "
                    "process without writing autopilot-done.txt");
            const std::string pane = ::xllama::wstring_to_utf8(a.arg);
            ApDispatchSync([this, pane]() {
                if (pane == "settings")
                    ShowSettings();
                else if (pane == "history")
                    ShowHistory();
                else
                    ShowImageDialog();
            });
            // Fence on the Opened event, NOT on ApDispatchSync returning.
            // ApDispatchSync unblocks at the coroutine's first suspension point,
            // and for ShowImageDialog that is a file read (GetFileAsync on the
            // last generated PNG) which happens BEFORE the dialog is built — so
            // the host would have photographed the chat view.
            if (!ApWaitAtomic(m_pane_open, true, std::chrono::seconds{60}))
                throw std::runtime_error("action " + std::to_string(i) + " show_pane '" + pane +
                                         "': dialog did not open within 60s");
            rendezvous(::xllama::wstring_to_utf8(a.label), a.timeout);
            // Closed unconditionally, including after a rendez-vous timeout: an
            // unattended capture may leave a mark unreleased, but it must not
            // leave a modal on screen for the next gate.
            ApDispatchSync([this]() {
                if (m_ap_dialog)
                    m_ap_dialog.Hide();
            });
            // Hide() returns before the coroutine's continuation has run, so the
            // wait is on Closed, not on Hide.
            if (!ApWaitAtomic(m_pane_open, false, kGrace))
                throw std::runtime_error("action " + std::to_string(i) + " show_pane '" + pane +
                                         "': dialog did not close");
            log_output("[autopilot] show_pane '" + pane + "' closed\n");
        } else if (a.op == "quit") {
            log_output("[autopilot] action " + std::to_string(i) + " quit\n");
            ::xllama::write_local_text("autopilot-done.txt", "ok");
            ApDispatchSync([]() { winrt::Windows::UI::Xaml::Application::Current().Exit(); });
            return;
        } else {
            // Not "unknown op" any more — validate_autopilot_script rejected
            // those before we got here. Reaching this means an op was added to
            // its table without a branch below, which no script can fix.
            throw std::runtime_error("op '" + a.op +
                                     "' is accepted by the validator but has no "
                                     "implementation in ApRun");
        }
        log_output("[autopilot] action " + std::to_string(i) + " " + a.op + " end\n");
    }
    // Fell off the end without an explicit quit: still a success.
    ::xllama::write_local_text("autopilot-done.txt", "ok");
}

} // namespace xllama

#endif // XLLAMA_UWP
