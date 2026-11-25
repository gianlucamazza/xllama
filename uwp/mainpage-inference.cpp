// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// MainPageController: prompt planning and inference turn. Split out of MainPage.cpp (#549); shared
// includes and helpers are in mainpage-internal.h.

#include "mainpage-internal.h"

#ifdef XLLAMA_UWP

namespace xllama {

MainPageController::PromptPlan
MainPageController::BuildPromptPlan(const std::string& user_text) const {
    // The whole conversation, oldest first, as (user, assistant) exchanges plus
    // the new user_text. What fits n_ctx is fit_prompt's decision, in the worker,
    // in tokens: a chars-per-token estimate may size the prompt for routing but
    // must not decide what the user receives (AGENTS.md invariants).
    PromptPlan plan;
    plan.n_ctx = ::xllama::kDefaultNCtx;
    std::string role;
    {
        const auto& manifest = CachedManifest();
        if (const auto* e = ::xllama::FindManifestEntry(manifest, m_model_filename)) {
            plan.n_ctx = ::xllama::catalogue_n_ctx(e->n_ctx, e->kind == L"gguf");
            role = ::xllama::wstring_to_utf8(e->role);
        }
    }
    plan.chars_per_token = ::xllama::chars_per_token_for_role(role);
    std::size_t chars = m_system_prompt.size() + user_text.size();
    for (size_t i = 0; i < m_current.messages.size(); ++i) {
        if (m_current.messages[i].role != xllama::ui::MessageRole::User)
            continue;
        std::string assistant;
        if (i + 1 < m_current.messages.size() &&
            m_current.messages[i + 1].role == xllama::ui::MessageRole::Assistant)
            assistant = m_current.messages[i + 1].content;
        chars += m_current.messages[i].content.size() + assistant.size();
        plan.turns.push_back({m_current.messages[i].content, std::move(assistant)});
    }
    plan.estimated_tokens = ::xllama::estimate_tokens_from_chars(chars, plan.chars_per_token);
    plan.prompt = chat_format().render_prompt(m_system_prompt, plan.turns, user_text);
    return plan;
}

xllama::ChatFormat MainPageController::chat_format() const {
    return ::xllama::chat_format_for(::xllama::wstring_to_utf8(m_model_filename));
}

std::string MainPageController::BuildDeltaPrompt(const std::string& user_text) const {
    // The persistent KV cache already holds everything through the previous
    // assistant's generated tokens. The chat format closes that turn (only a
    // newline if the model already emitted the stop token; the full turn close
    // otherwise) and appends the new user turn + assistant header. Concatenated
    // onto the KV this reproduces exactly what BuildPrompt would have built.
    return chat_format().render_delta(user_text, m_kv_last_ended_with_stop);
}

// #170b: the KV of the conversation being left is worth ~12 KiB per resident
// token of prefill we would otherwise repeat on return (measured: a 1476-token
// conversation is a 17.6 MB file that loads in 33 ms against a 4.5 s
// re-prefill on console). Written on a detached thread — the UI thread must
// not wait on tens of MB — under the hub lock, and only while the resident
// session is still the one this conversation was built on.
//
// #216: a fire-and-forget save races the next turn. The worker holds hub.mtx
// for the whole generate, so if "Say hello" on a new chat starts before the
// snapshot finishes, save_state can capture the *new* conversation's KV and
// write it to the *old* conversation's path. On return, load_state succeeds
// (log: restored) but the #170a prefix never matches → full re-prefill (ret ≈
// cold). The gate under `all` surfaces this ~1/6; standalone rarely does.
// Fix: one outstanding save future; every generate waits before touching the
// session. Disk I/O still stays off the UI thread.
void MainPageController::WaitKvSnapshotSave() {
    std::future<void> f;
    {
        std::lock_guard<std::mutex> lk(m_kv_save_mutex);
        if (m_kv_save_future.valid())
            f = std::move(m_kv_save_future);
    }
    if (f.valid())
        f.wait();
}

void MainPageController::SaveKvSnapshotAsync() {
    if (!m_kv_reuse || !m_kv_valid || m_current.messages.empty())
        return;
    // Thinking models: the saved history holds the STRIPPED answer while the KV
    // holds the full <think> stream, so the #170a prefix diff always diverges
    // inside the first assistant turn on return — and on LFM's hybrid cache a
    // tail rewind is refused, collapsing to a full re-prefill. The snapshot
    // would cost tens of MB of writes to buy nothing.
    if (chat_format().strip_thinking_content)
        return;
    const std::string routed =
        ::xllama::wstring_to_utf8(m_active_model.empty() ? m_model_filename : m_active_model);
    if (!::xllama::model_uses_llama_backend(routed))
        return; // ORT keeps no serialisable per-sequence state
    CreateDirectoryW(local_wpath(L"kv").c_str(), nullptr);
    const auto store = kv_store();
    const std::string path = store.path_for(m_current.id);
    if (path.empty())
        return;
    const uint64_t epoch = m_kv_epoch.load();
    // Finish any previous save before scheduling another (same path possible).
    WaitKvSnapshotSave();
    {
        std::lock_guard<std::mutex> lk(m_kv_save_mutex);
        m_kv_save_future = std::async(std::launch::async, [store, path, epoch]() {
            auto& hub = ::xllama::session_hub();
            std::lock_guard<std::mutex> hub_lk(hub.mtx);
            if (!hub.session || hub.session->kv_epoch() != epoch)
                return; // the resident KV is not this conversation's any more
                        // (the API, a reload or another model) — nothing to save
            std::string err;
            // A snapshot larger than the whole pool would be pruned right after
            // it was written: refuse it before the state is copied (#570).
            if (hub.session->save_state(path, &err,
                                        static_cast<std::size_t>(::xllama::kKvStoreMaxBytes)))
                store.prune(::xllama::kKvStoreMaxFiles, ::xllama::kKvStoreMaxBytes);
            else
                ::xllama::log_output("[xllama] KV snapshot not saved: " + err + "\n");
        });
    }
}

// ---------------------------------------------------------------------------
// StartInference: called on UI thread; spawns background thread
// ---------------------------------------------------------------------------

bool MainPageController::StartInference(std::wstring const& prompt_w) {
    auto compute_permit = ::xllama::try_acquire_compute_job();
    if (!compute_permit && !::xllama::session_hub().preloading.load(std::memory_order_acquire)) {
        m_promptInput.Text(winrt::hstring(prompt_w.c_str()));
        SetStatus(L"Busy — another compute job is active", StatusKind::Error);
        return false;
    }
    m_abort.store(false);
    SetStatus(L"Loading model...", StatusKind::Working);
    SetRunning(true);

    // Add user message to display + prepare empty assistant paragraph
    AddUserParagraph(prompt_w);
    m_metricsText.Text(L"");
    m_live_decode_tok_s.reset();
    m_live_tokens.reset();

    // Reset streaming counters
    m_tokens_received.store(0);
    {
        std::lock_guard<std::mutex> lk(m_token_mutex);
        m_token_buffer.clear();
    }
    m_gen_start = std::chrono::steady_clock::now();
    m_first_token_seen.store(false);
    m_status_flipped_to_generating = false;

    // Start flush timer (40 ms tick: batches tokens, updates live tok/s)
    auto self = shared_from_this();
    if (!m_flush_timer) {
        m_flush_timer = winrt::Windows::UI::Xaml::DispatcherTimer{};
        m_flush_timer.Interval(std::chrono::milliseconds(40));
    } else {
        m_flush_timer.Stop();
        m_flush_timer.Tick(m_flush_tick_token); // revoke previous handler
    }
    auto weak_self = std::weak_ptr<MainPageController>(self);
    m_flush_tick_token = m_flush_timer.Tick([weak_self](IInspectable const&, IInspectable const&) {
        if (auto s = weak_self.lock())
            s->FlushTokenBuffer();
    });
    m_flush_timer.Start();

    // Build multi-turn ChatML prompt from conversation history.
    // BuildPrompt uses existing m_current.messages (prev turns) and appends user_text.
    std::string user_text = ::xllama::wstring_to_utf8(prompt_w);
    // Two independent conditions, and tying them together cost every
    // conversation its title: NewChat() already assigns an id (it needs one for
    // the KV snapshot path), so `id.empty()` is false by the time the first
    // message is sent and the title was never derived. Only a conversation that
    // reached here with no id at all — the very first one after a cold launch —
    // ever got one. Measured on console: 14 of 14 saved conversations had
    // "title":"", and History rendered them as "(2 msgs) • today 13:20".
    if (m_current.id.empty())
        m_current.id = xllama::ui::ChatHistory::NewId();
    if (m_current.title.empty())
        m_current.title = xllama::ui::ChatHistory::TitleFrom(user_text);
    const PromptPlan plan = BuildPromptPlan(user_text);
    std::string full_prompt = plan.prompt;

    // Is the base model a GGUF (llama.cpp backend)? On Xbox llama.cpp is CPU-only
    // (no ggml GPU backend), so EP routing is meaningless and stays gated off for
    // GGUF (below). KV-cache reuse, however, IS supported: LlamaSession keeps a
    // persistent llama_context, so a reuse turn appends only the delta (turn-2
    // prefill 4.07×). Reuse is gated by kv_reuse_supported_for_model(), not by
    // this flag.
    bool base_is_gguf = false;
    {
        // Cached: this runs on the UI thread on every turn, and the manifest
        // JSON only changes on a personalized-model publish (which invalidates).
        const auto& manifest = CachedManifest();
        const auto* e = ::xllama::FindManifestEntry(manifest, m_model_filename);
        base_is_gguf = e && e->kind == L"gguf";
    }

    // Stage 3: decide EP routing once per conversation (sticky — the KV cache is
    // per-EP). m_active_model is cleared on new/loaded chat, so this fires on the
    // first turn and stays fixed after. Default (m_routing==0) keeps the CPU model.
    if (m_active_model.empty()) {
        // Filesystem probe (LocalState + InstalledPath stats) — only the sticky
        // first-turn decision consumes it, so keep it off the per-turn path (#174).
        // GPU is available only when every weight/data pin classifies Ok on a
        // size-only listing (`weight_pins_verified`), the LocalState `.complete`
        // marker exists (hashes ran at download/promotion), and no model write
        // is in flight. Licence and notice rows do not participate. The loose
        // "any ONNX/GGUF exists" probe used to route a long turn at a
        // still-downloading DML tree (#370).
        const std::wstring gpu_name = ::xllama::utf8_to_wstring(m_gpu_model);
        auto local_folder = winrt::Windows::Storage::ApplicationData::Current().LocalFolder();
        const std::wstring local_gpu_dir =
            std::wstring(local_folder.Path().c_str()) + L"\\models\\" + gpu_name;
        std::vector<PresentFile> gpu_present;
        std::vector<ManifestFilePin> gpu_pins;
        {
            const auto& manifest = CachedManifest();
            if (const auto* e = ::xllama::FindManifestEntry(manifest, gpu_name)) {
                gpu_present.reserve(e->files.size());
                gpu_pins.reserve(e->files.size());
                for (const auto& f : e->files) {
                    uint64_t sz = 0;
                    std::error_code ec;
                    const auto p = std::filesystem::path(local_gpu_dir) / f.filename;
                    if (std::filesystem::is_regular_file(p, ec)) {
                        const auto n = std::filesystem::file_size(p, ec);
                        if (!ec)
                            sz = static_cast<uint64_t>(n);
                    }
                    gpu_present.push_back({f.filename, sz, {}});
                    gpu_pins.push_back(make_file_pin(f.filename, f.approx_bytes,
                                                     ::xllama::wstring_to_utf8(f.sha256),
                                                     ::xllama::wstring_to_utf8(f.role)));
                }
            }
        }
        const bool expected_verified = ::xllama::gpu_tree_ready_for_routing(
            ::xllama::weight_pins_verified(gpu_present, gpu_pins),
            ::xllama::ModelDownloader::IsComplete(local_gpu_dir));
        const bool write_in_progress = ::xllama::ModelDownloader::WriterGate().busy();
        const auto gpu_state = ::xllama::gpu_route_readiness(expected_verified, write_in_progress);
        const bool gpu_provisioned =
            ::xllama::gpu_ready_for_routing(expected_verified, write_in_progress);
        const bool loose_present =
            expected_verified ? true : ::xllama::IsModelProvisioned(gpu_name);
        ::xllama::RoutingSettings rs;
        rs.mode = static_cast<::xllama::RoutingMode>(m_routing);
        rs.cpu_model = ::xllama::wstring_to_utf8(m_model_filename);
        rs.gpu_model = m_gpu_model;

        int n_tok = 0;
        if (m_routing == 2 && !base_is_gguf) {
            // Exact count only when the CPU session is already resident (a
            // name-compare no-op). The previous shape loaded the CPU model
            // just to tokenize — ON THE UI THREAD — and then could tear it
            // down for the DML model (+1.2 s, uwp-constraints §5, 2786 ms vs
            // 1570 ms). With no session resident the role's chars-per-token
            // estimate sizes it. Either way routing sees at most the context
            // ceiling, which is what fit_prompt will prefill (#133).
            // try_lock: if the hub is busy (LAN API mid-turn) fall back to the
            // estimator rather than blocking the UI thread.
            auto& hub = ::xllama::session_hub();
            std::unique_lock<std::mutex> hub_lk(hub.mtx, std::try_to_lock);
            const int counted = (hub_lk.owns_lock() && hub.session && hub.model == rs.cpu_model)
                                    ? hub.session->count_tokens(full_prompt)
                                    : ::xllama::estimate_tokens_from_chars(full_prompt.size(),
                                                                           plan.chars_per_token);
            n_tok = ::xllama::routing_prompt_tokens(counted, plan.n_ctx);
        }

        // #91/#95: for a gpu_model that is not a parity-validated DML text asset,
        // decide_routing resolves every mode to the CPU model and the GPU model is
        // intentionally not provisioned — a missing gpu_model must not block the
        // turn or nag the user. A validated asset that is only partial or mid-write
        // is not "missing": fall through to CPU via gpu_provisioned == false.
        const bool warn_gpu_missing =
            ::xllama::dml_text_model_ok(m_gpu_model) && !loose_present && !write_in_progress;
        if (gpu_state != ::xllama::GpuRouteReadiness::Ready &&
            ::xllama::dml_text_model_ok(m_gpu_model) && !warn_gpu_missing &&
            (m_routing == 1 || (m_routing == 2 && !base_is_gguf && n_tok > rs.token_threshold))) {
            ::xllama::log_output(::xllama::gpu_route_fallback_log(gpu_state));
        }
        if (m_routing == 1 && warn_gpu_missing) {
            SetStatus(L"GPU model '" + ::xllama::utf8_to_wstring(m_gpu_model) +
                          L"' is not on this console.\n"
                          L"Upload it to LocalState\\models\\" +
                          ::xllama::utf8_to_wstring(m_gpu_model) +
                          L" via Device Portal or USB (see docs/model-selection.md).",
                      StatusKind::Error);
            SetRunning(false);
            return true;
        }
        if (m_routing == 2 && n_tok > rs.token_threshold && warn_gpu_missing) {
            SetStatus(L"GPU model '" + ::xllama::utf8_to_wstring(m_gpu_model) +
                          L"' is not on this console — staying on CPU.\n"
                          L"Upload it to LocalState\\models\\" +
                          ::xllama::utf8_to_wstring(m_gpu_model) + L" to enable auto-routing.",
                      StatusKind::Error);
            SetRunning(false);
            return true;
        }

        const auto decision = ::xllama::decide_routing(rs, n_tok, base_is_gguf, gpu_provisioned);
        m_active_model = ::xllama::utf8_to_wstring(decision.active_model);

        if (m_routing == 1) {
            ::xllama::log_output("[xllama] routing: gpu-only\n");
        } else if (m_routing == 2 && !base_is_gguf) {
            char rbuf[160];
            snprintf(rbuf, sizeof(rbuf),
                     "[xllama] routing: auto → %s (%d tok, threshold %d, model=%s)\n",
                     decision.use_gpu ? "gpu" : "cpu", decision.token_count, rs.token_threshold,
                     decision.active_model.c_str());
            ::xllama::log_output(rbuf);
        }
    }

    // KV-cache reuse decision (continuous decoding). Reuse only when enabled, the
    // persistent generator already holds this conversation (m_kv_valid), and the
    // conversation is not expected to outgrow the context this round (RewindTo
    // cannot drop from the head, so an overflowing delta forces a full re-prefill).
    // The estimate only decides whether to ATTEMPT reuse; a wrong guess costs the
    // fail-fast retry below, never the reply's content. A reuse turn appends only the delta;
    // otherwise we (re)prefill the full prompt — which also (re)seeds the persistent generator.
    const std::string routed_model =
        ::xllama::wstring_to_utf8(m_active_model.empty() ? m_model_filename : m_active_model);
    // KV reuse now works for both backends: ORT-GenAI (persistent generator) and
    // GGUF/llama.cpp (persistent llama_context in LlamaSession). ep_kv_ok still
    // excludes the DirectML routing model (continuous decoding unsupported there).
    const bool ep_kv_ok = ::xllama::kv_reuse_supported_for_model(routed_model);
    // #169: on the llama backend a trimmed round no longer forces a full
    // re-prefill — the session evicts the oldest resident tokens (context
    // shift) when the delta would overflow n_ctx, keeping long chats in the
    // reuse regime. If the arch cannot shift, the continuation fail-fasts
    // (#173) and the retry below re-seeds with the trimmed full prompt.
    const bool shift_capable = ::xllama::model_uses_llama_backend(routed_model);
    const bool expect_overflow =
        plan.estimated_tokens > ::xllama::max_prompt_tokens_for_n_ctx(plan.n_ctx);
    bool do_reuse = m_kv_reuse && m_kv_valid && (!expect_overflow || shift_capable) && ep_kv_ok;
    bool kv_reuse = m_kv_reuse && ep_kv_ok;
    std::string delta_prompt = do_reuse ? BuildDeltaPrompt(user_text) : std::string();

    // Record user message in history AFTER building prompt (avoids duplicate)
    {
        xllama::ui::ChatMessage umsg;
        umsg.role = xllama::ui::MessageRole::User;
        umsg.content = user_text;
        umsg.ts_unix = static_cast<int64_t>(std::time(nullptr));
        m_current.messages.push_back(std::move(umsg));
    }

    std::string model =
        ::xllama::wstring_to_utf8(m_active_model.empty() ? m_model_filename : m_active_model);
    auto dispatcher = m_root.Dispatcher();

    // Per-model chat format (stop sequences + output post-processing) — resolved
    // on the UI thread (chat_format() reads m_model_filename) and captured by value.
    xllama::ChatFormat fmt = chat_format();

    // #169: the pinned head a context shift must never evict — the system
    // block render_prompt puts before the first user turn. Rendered here
    // (m_system_prompt is UI-thread state) and tokenized per turn in the
    // worker, where the session is available.
    const std::string sys_prefix = fmt.render_system_prefix(m_system_prompt);

    // #170b: the snapshot to try when this conversation's KV is not resident.
    const std::string kv_path = kv_reuse ? kv_store().path_for(m_current.id) : std::string();

    // Copies for the worker: the surviving turns and the numbers fit_prompt needs.
    // m_system_prompt / m_n_predict are UI-thread state, read here.
    const std::vector<::xllama::ChatTurn> plan_turns = plan.turns;
    const std::string system_prompt = m_system_prompt;
    int plan_n_ctx = plan.n_ctx; // replaced by the loaded session's n_ctx in the worker
    const int plan_n_predict = m_n_predict;

    std::thread([self, full_prompt, delta_prompt, do_reuse, kv_reuse, ep_kv_ok, model, fmt,
                 sys_prefix, kv_path, dispatcher, plan_turns, user_text, system_prompt, plan_n_ctx,
                 plan_n_predict, compute_permit, prompt_w]() mutable {
        try {
            if (!compute_permit) {
                compute_permit = ::xllama::try_acquire_compute_job_while_preloading(
                    ::xllama::session_hub().preloading);
                if (!compute_permit) {
                    dispatcher.RunAsync(CoreDispatcherPriority::Normal, [self, prompt_w]() {
                        self->m_promptInput.Text(winrt::hstring(prompt_w.c_str()));
                        self->SetStatus(L"Busy — another compute job is active", StatusKind::Error);
                        self->SetRunning(false);
                    });
                    return;
                }
            }
            // #216: drain any #170b snapshot save before taking hub.mtx. The save
            // also locks the hub; waiting first keeps the leave-conversation
            // token list on the snapshot path instead of a later turn's KV.
            self->WaitKvSnapshotSave();

            // One hub lock for the whole turn: the resident session cannot be
            // swapped from under us (LAN API requests report busy meanwhile,
            // exactly as they do against each other).
            auto& hub = ::xllama::session_hub();
            std::unique_lock<std::mutex> hub_lk(hub.mtx, std::try_to_lock);
            if (!hub_lk.owns_lock()) {
                // Another job holds the single slot: a LAN request, a pre-load,
                // a KV snapshot save or a pull's model swap. Say which when we
                // can, instead of a silent stall, then wait for it.
                std::wstring waiting = L"Waiting for the model to be free...";
    #ifndef XLLAMA_STORE_SKU
                if (::xllama::api::activity().snapshot().in_flight)
                    waiting = L"Waiting for LAN API request to finish...";
    #endif
                dispatcher.RunAsync(CoreDispatcherPriority::Normal, [self, waiting]() {
                    self->SetStatus(waiting, StatusKind::Working);
                });
                hub_lk.lock();
                dispatcher.RunAsync(CoreDispatcherPriority::Normal, [self]() {
                    self->SetStatus(L"Loading model...", StatusKind::Working);
                });
            }
            std::string load_err;
            if (!self->EnsureSession(model, &load_err)) {
                dispatcher.RunAsync(CoreDispatcherPriority::Normal, [self, load_err]() {
                    self->SetStatus(::xllama::utf8_to_wstring("Load failed: " + load_err),
                                    StatusKind::Error);
                    self->SetRunning(false);
                });
                return;
            }
            // Budget against the context the resident session holds, not the
            // UI-thread manifest cache (a LocalState manifest override can leave
            // it stale) and not the requested n_ctx, which a GGUF session caps
            // at its trained context (#570).
            const int loaded_n_ctx = hub.session ? hub.session->context_length() : 0;
            plan_n_ctx = loaded_n_ctx > 0 ? loaded_n_ctx : hub.params.n_ctx;

            // #170b: this conversation's KV is not resident (switched away and
            // back, or the app was restarted). Restoring the snapshot does not
            // change the turn below — it stays a full-prompt turn, and the
            // #170a prefix diff is what turns the restored cache into a delta
            // prefill. A missing, stale or foreign snapshot just fails here and
            // the turn prefills as it always did.
            if (!do_reuse && !kv_path.empty()) {
                std::string kv_err;
                if (self->m_turn_session->load_state(kv_path, &kv_err))
                    ::xllama::log_output("[xllama] KV snapshot restored (#170b)\n");
            }

            auto on_status = [self, dispatcher](const std::string& s) {
                // The backends emit "generating" BEFORE prefill — nothing is
                // generated yet, and on a long prompt that is several seconds of
                // the user staring at a word that is not true. Call the phase
                // what it is; FlushTokenBuffer flips it on the first real token.
                auto ws = ::xllama::utf8_to_wstring(s == "generating" ? "reading prompt" : s);
                StatusKind k =
                    (s.rfind("error:", 0) == 0) ? StatusKind::Error : StatusKind::Working;
                dispatcher.RunAsync(CoreDispatcherPriority::Normal,
                                    [self, ws, k]() { self->SetStatus(ws, k); });
            };
            // Token accumulation — no per-token RunAsync dispatch (batched by flush timer)
            auto on_token = [self](std::string_view tok) {
                // Stamp prefill-end on the first token — the only place that
                // knows when the model stopped reading and started writing.
                // This runs on the inference thread; FlushTokenBuffer reads the
                // timestamp on the UI thread. Write the value FIRST, then
                // publish with a release store, so an acquire load of the flag
                // guarantees the timestamp is visible. (Relaxed here would be a
                // data race that x64 happens to hide.)
                if (!self->m_first_token_seen.load(std::memory_order_relaxed)) {
                    self->m_first_token_at = std::chrono::steady_clock::now();
                    self->m_first_token_seen.store(true, std::memory_order_release);
                }
                self->m_tokens_received.fetch_add(1, std::memory_order_relaxed);
                std::lock_guard<std::mutex> lk(self->m_token_mutex);
                self->m_token_buffer += tok;
            };

            auto run_turn = [&](const std::string& p, bool reuse, bool reset) {
                xllama::GenerateParams gp;
                gp.prompt = p;
                gp.n_predict = self->m_n_predict;
                gp.temperature = self->m_temperature;
                gp.top_p = self->m_top_p;
                gp.top_k = self->m_top_k;
                gp.repetition_penalty = self->m_repetition_penalty;
                gp.abort_flag = &self->m_abort;
                gp.stop_sequences = fmt.stop_sequences;
                gp.reuse_kv = reuse;
                gp.reset_kv = reset;
                // A continuation is only valid on the KV this conversation left;
                // the session refuses it otherwise and the caller retries in full.
                if (reuse && !reset)
                    gp.expected_kv_epoch = self->m_kv_epoch.load();
                // count_tokens matches generate()'s full-prompt tokenization
                // (BOS included), and the system block ends on a special
                // token, so the boundary is exact. Ignored unless a context
                // shift actually runs (#169).
                gp.n_keep = self->m_turn_session->count_tokens(sys_prefix);
                // A context shift evicts whole turns, up to this terminator.
                gp.turn_close = fmt.turn_close;
                gp.on_status = on_status;
                gp.on_token = on_token;
                return self->m_turn_session->generate(gp);
            };

            // THE budget enforcement point. The estimate upstream chose what routing
            // would see; here the tokenizer of the model that will generate decides
            // what actually fits, so no chars-per-token constant can shorten a reply
            // behind the user's back (xllama::fit_prompt). Lazy: a reuse turn sends
            // only the delta and never needs it — but its retry does.
            bool fitted = false;
            std::string fit_error; // set when even the final message cannot fit
            auto fit_full_prompt = [&]() -> const std::string& {
                if (fitted)
                    return full_prompt;
                fitted = true;
                const ::xllama::PromptFit fit =
                    ::xllama::fit_prompt(fmt, system_prompt, plan_turns, user_text, plan_n_ctx,
                                         plan_n_predict, [self](const std::string& text) {
                                             return self->m_turn_session->count_tokens(text);
                                         });
                char fb[192];
                snprintf(fb, sizeof(fb),
                         "[xllama] prompt budget: %d tok exact, dropped %d old turn(s), "
                         "n_ctx %d, reply %d%s\n",
                         fit.n_tokens, fit.dropped, plan_n_ctx, plan_n_predict,
                         fit.fits ? "" : " — DOES NOT FIT");
                ::xllama::log_output(fb);
                if (!fit.fits) {
                    // Logged as well as shown: the refusal never reaches the
                    // session, so nothing else records it in xllama.log.
                    fit_error = ::xllama::prompt_too_long_message(
                        "final user message", fit.n_tokens, plan_n_predict, plan_n_ctx);
                    ::xllama::log_output("[xllama] chat: " + fit_error + "\n");
                }
                if (fit.dropped > 0) {
                    const int dropped = fit.dropped;
                    dispatcher.RunAsync(CoreDispatcherPriority::Normal, [self, dropped]() {
                        self->SetStatus(L"Context trimmed — " + std::to_wstring(dropped) +
                                        L" old turn(s) dropped");
                    });
                }
                full_prompt = fit.prompt;
                return full_prompt;
            };
            // A full-prompt turn, or the same refusal the LAN API gives (400)
            // when the final message alone plus the reply's room exceeds n_ctx —
            // never a silently shortened reply.
            auto run_full_turn = [&](bool reuse, bool reset) {
                const std::string& prompt = fit_full_prompt();
                if (!fit_error.empty()) {
                    xllama::InferenceResult refused;
                    refused.error_msg = fit_error;
                    return refused;
                }
                return run_turn(prompt, reuse, reset);
            };

            xllama::InferenceResult res;
            if (do_reuse) {
                res = run_turn(delta_prompt, /*reuse=*/true, /*reset=*/false);
                // A continuation that fails before emitting any token (e.g. appending
                // to a finished generator) falls back to a full re-prefill. No tokens
                // were streamed yet, so the UI stays clean.
                if (!res.success && res.n_eval == 0) {
                    ::xllama::log_output("[xllama] KV reuse failed, retrying with full prefill\n");
                    res = run_full_turn(/*reuse=*/kv_reuse, /*reset=*/true);
                }
            } else {
                // First turn / post-reset: seed the persistent generator (reuse+reset)
                // when KV reuse is enabled, else a pure stateless turn.
                res = run_full_turn(/*reuse=*/kv_reuse, /*reset=*/kv_reuse);
            }

            // The KV this conversation now owns (still under hub.mtx); the next
            // continuation must find the session at this epoch.
            self->m_kv_epoch = self->m_turn_session->kv_epoch();

            std::wstring metrics;
            if (res.success) {
                wchar_t buf[256];
                double dt = (res.n_eval > 0 && res.t_eval_ms > 0)
                                ? (double)res.n_eval / (res.t_eval_ms / 1000.0)
                                : 0.0;
                // #130: lead with time-to-first-token. The app streams, so TTFT
                // is the wait the user actually experiences; the decode rate
                // that follows is well above reading speed on both backends and
                // is the less interesting number. res.t_p_eval_ms was measured
                // by both backends all along and simply never shown.
                const double ttft_s = res.t_p_eval_ms / 1000.0;
                if (res.n_p_eval > 0 && res.t_p_eval_ms > 0) {
                    swprintf_s(buf,
                               L"%.1fs to first token (%d tok prompt)  ·  decode %.1f tok/s  ·  "
                               L"%d tok  ·  peak %zu MB",
                               ttft_s, res.n_p_eval, dt, res.n_eval, res.peak_ws_mb);
                } else {
                    swprintf_s(buf, L"decode %.1f tok/s  ·  %d tok  ·  peak %zu MB", dt, res.n_eval,
                               res.peak_ws_mb);
                }
                metrics = buf;
            } else {
                metrics = ::xllama::utf8_to_wstring(res.error_msg.empty() ? "inference failed"
                                                                          : res.error_msg);
            }

            std::string output_text = fmt.postprocess_output(res.output_text);
            bool was_aborted = self->m_abort.load();
            // A thinking model that spent its whole budget reasoning leaves
            // NOTHING after postprocess (strip_thinking_blocks drops an unclosed
            // <think> to EOF). Without a stand-in the turn vanished: no message
            // saved, no FinalizeStreamedTurn, the streamed chain of thought left
            // orphaned on screen and a status of "Done". Say what happened
            // instead — the fix for the cut-off reply is the Max-new-tokens box.
            //
            // Only for a turn that COMPLETED: a cancel or a failure mid-thought
            // empties the output too, and blaming the token budget there would
            // be a lie (and would persist a reply the user stopped).
            const bool thinking_only =
                res.success && !was_aborted && output_text.empty() && !res.output_text.empty();
            if (thinking_only) {
                output_text = "(reasoning only — the answer did not fit; raise \"Max new "
                              "tokens\" in Settings)";
                ::xllama::log_output(
                    "[xllama] postprocess left no answer (truncated reasoning block)\n");
            }
            dispatcher.RunAsync(
                CoreDispatcherPriority::Normal,
                [self, metrics, res, output_text, was_aborted, ep_kv_ok, thinking_only]() {
                    self->m_metricsText.Text(metrics);
                    self->m_last_inference = res;
                    self->m_live_decode_tok_s.reset();
                    self->m_live_tokens.reset();
                    if (self->m_debug_dashboard)
                        self->RefreshDebugDashboard();
                    const wchar_t* status_text = L"Error";
                    StatusKind status_kind = StatusKind::Error;
                    if (was_aborted) {
                        status_text = L"Cancelled";
                        status_kind = StatusKind::Info;
                    } else if (res.success && thinking_only) {
                        // Not a failure and not a normal answer: the model spent the
                        // whole budget reasoning.
                        status_text = L"Reasoning cut off — raise Max new tokens";
                        status_kind = StatusKind::Info;
                    } else if (res.success) {
                        status_text = L"Done";
                        status_kind = StatusKind::Success;
                    }
                    self->SetStatus(status_text, status_kind);
                    self->SetRunning(false); // also stops timer + flushes remaining tokens
                    // KV-reuse bookkeeping: the persistent generator now holds this
                    // turn only if it completed cleanly. On failure or abort, force a
                    // fresh generator (full re-prefill) next turn.
                    if (self->m_kv_reuse && res.success && !was_aborted && ep_kv_ok) {
                        self->m_kv_valid = true;
                        self->m_kv_last_ended_with_stop = res.ended_with_stop;
                    } else {
                        self->m_kv_valid = false;
                    }
                    // Save assistant response (partial-flagged if user aborted)
                    if (!output_text.empty()) {
                        xllama::ui::ChatMessage amsg;
                        amsg.role = xllama::ui::MessageRole::Assistant;
                        amsg.content = output_text;
                        amsg.ts_unix = static_cast<int64_t>(std::time(nullptr));
                        amsg.partial = was_aborted;
                        self->m_current.messages.push_back(std::move(amsg));
                    }
                    self->SaveCurrentConversation(was_aborted);
                    if (!was_aborted && !output_text.empty())
                        self->FinalizeStreamedTurn(output_text);
                });
        } catch (const std::exception& ex) {
            ::xllama::log_output(std::string("[xllama] thread terminated: ") + ex.what() + "\n");
            dispatcher.RunAsync(CoreDispatcherPriority::Normal, [self]() {
                self->SetStatus(L"Fatal error — see xllama.log", StatusKind::Error);
                self->SaveCurrentConversation(/*partial=*/true);
                self->SetRunning(false);
            });
        } catch (...) {
            ::xllama::log_output("[xllama] thread terminated: unknown exception\n");
            dispatcher.RunAsync(CoreDispatcherPriority::Normal, [self]() {
                self->SetStatus(L"Fatal error — see xllama.log", StatusKind::Error);
                self->SaveCurrentConversation(/*partial=*/true);
                self->SetRunning(false);
            });
        }
    }).detach();
    return true;
}

// ---------------------------------------------------------------------------
// Event handlers (wired via weak_ptr lambda in ctor; kept for direct call)
// ---------------------------------------------------------------------------

void MainPageController::OnRunClick(IInspectable const&, RoutedEventArgs const&) {
    auto prompt = m_promptInput.Text();
    if (prompt.empty()) {
        SetStatus(L"Enter a prompt first");
        return;
    }
    StartInference(std::wstring(prompt.c_str()));
}

void MainPageController::OnCancelClick(IInspectable const&, RoutedEventArgs const&) {
    // Which job this targets is policy, and it lives in cancel_policy.h where
    // the host tests can reach it — the combination that shipped broken (image
    // running, generic flag also set) is not reproducible from this file.
    // m_is_running is the generic flag: SetRunning() is called by all three
    // jobs, so it is passed last and the specific flags decide.
    switch (::xllama::cancel_target(m_diffuse_running.load(), m_train_running.load(),
                                    m_is_running.load())) {
    case ::xllama::CancelTarget::Image:
        ::xllama::write_local_text("diffuse-cancel.flag", "cancel");
        SetStatus(L"Cancelling image...");
        m_cancelButton.IsEnabled(false);
        return;
    case ::xllama::CancelTarget::Training:
        m_train_abort.store(true);
        SetStatus(L"Cancelling training (between epochs)...");
        m_cancelButton.IsEnabled(false);
        return;
    case ::xllama::CancelTarget::Text:
        m_abort.store(true);
        SetStatus(L"Cancelling...");
        m_cancelButton.IsEnabled(false);
        return;
    case ::xllama::CancelTarget::None:
        // Reachable from the B button on an idle chat, never from the Cancel
        // button (which is disabled unless a job runs). Doing nothing is the
        // whole point: there is no job to abort.
        return;
    }
}

} // namespace xllama

#endif // XLLAMA_UWP
