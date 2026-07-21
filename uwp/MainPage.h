// Copyright (c) 2024 Venere Labs
// SPDX-License-Identifier: MIT

#pragma once

#ifdef XLLAMA_UWP

    #include "chat-history.h"
    #include "inference-bridge.h"
    #include "model-downloader.h"
    #include "pch.h"
    #include "xllama/chat_prompt.h"
    #include "xllama/session.h"

    #include <atomic>
    #include <chrono>
    #include <cstdint>
    #include <functional>
    #include <memory>
    #include <mutex>
    #include <string>
    #include <vector>

namespace xllama {

enum class StatusKind { Info, Working, Success, Error };

// Plain C++ class that owns the UI tree.
// Not a WinRT runtimeclass — avoids the XAML metadata provider (IXamlMetadataProvider)
// and MarkupCompilePass2 entirely. The UI is built programmatically in BuildUI().
class MainPageController : public std::enable_shared_from_this<MainPageController> {
  public:
    MainPageController();
    void Init(); // must be called once after make_shared — wires handlers, loads model

    winrt::Windows::UI::Xaml::Controls::Page Root() const {
        return m_root;
    }

    // Autopilot: scripted validation of the real XAML UI (console has no human
    // input path in Dev Mode). No-op unless LocalState\autopilot.flag exists;
    // the flag is consumed, actions come from LocalState\autopilot.json, the
    // outcome lands in LocalState\autopilot-done.txt ("ok" | "error: ...").
    void StartAutopilotIfRequested(); // called from App::OnLaunched

  private:
    // One parsed autopilot action (see ApParseScript in MainPage.cpp).
    struct ApAction {
        std::string op;   // load_chat|send|new_chat|set_model|set_api|set_routing|set_sampling|
                          // set_kv_reuse|generate_image|rate|quit
        std::wstring arg; // id / text / model name / image prompt / rate label
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
    static bool ApParseScript(const std::string& json_utf8, std::vector<ApAction>& out,
                              std::chrono::seconds& total_cap, std::string& err);
    void ApRun(std::vector<ApAction> actions, std::chrono::seconds total_cap);
    void ApDispatchSync(std::function<void()> fn); // sync hop to the UI thread (MTA-only)
    bool ApWaitAtomic(std::atomic<bool>& flag, bool want, std::chrono::seconds timeout);
    void BuildUI();
    void LoadModelName();
    winrt::fire_and_forget EnsureModelAsync();
    // Provision one catalogue/USB/bundled model dir. When |set_app_ready| is true
    // (chat model), enables Run on success and may queue gpu_model download.
    winrt::fire_and_forget EnsureModelNamedAsync(std::wstring model_name,
                                                 bool set_app_ready = false);
    void EnsureGpuModelIfNeeded();
    void StartInference(std::wstring const& prompt);
    void OnRunClick(winrt::Windows::Foundation::IInspectable const&,
                    winrt::Windows::UI::Xaml::RoutedEventArgs const&);
    void OnCancelClick(winrt::Windows::Foundation::IInspectable const&,
                       winrt::Windows::UI::Xaml::RoutedEventArgs const&);
    void AppendOutput(std::wstring const& text);
    void SetStatus(std::wstring const& msg, StatusKind kind = StatusKind::Info);
    void SetRunning(bool running);
    void FlushTokenBuffer(); // flush m_token_buffer to RichTextBlock on UI thread

    // Multi-turn chat
    void NewChat();
    winrt::fire_and_forget ShowHistory();
    void LoadConversation(const std::string& id);
    void RenderConversation();
    void AddUserParagraph(std::wstring const& text);
    void AppendFeedbackControls(winrt::Windows::UI::Xaml::Documents::Paragraph const& paragraph,
                                size_t assistant_index);
    bool SubmitFeedback(size_t assistant_index, const std::string& label,
                        const std::string& preferred_assistant = {}, std::string* err = nullptr);
    winrt::fire_and_forget ShowCorrectionDialog(size_t assistant_index);
    std::string BuildPrompt(const std::string& user_text, int* out_dropped = nullptr) const;
    // Only the new turn's tokens, appended to the reused KV cache.
    std::string BuildDeltaPrompt(const std::string& user_text) const;
    // Per-architecture chat template for the current model (ChatML default, Gemma, ...).
    // Built on demand (reads m_model_filename); do not call from the worker thread.
    xllama::ChatFormat chat_format() const;
    void LoadSettings();
    void SaveSettings();
    winrt::fire_and_forget ShowSettings();
    void ApplyApiSettings(bool enabled, int port);
    // Image generation UX: shows the last diffuse-out.png and runs SD-Turbo
    // in-process (plain ORT DML coexists with the XAML compositor).
    winrt::fire_and_forget ShowImageDialog();
    void StartDiffusion();
    void PollDiffuseProgress();
    void FinishDiffusion();
    void SaveCurrentConversation(bool partial = false);
    // Must be called from background thread; builds/rebuilds m_session if needed.
    bool EnsureSession(const std::string& model, std::string* err_out = nullptr);

    // UI controls (populated by BuildUI)
    winrt::Windows::UI::Xaml::Controls::Page m_root{nullptr};
    winrt::Windows::UI::Xaml::Controls::TextBlock m_modelText{nullptr};
    winrt::Windows::UI::Xaml::Controls::TextBlock m_statusText{nullptr};
    winrt::Windows::UI::Xaml::Controls::ProgressBar m_loadingBar{nullptr};
    winrt::Windows::UI::Xaml::Controls::ScrollViewer m_outputScroll{nullptr};
    winrt::Windows::UI::Xaml::Controls::TextBox m_promptInput{nullptr};
    winrt::Windows::UI::Xaml::Controls::RichTextBlock m_outputBody{nullptr};
    winrt::Windows::UI::Xaml::Documents::Paragraph m_currentParagraph{nullptr};
    winrt::Windows::UI::Xaml::Controls::TextBlock m_metricsText{nullptr};
    winrt::Windows::UI::Xaml::Controls::Button m_runButton{nullptr};
    winrt::Windows::UI::Xaml::Controls::Button m_cancelButton{nullptr};
    winrt::Windows::UI::Xaml::Controls::Button m_newChatButton{nullptr};
    winrt::Windows::UI::Xaml::Controls::Button m_historyButton{nullptr};
    winrt::Windows::UI::Xaml::Controls::Button m_settingsButton{nullptr};
    winrt::Windows::UI::Xaml::Controls::Button m_imageButton{nullptr};
    winrt::Windows::UI::Xaml::DispatcherTimer m_flush_timer{nullptr};
    winrt::event_token m_flush_tick_token{};

    // Persistent inference session — loaded once, reused across chat turns.
    std::unique_ptr<xllama::Session> m_session;
    std::string m_session_model;

    // KV-cache reuse (continuous decoding) state.
    //   m_kv_reuse: feature toggle (settings.json "kv_reuse", default true).
    //   m_kv_valid: the session's persistent generator currently holds the KV for
    //               m_current through the last completed turn — the next turn can
    //               append only its delta. Cleared on new/loaded chat, settings
    //               change, abort, context eviction, or any generator failure.
    //   m_kv_last_ended_with_stop: whether the last turn stopped on <|im_end|>
    //               (already in KV) vs the n_predict cap (not) — drives the delta.
    bool m_kv_reuse{true};
    bool m_kv_valid{false};
    bool m_kv_last_ended_with_stop{false};

    // Per-conversation CPU/GPU routing (Stage 3). The GPU (DML fp16) EP wins the
    // prefill of long prompts; the CPU EP wins decode. Routing is decided at a
    // conversation's first turn and sticky for its lifetime (the KV cache is
    // per-EP). m_routing: 0 = CPU only (default = current behaviour), 1 = GPU
    // only, 2 = auto (route by first-prompt length). m_active_model holds the
    // routed model dir for the current conversation.
    int m_routing{0};
    std::string m_gpu_model{"smollm2-360m-dml-fp16-v2"};
    std::wstring m_active_model;

    // Image generation: TAESD tiny VAE replaces the full VAE decoder in-place
    // under models\sd-turbo-fp16\vae_decoder\ (same UNet/text_encoder).
    bool m_diffuse_taesd{false};
    uint32_t m_diffuse_seed{0};      // persisted UI value; 0 = choose a random seed
    uint32_t m_last_diffuse_seed{0}; // concrete seed used by the active/last run

    // Serialize detached API lifecycle workers and discard stale requests so
    // rapid Settings changes cannot apply out of order.
    std::mutex m_api_settings_mutex;
    std::atomic<uint64_t> m_api_settings_generation{0};

    // Token streaming state (written from bg thread, flushed on UI thread via timer)
    std::mutex m_token_mutex;
    std::string m_token_buffer;
    std::atomic<int> m_tokens_received{0};
    // Turn start, i.e. the moment the user's message was submitted. Includes
    // model load and prefill, so it is NOT a decode-rate denominator (#130).
    std::chrono::steady_clock::time_point m_gen_start;
    // First token out. Prefill ends here, decode begins. The app streams, so
    // this is the latency the user actually feels — everything after it arrives
    // faster than anyone reads (33-59 tok/s against ~10 tok/s of reading).
    // Measured but never surfaced before #130; the live counter divided by
    // m_gen_start and so reported a decode rate diluted by prefill.
    std::chrono::steady_clock::time_point m_first_token_at;
    std::atomic<bool> m_first_token_seen{false};
    // UI-thread latch: the status line is flipped from "reading prompt" to
    // "generating" once, on the first flushed batch. Separate from
    // m_first_token_seen, which is written by the inference thread.
    bool m_status_flipped_to_generating{false};

    // Multi-turn chat state
    xllama::ui::ChatHistory m_history;
    xllama::ui::Conversation m_current;
    std::string m_system_prompt{"You are a helpful AI assistant."};

    // Autoscroll state: true while the user has not scrolled up during streaming
    bool m_at_bottom{true};

    // Sampling parameters (persisted in settings.json, exposed in Settings dialog).
    // Seeded from the single source in sampling.h (#125) so the GUI defaults
    // cannot drift from the CLI/API defaults.
    float m_temperature{::xllama::sampling_defaults::kTemperature};
    float m_top_p{::xllama::sampling_defaults::kTopP};
    int m_top_k{::xllama::sampling_defaults::kTopK};
    float m_repetition_penalty{::xllama::sampling_defaults::kRepetitionPenalty};
    int m_n_predict{512};

    std::atomic<bool> m_abort{false};
    std::atomic<bool> m_is_running{false};
    std::atomic<bool> m_diffuse_running{false};
    // Set once EnsureModelAsync lands a usable chat model (any source); the
    // autopilot driver gates its first action on this.
    std::atomic<bool> m_model_ready{false};
    winrt::Windows::UI::Xaml::DispatcherTimer m_diffuse_timer{nullptr};
    winrt::event_token m_diffuse_tick_token{};
    std::wstring m_model_filename;
};

} // namespace xllama

#endif // XLLAMA_UWP
