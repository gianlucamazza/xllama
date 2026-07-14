// Copyright (c) 2024 Venere Labs
// SPDX-License-Identifier: MIT

#pragma once

#ifdef XLLAMA_UWP

    #include "chat-history.h"
    #include "inference-bridge.h"
    #include "model-downloader.h"
    #include "pch.h"
    #include "xllama/session.h"

    #include <atomic>
    #include <chrono>
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
        std::string op;   // load_chat|send|new_chat|set_model|generate_image|quit
        std::wstring arg; // id / text / model name / image prompt
        int steps{1};     // generate_image
        unsigned seed{42};
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
    std::string BuildChatMLPrompt(const std::string& user_text, int* out_dropped = nullptr) const;
    // Only the new turn's ChatML tokens, appended to the reused KV cache.
    std::string BuildDeltaPrompt(const std::string& user_text) const;
    // Model-specific suffix after <|im_start|>assistant (e.g. Qwen no-think prefill).
    std::string AssistantGenSuffix() const;
    void LoadSettings();
    void SaveSettings();
    winrt::fire_and_forget ShowSettings();
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
    std::string m_gpu_model{"smollm2-360m-dml-fp16"};
    std::wstring m_active_model;

    // Image generation: TAESD tiny VAE replaces the full VAE decoder in-place
    // under models\sd-turbo-fp16\vae_decoder\ (same UNet/text_encoder).
    bool m_diffuse_taesd{false};

    // Token streaming state (written from bg thread, flushed on UI thread via timer)
    std::mutex m_token_mutex;
    std::string m_token_buffer;
    std::atomic<int> m_tokens_received{0};
    std::chrono::steady_clock::time_point m_gen_start;

    // Multi-turn chat state
    xllama::ui::ChatHistory m_history;
    xllama::ui::Conversation m_current;
    std::string m_system_prompt{"You are a helpful AI assistant."};

    // Autoscroll state: true while the user has not scrolled up during streaming
    bool m_at_bottom{true};

    // Sampling parameters (persisted in settings.json, exposed in Settings dialog)
    float m_temperature{0.8f};
    float m_top_p{0.9f};
    int m_top_k{40};
    float m_repetition_penalty{1.1f};
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
