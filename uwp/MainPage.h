// Copyright (c) 2024 Venere Labs
// SPDX-License-Identifier: MIT

#pragma once

#ifdef XLLAMA_UWP

    #include "chat-history.h"
    #include "inference-bridge.h"
    #include "pch.h"

    #include <atomic>
    #include <chrono>
    #include <memory>
    #include <mutex>
    #include <string>

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

  private:
    void BuildUI();
    void LoadModelName();
    winrt::fire_and_forget CheckBenchMode();
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
    std::string BuildChatMLPrompt(const std::string& user_text) const;
    void LoadSettings();
    void SaveCurrentConversation(bool partial = false);

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
    winrt::Windows::UI::Xaml::DispatcherTimer m_flush_timer{nullptr};
    winrt::event_token m_flush_tick_token{};

    // Token streaming state (written from bg thread, flushed on UI thread via timer)
    std::mutex m_token_mutex;
    std::string m_token_buffer;
    std::atomic<int> m_tokens_received{0};
    std::chrono::steady_clock::time_point m_gen_start;

    // Multi-turn chat state
    xllama::ui::ChatHistory m_history;
    xllama::ui::Conversation m_current;
    std::string m_system_prompt{"You are a helpful AI assistant."};

    std::atomic<bool> m_abort{false};
    std::atomic<bool> m_is_running{false};
    std::wstring m_model_filename;
};

} // namespace xllama

#endif // XLLAMA_UWP
