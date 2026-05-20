// Copyright (c) 2024 Venere Labs
// SPDX-License-Identifier: MIT

#pragma once

#ifdef XLLAMA_UWP

    #include "llama-bridge.h"
    #include "pch.h"

    #include <atomic>
    #include <memory>
    #include <string>

namespace xllama {

// Plain C++ class that owns the UI tree.
// Not a WinRT runtimeclass — avoids the XAML metadata provider (IXamlMetadataProvider)
// and MarkupCompilePass2 entirely. The UI is built programmatically in BuildUI().
class MainPageController : public std::enable_shared_from_this<MainPageController> {
  public:
    MainPageController();

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
    void SetStatus(std::wstring const& status);
    void SetRunning(bool running);

    // UI controls (populated by BuildUI)
    winrt::Windows::UI::Xaml::Controls::Page m_root{nullptr};
    winrt::Windows::UI::Xaml::Controls::TextBlock m_modelText{nullptr};
    winrt::Windows::UI::Xaml::Controls::TextBlock m_statusText{nullptr};
    winrt::Windows::UI::Xaml::Controls::ProgressBar m_loadingBar{nullptr};
    winrt::Windows::UI::Xaml::Controls::ScrollViewer m_outputScroll{nullptr};
    winrt::Windows::UI::Xaml::Controls::TextBox m_promptInput{nullptr};
    winrt::Windows::UI::Xaml::Controls::TextBlock m_outputText{nullptr};
    winrt::Windows::UI::Xaml::Controls::TextBlock m_metricsText{nullptr};
    winrt::Windows::UI::Xaml::Controls::Button m_runButton{nullptr};
    winrt::Windows::UI::Xaml::Controls::Button m_cancelButton{nullptr};

    std::atomic<bool> m_abort{false};
    std::wstring m_model_filename;
};

} // namespace xllama

#endif // XLLAMA_UWP
