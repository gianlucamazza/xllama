#pragma once

#ifdef XLLAMA_UWP

#include "pch.h"
#include "MainPage.g.h"
#include "llama-bridge.h"

#include <atomic>
#include <string>
#include <thread>

namespace winrt::xllama::implementation {

struct MainPage : MainPageT<MainPage> {
    MainPage();

    void OnRunClick(winrt::Windows::Foundation::IInspectable const&,
                    winrt::Windows::UI::Xaml::RoutedEventArgs const&);
    void OnCancelClick(winrt::Windows::Foundation::IInspectable const&,
                       winrt::Windows::UI::Xaml::RoutedEventArgs const&);

private:
    std::atomic<bool>    m_abort{false};
    std::wstring         m_model_filename;

    void LoadModelName();
    winrt::fire_and_forget CheckBenchMode();
    void StartInference(std::wstring const& prompt);
    void AppendOutput(std::wstring const& text);
    void SetStatus(std::wstring const& status);
    void SetRunning(bool running);
};

} // namespace winrt::xllama::implementation

namespace winrt::xllama::factory_implementation {

struct MainPage : MainPageT<MainPage, implementation::MainPage> {};

} // namespace winrt::xllama::factory_implementation

#endif // XLLAMA_UWP
