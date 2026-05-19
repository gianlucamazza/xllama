#pragma once

#ifdef XLLAMA_UWP

#include "pch.h"

namespace xllama {

struct App : winrt::implements<App,
    winrt::Windows::ApplicationModel::Core::IFrameworkViewSource,
    winrt::Windows::ApplicationModel::Core::IFrameworkView>
{
    // IFrameworkViewSource
    winrt::Windows::ApplicationModel::Core::IFrameworkView CreateView();

    // IFrameworkView
    void Initialize(winrt::Windows::ApplicationModel::Core::CoreApplicationView const&);
    void Load(winrt::hstring const&);
    void Uninitialize();
    void Run();
    void SetWindow(winrt::Windows::UI::Core::CoreWindow const&);
};

} // namespace xllama

#endif // XLLAMA_UWP
