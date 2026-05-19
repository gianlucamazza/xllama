#pragma once

#ifdef XLLAMA_UWP

#include "pch.h"
#include "App.g.h"

namespace winrt::xllama::implementation {

struct App : AppT<App> {
    App();
    // Overrides AppT<App>::InitializeComponent which would load App.xbf (not in package).
    // App.xaml is minimal (no resources); a no-op is correct here.
    void InitializeComponent();
    void OnLaunched(winrt::Windows::ApplicationModel::Activation::LaunchActivatedEventArgs const&);
};

} // namespace winrt::xllama::implementation

namespace winrt::xllama::factory_implementation {

struct App : AppT<App, implementation::App> {};

} // namespace winrt::xllama::factory_implementation

#endif // XLLAMA_UWP
