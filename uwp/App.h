// Copyright (c) 2024 Venere Labs
// SPDX-License-Identifier: MIT

#pragma once

#ifdef XLLAMA_UWP

    #include "App.g.h"
    #include "pch.h"

namespace winrt::xllama::implementation {

struct App : AppT<App> {
    App();
    // Overrides AppT<App>::InitializeComponent. App.xaml is build-time metadata
    // only and is not loaded at runtime on Xbox.
    void InitializeComponent();
    void OnLaunched(winrt::Windows::ApplicationModel::Activation::LaunchActivatedEventArgs const&);
};

} // namespace winrt::xllama::implementation

namespace winrt::xllama::factory_implementation {

struct App : AppT<App, implementation::App> {};

} // namespace winrt::xllama::factory_implementation

#endif // XLLAMA_UWP
