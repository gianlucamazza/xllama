// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#pragma once

#ifdef XLLAMA_UWP

    #include "App.g.h"
    #include "pch.h"

    #include <memory>

namespace xllama {
class MainPageController;
}

namespace winrt::xllama::implementation {

struct App : AppT<App> {
    App();
    void OnLaunched(winrt::Windows::ApplicationModel::Activation::LaunchActivatedEventArgs const&);

  private:
    std::shared_ptr<::xllama::MainPageController> m_controller;
};

} // namespace winrt::xllama::implementation

namespace winrt::xllama::factory_implementation {

struct App : AppT<App, implementation::App> {};

} // namespace winrt::xllama::factory_implementation

#endif // XLLAMA_UWP
