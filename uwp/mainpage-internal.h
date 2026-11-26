// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// Private to MainPage.cpp and the uwp/mainpage-*.cpp translation units (#549):
// the includes, using directives and small helpers every part of
// MainPageController shares. Nothing else includes it.
#pragma once

#ifdef XLLAMA_UWP

// clang-format off
#include "pch.h"
#include "MainPage.h"
// clang-format on

    #ifndef XLLAMA_STORE_SKU
        #include "api-server.h"
    #endif
    #include "chat-history.h"
    #include "inference-bridge.h"
    #include "xllama/app_settings.h"
    #include "xllama/cancel_policy.h"
    #include "xllama/chat_prompt.h"
    #include "xllama/compute_admission.h"
    #include "xllama/debug_dashboard.h"
    #include "xllama/diffusion/request.h"
    #include "xllama/diffusion/residency.h"
    #include "xllama/kv_store.h"
    #include "xllama/local_knob.h"
    #include "xllama/model_file_plan.h"
    #include "xllama/model_provision.h"
    #include "xllama/path_utils.h"
    #include "xllama/personalize.h"
    #include "xllama/platform.h"
    #include "xllama/preference_capture.h"
    #include "xllama/prompt_budget.h"
    #include "xllama/routing_policy.h"
    #include "xllama/stability_attribution.h"
    #include "xllama/training.h"
    #include "xllama/utf8_utils.h"

    #include <winrt/Windows.Data.Json.h>
    #include <winrt/Windows.Networking.Connectivity.h>
    #include <winrt/Windows.Networking.h>
    #include <winrt/Windows.UI.Xaml.Media.Imaging.h>
    #ifndef XLLAMA_STORE_SKU
        #include "xllama/api_activity.h"
    #endif

    #include <algorithm>
    #include <chrono>
    #include <cstdio>
    #include <ctime>
    #include <filesystem>
    #include <limits>
    #include <mutex>
    #include <optional>
    #include <string>
    #include <thread>
    #include <vector>

using namespace winrt;
using namespace winrt::Windows::Foundation;
using namespace winrt::Windows::Storage;
using namespace winrt::Windows::UI::Core;
using namespace winrt::Windows::UI::Xaml;
using namespace winrt::Windows::UI::Xaml::Controls;
using namespace winrt::Windows::UI::Text;

namespace xllama {

// First-launch / empty-settings chat model (defined in MainPage.cpp).
std::wstring DefaultChatModelId();

// Wide LocalFolder path helper.
inline std::wstring local_wpath(const wchar_t* filename_w) {
    auto folder = ApplicationData::Current().LocalFolder();
    return std::wstring(folder.Path().c_str()) + L"\\" + filename_w;
}

// #170b: where per-conversation KV snapshots live. Created on demand by the
// save path; every reader tolerates its absence.
inline ::xllama::KvStore kv_store() {
    return ::xllama::KvStore{::xllama::wstring_to_utf8(local_wpath(L"kv"))};
}

inline std::wstring format_diffuse_stage(const std::string& stage) {
    if (stage == "start")
        return L"Starting image generation...";
    if (stage == "text_encoder")
        return L"Text encoder (GPU)...";
    if (stage.rfind("unet ", 0) == 0)
        return L"UNet " + ::xllama::utf8_to_wstring(stage.substr(5)) + L" (GPU)...";
    if (stage == "vae")
        return L"VAE decode (GPU)...";
    if (stage == "done")
        return L"Image ready — open [*] Image to view";
    if (stage == "cancelled")
        return L"Image generation cancelled";
    if (stage == "error")
        return L"Image generation failed — see xllama.log";
    return ::xllama::utf8_to_wstring(stage);
}

// Every editable box in this app is reached with a gamepad, and the on-screen
// keyboard belongs to a deliberate A press — not to focus arriving on its own.
// Without this, opening a dialog moves focus to its first TextBox and the
// keyboard slides up over the dialog's own content: measured, it covered the
// generated image in a Store listing screenshot and the last third of the demo
// capture.
inline void
no_keyboard_on_programmatic_focus(winrt::Windows::UI::Xaml::Controls::TextBox const& box) {
    box.PreventKeyboardDisplayOnProgrammaticFocus(true);
}

} // namespace xllama

#endif // XLLAMA_UWP
