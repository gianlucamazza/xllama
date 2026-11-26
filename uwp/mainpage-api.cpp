// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// MainPageController: LAN API panel. Split out of MainPage.cpp (#549); shared
// includes and helpers are in mainpage-internal.h.

#include "mainpage-internal.h"

#ifdef XLLAMA_UWP

namespace xllama {

    #ifndef XLLAMA_STORE_SKU

// The console's LAN IPv4: the address on the adapter of the internet profile,
// else the first IPv4 host name. Empty if none (shown as a placeholder).
static std::wstring ResolveLanIpv4() {
    using namespace winrt::Windows::Networking;
    using namespace winrt::Windows::Networking::Connectivity;
    try {
        winrt::guid adapter{};
        bool have_adapter = false;
        if (auto profile = NetworkInformation::GetInternetConnectionProfile();
            profile && profile.NetworkAdapter()) {
            adapter = profile.NetworkAdapter().NetworkAdapterId();
            have_adapter = true;
        }
        std::wstring fallback;
        for (auto const& host : NetworkInformation::GetHostNames()) {
            if (host.Type() != HostNameType::Ipv4 || !host.IPInformation())
                continue;
            std::wstring ip(host.CanonicalName().c_str());
            auto nic = host.IPInformation().NetworkAdapter();
            if (have_adapter && nic && nic.NetworkAdapterId() == adapter)
                return ip;
            if (fallback.empty())
                fallback = ip;
        }
        return fallback;
    } catch (...) {
        return {};
    }
}

static std::wstring FormatSeconds(double seconds) {
    wchar_t buf[32];
    if (seconds < 60.0)
        swprintf(buf, 32, L"%.1fs", seconds);
    else
        swprintf(buf, 32, L"%dm%02ds", static_cast<int>(seconds) / 60,
                 static_cast<int>(seconds) % 60);
    return buf;
}

void MainPageController::StartApiPanel() {
    auto weak_self = weak_from_this();
    auto dispatcher = m_root.Dispatcher();
    // Host names are a synchronous WinRT query: resolve once, off the UI thread.
    std::thread([weak_self, dispatcher]() {
        winrt::init_apartment();
        std::wstring ip = ResolveLanIpv4();
        dispatcher.RunAsync(CoreDispatcherPriority::Normal, [weak_self, ip]() {
            if (auto s = weak_self.lock()) {
                s->m_lan_ip = ip;
                s->RefreshApiPanel();
            }
        });
    }).detach();

    m_api_timer = winrt::Windows::UI::Xaml::DispatcherTimer{};
    m_api_timer.Interval(std::chrono::seconds(1));
    m_api_tick_token = m_api_timer.Tick([weak_self](IInspectable const&, IInspectable const&) {
        if (auto s = weak_self.lock())
            s->RefreshApiPanel();
    });
    m_api_timer.Start();
    RefreshApiPanel();
}

void MainPageController::RefreshApiPanel() {
    namespace Media = winrt::Windows::UI::Xaml::Media;
    const auto server = ::xllama::api::server_status();
    if (server.state == ::xllama::api::ServerState::Stopped) {
        m_apiPanel.Visibility(Visibility::Collapsed);
        return;
    }
    m_apiPanel.Visibility(Visibility::Visible);

    winrt::Windows::UI::Color color{255, 100, 220, 100}; // listening: green
    if (server.state == ::xllama::api::ServerState::Starting)
        color = {255, 240, 190, 60}; // amber
    else if (server.state == ::xllama::api::ServerState::Error)
        color = {255, 240, 80, 70}; // red
    const std::wstring host = m_lan_ip.empty() ? L"<console IP>" : m_lan_ip;
    const std::wstring port = server.port > 0 ? L":" + std::to_wstring(server.port) : L"";
    m_apiEndpointText.Text(L"\u25CF LAN API  http://" + host + port + L"/v1");
    m_apiEndpointText.Foreground(Media::SolidColorBrush(color));

    const auto snap = ::xllama::api::activity().snapshot();
    if (server.state == ::xllama::api::ServerState::Error) {
        m_apiNowText.Text(::xllama::utf8_to_wstring(server.message));
    } else if (server.state == ::xllama::api::ServerState::Starting) {
        m_apiNowText.Text(L"starting...");
    } else if (snap.in_flight) {
        std::wstring now = L"\u25B6 " + ::xllama::utf8_to_wstring(snap.in_flight->endpoint);
        if (!snap.in_flight->model.empty())
            now += L" \u00B7 " + ::xllama::utf8_to_wstring(snap.in_flight->model);
        m_apiNowText.Text(now + L" \u00B7 " + FormatSeconds(snap.in_flight->elapsed_s));
    } else {
        m_apiNowText.Text(L"idle");
    }

    std::wstring counts = std::to_wstring(snap.requests) +
                          (snap.requests == 1 ? L" request" : L" requests") + L" \u00B7 " +
                          std::to_wstring(snap.busy) + L" busy";
    if (snap.last) {
        counts += L" \u00B7 last " + FormatSeconds(snap.last->elapsed_s);
        if (snap.last->prompt_tokens >= 0 && snap.last->completion_tokens >= 0)
            counts += L" " + std::to_wstring(snap.last->prompt_tokens) + L"\u2192" +
                      std::to_wstring(snap.last->completion_tokens) + L" tok";
    }
    m_apiCountsText.Text(counts);

    // A recent error stays visible for 10 minutes, then the panel calms down.
    if (snap.last_error && snap.last_error->age_s < 600.0) {
        m_apiErrorText.Text(L"! " + ::xllama::utf8_to_wstring(snap.last_error->message) + L" (" +
                            FormatSeconds(snap.last_error->age_s) + L" ago)");
        m_apiErrorText.Visibility(Visibility::Visible);
    } else {
        m_apiErrorText.Visibility(Visibility::Collapsed);
    }
}

void MainPageController::ApplyApiSettings(bool enabled, int port) {
    if (enabled) {
        ::xllama::write_local_text("api-port.txt", std::to_string(port));
        ::xllama::write_local_text("api.flag", "enabled");
    } else {
        _wremove(local_wpath(L"api.flag").c_str());
    }

    SetStatus(enabled ? L"Starting LAN API..." : L"Stopping LAN API...", StatusKind::Working);
    auto self = shared_from_this();
    auto dispatcher = m_root.Dispatcher();
    const uint64_t generation = ++m_api_settings_generation;
    std::thread([self, dispatcher, enabled, port, generation]() {
        winrt::init_apartment();
        std::lock_guard<std::mutex> lifecycle_lock(self->m_api_settings_mutex);
        if (generation != self->m_api_settings_generation.load())
            return;
        if (enabled)
            ::xllama::api::start_server(port);
        else
            ::xllama::api::stop_server();
        const auto status = ::xllama::api::server_status();
        dispatcher.RunAsync(CoreDispatcherPriority::Normal, [self, status, generation]() {
            if (generation != self->m_api_settings_generation.load())
                return;
            if (status.state == ::xllama::api::ServerState::Running) {
                self->SetStatus(L"LAN API listening on port " + std::to_wstring(status.port),
                                StatusKind::Success);
            } else if (status.state == ::xllama::api::ServerState::Stopped) {
                self->SetStatus(L"LAN API stopped", StatusKind::Success);
            } else {
                self->SetStatus(L"LAN API error: " + ::xllama::utf8_to_wstring(status.message),
                                StatusKind::Error);
            }
        });
    }).detach();
}

    #endif // !XLLAMA_STORE_SKU

} // namespace xllama

#endif // XLLAMA_UWP
