// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// MainPageController: debug dashboard. Split out of MainPage.cpp (#549); shared
// includes and helpers are in mainpage-internal.h.

#include "mainpage-internal.h"

#ifdef XLLAMA_UWP

namespace xllama {

void MainPageController::StartDebugDashboardTimer() {
    if (m_debug_timer)
        return;
    auto weak_self = weak_from_this();
    m_debug_timer = winrt::Windows::UI::Xaml::DispatcherTimer{};
    m_debug_timer.Interval(std::chrono::seconds(1));
    m_debug_tick_token = m_debug_timer.Tick([weak_self](IInspectable const&, IInspectable const&) {
        if (auto s = weak_self.lock())
            s->RefreshDebugDashboard();
    });
    m_debug_timer.Start();
}

void MainPageController::StopDebugDashboardTimer() {
    if (!m_debug_timer)
        return;
    m_debug_timer.Tick(m_debug_tick_token);
    m_debug_timer.Stop();
    m_debug_timer = nullptr;
}

void MainPageController::ApplyDebugDashboard(bool enabled, bool persist_flag) {
    m_debug_dashboard = enabled;
    if (persist_flag) {
        const std::string flag(::xllama::kDebugDashboardFlagName);
        if (enabled)
            ::xllama::write_local_text(flag, "enabled");
        else
            _wremove(local_wpath(::xllama::utf8_to_wstring(flag).c_str()).c_str());
    }
    if (!m_debugDashboardPanel)
        return;
    m_debugDashboardPanel.Visibility(enabled ? Visibility::Visible : Visibility::Collapsed);
    if (enabled) {
        StartDebugDashboardTimer();
        RefreshDebugDashboard();
    } else {
        StopDebugDashboardTimer();
        if (m_debugDashboardText)
            m_debugDashboardText.Text(L"");
    }
}

void MainPageController::RefreshDebugDashboard() {
    if (!m_debug_dashboard || !m_debugDashboardText)
        return;

    ::xllama::DebugDashboardInputs in;
    auto& hub = ::xllama::session_hub();
    in.preloading = hub.preloading.load();
    {
        std::unique_lock<std::mutex> lk(hub.mtx, std::try_to_lock);
        if (lk.owns_lock()) {
            in.hub_snapshot_available = true;
            in.session_loaded = static_cast<bool>(hub.session);
            in.resident_model_id = hub.model;
            if (hub.session) {
                const int loaded_n_ctx = hub.session->context_length();
                in.n_ctx = loaded_n_ctx > 0 ? loaded_n_ctx : hub.params.n_ctx;
                in.n_gpu_layers = hub.params.n_gpu_layers;
                if (!hub.params.lora_path.empty())
                    in.lora_path = hub.params.lora_path;
            }
        }
    }

    #ifndef XLLAMA_STORE_SKU
    in.lan_status_available = true;
    const auto server = ::xllama::api::server_status();
    switch (server.state) {
    case ::xllama::api::ServerState::Starting:
        in.lan_state = ::xllama::DebugLanState::Starting;
        break;
    case ::xllama::api::ServerState::Running:
        in.lan_state = ::xllama::DebugLanState::Running;
        break;
    case ::xllama::api::ServerState::Error:
        in.lan_state = ::xllama::DebugLanState::Error;
        break;
    default:
        in.lan_state = ::xllama::DebugLanState::Stopped;
        break;
    }
    in.lan_port = server.port;
    in.activity_available = true;
    in.activity = ::xllama::api::activity().snapshot();
    #endif

    in.last_inference = m_last_inference;
    in.live_decode_tok_s = m_live_decode_tok_s;
    in.live_tokens = m_live_tokens;
    in.gui_generating = m_is_running.load();
    in.gui_diffusing = m_diffuse_running.load();
    in.gui_training = m_train_running.load();

    const auto snap = ::xllama::compose_debug_dashboard(in);
    m_debugDashboardText.Text(::xllama::utf8_to_wstring(::xllama::format_debug_dashboard(snap)));
}

} // namespace xllama

#endif // XLLAMA_UWP
