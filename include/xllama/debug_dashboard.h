// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// Optional debug dashboard: compose real runtime readings for the UI.
// Callers supply already-observed values (SessionHub under mtx, ApiActivity
// snapshot, last InferenceResult). Missing sources stay unavailable — this
// header does not invent, animate, or placeholder numbers.
// Header-only and WinRT-free so it is host-testable (tests/test_debug_dashboard.cpp).

#pragma once

#include "xllama/api_activity.h"
#include "xllama/inference_params.h"

#include <cstdint>
#include <cstdio>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

namespace xllama {

inline constexpr const char* kDebugDashboardFlagName = "debug-dashboard.flag";
inline constexpr const char* kDebugDashboardSettingsKey = "debug_dashboard";
inline constexpr const char* kDebugUnavailable = "unavailable";

inline std::string_view debug_dashboard_flag_basename() {
    return kDebugDashboardFlagName;
}

inline std::string_view debug_dashboard_settings_key() {
    return kDebugDashboardSettingsKey;
}

// Off unless a persisted setting or a LocalState flag opts in. Either source
// is enough; neither is a host-specific path.
inline bool debug_dashboard_opt_in(bool settings_enabled, bool flag_present) {
    return settings_enabled || flag_present;
}

enum class DebugAdmissionState { Idle, Preloading, Busy };

enum class DebugLanState { Unavailable, Stopped, Starting, Running, Error };

template <typename T> struct DebugMetric {
    bool available = false;
    T value{};

    static DebugMetric make(T v) {
        return DebugMetric{true, std::move(v)};
    }
    static DebugMetric none() {
        return {};
    }
};

struct DebugDashboardInputs {
    // Fill only when the caller observed SessionHub under mtx (or try_lock).
    bool hub_snapshot_available = false;
    bool session_loaded = false;
    std::string resident_model_id;
    std::optional<int> n_ctx;
    std::optional<int> n_gpu_layers;
    std::optional<std::string> lora_path;
    bool preloading = false;

    bool lan_status_available = false;
    DebugLanState lan_state = DebugLanState::Unavailable;
    int lan_port = 0;

    bool activity_available = false;
    ApiActivity::Snapshot activity;

    std::optional<InferenceResult> last_inference;
    std::optional<double> live_decode_tok_s;
    std::optional<int> live_tokens;

    bool gui_generating = false;
    bool gui_diffusing = false;
    bool gui_training = false;

    // No queue exists on current main; leave unset so the snapshot says so.
    std::optional<uint64_t> queue_depth;
};

struct DebugDashboardSnapshot {
    DebugMetric<std::string> resident_model;
    DebugMetric<bool> session_loaded;
    DebugMetric<int> n_ctx;
    DebugMetric<int> n_gpu_layers;
    DebugMetric<std::string> lora_path;
    DebugMetric<bool> preloading;

    DebugMetric<DebugLanState> lan_state;
    DebugMetric<int> lan_port;
    DebugMetric<uint64_t> lan_requests;
    DebugMetric<uint64_t> lan_ok;
    DebugMetric<uint64_t> lan_failed;
    DebugMetric<uint64_t> lan_busy;
    DebugMetric<uint64_t> lan_connected;
    DebugMetric<std::string> lan_in_flight;
    DebugMetric<std::string> lan_last_error;

    DebugMetric<double> decode_tok_s;
    DebugMetric<double> prompt_tok_s;
    DebugMetric<double> t_p_eval_ms;
    DebugMetric<double> t_eval_ms;
    DebugMetric<double> t_first_token_ms;
    DebugMetric<int> n_prompt;
    DebugMetric<int> n_p_eval;
    DebugMetric<int> n_eval;
    DebugMetric<int> live_tokens;
    DebugMetric<double> live_decode_tok_s;

    DebugMetric<DebugAdmissionState> admission;
    DebugMetric<uint64_t> queue_depth;
};

inline DebugMetric<double> decode_tokens_per_second(const InferenceResult& r) {
    if (!r.success || r.n_eval <= 0 || r.t_eval_ms <= 0.0)
        return {};
    return {true, static_cast<double>(r.n_eval) / (r.t_eval_ms / 1000.0)};
}

inline DebugMetric<double> prompt_tokens_per_second(const InferenceResult& r) {
    if (!r.success || r.n_p_eval <= 0 || r.t_p_eval_ms <= 0.0)
        return {};
    return {true, static_cast<double>(r.n_p_eval) / (r.t_p_eval_ms / 1000.0)};
}

inline DebugAdmissionState admission_state(const DebugDashboardInputs& in) {
    if (in.preloading)
        return DebugAdmissionState::Preloading;
    const bool api_busy = in.activity_available && in.activity.in_flight.has_value();
    if (api_busy || in.gui_generating || in.gui_diffusing || in.gui_training)
        return DebugAdmissionState::Busy;
    return DebugAdmissionState::Idle;
}

inline const char* admission_state_label(DebugAdmissionState s) {
    switch (s) {
    case DebugAdmissionState::Preloading:
        return "preloading";
    case DebugAdmissionState::Busy:
        return "busy";
    case DebugAdmissionState::Idle:
        return "idle";
    }
    return kDebugUnavailable;
}

inline const char* lan_state_label(DebugLanState s) {
    switch (s) {
    case DebugLanState::Stopped:
        return "stopped";
    case DebugLanState::Starting:
        return "starting";
    case DebugLanState::Running:
        return "running";
    case DebugLanState::Error:
        return "error";
    case DebugLanState::Unavailable:
        break;
    }
    return kDebugUnavailable;
}

inline DebugDashboardSnapshot compose_debug_dashboard(const DebugDashboardInputs& in) {
    DebugDashboardSnapshot s;
    s.preloading = {true, in.preloading};
    s.admission = {true, admission_state(in)};

    if (in.hub_snapshot_available) {
        s.session_loaded = {true, in.session_loaded};
        if (in.session_loaded) {
            s.resident_model = {true, in.resident_model_id};
            if (in.n_ctx)
                s.n_ctx = {true, *in.n_ctx};
            if (in.n_gpu_layers)
                s.n_gpu_layers = {true, *in.n_gpu_layers};
            s.lora_path = {true, in.lora_path.value_or(std::string{})};
        } else {
            s.resident_model = {true, std::string{}};
        }
    }

    if (in.lan_status_available)
        s.lan_state = {true, in.lan_state};
    if (in.lan_status_available && in.lan_port > 0)
        s.lan_port = {true, in.lan_port};

    if (in.activity_available) {
        s.lan_requests = {true, in.activity.requests};
        s.lan_ok = {true, in.activity.ok};
        s.lan_failed = {true, in.activity.failed};
        s.lan_busy = {true, in.activity.busy};
        s.lan_connected = {true, in.activity.connected};
        if (in.activity.in_flight) {
            std::string line = in.activity.in_flight->endpoint;
            if (!in.activity.in_flight->model.empty())
                line += " · " + in.activity.in_flight->model;
            char buf[32];
            std::snprintf(buf, sizeof(buf), " · %.1fs", in.activity.in_flight->elapsed_s);
            line += buf;
            s.lan_in_flight = {true, std::move(line)};
        } else {
            s.lan_in_flight = {true, std::string("idle")};
        }
        if (in.activity.last_error)
            s.lan_last_error = {true, in.activity.last_error->message};
        else
            s.lan_last_error = {true, std::string{}};
    }

    if (in.last_inference) {
        const InferenceResult& r = *in.last_inference;
        s.decode_tok_s = decode_tokens_per_second(r);
        s.prompt_tok_s = prompt_tokens_per_second(r);
        if (r.t_p_eval_ms > 0.0)
            s.t_p_eval_ms = {true, r.t_p_eval_ms};
        if (r.t_eval_ms > 0.0)
            s.t_eval_ms = {true, r.t_eval_ms};
        if (r.t_first_token_ms > 0.0)
            s.t_first_token_ms = {true, r.t_first_token_ms};
        if (r.n_prompt > 0)
            s.n_prompt = {true, r.n_prompt};
        if (r.n_p_eval > 0)
            s.n_p_eval = {true, r.n_p_eval};
        if (r.n_eval > 0)
            s.n_eval = {true, r.n_eval};
    }

    if (in.live_decode_tok_s)
        s.live_decode_tok_s = {true, *in.live_decode_tok_s};
    if (in.live_tokens)
        s.live_tokens = {true, *in.live_tokens};
    if (in.queue_depth)
        s.queue_depth = {true, *in.queue_depth};
    return s;
}

inline std::string format_debug_bool(const DebugMetric<bool>& m) {
    if (!m.available)
        return kDebugUnavailable;
    return m.value ? "yes" : "no";
}

inline std::string format_debug_text(const DebugMetric<std::string>& m, const char* empty_label) {
    if (!m.available)
        return kDebugUnavailable;
    if (m.value.empty())
        return empty_label;
    return m.value;
}

template <typename T> std::string format_debug_count(const DebugMetric<T>& m) {
    if (!m.available)
        return kDebugUnavailable;
    return std::to_string(m.value);
}

inline std::string format_debug_fixed(const DebugMetric<double>& m, const char* fmt) {
    if (!m.available)
        return kDebugUnavailable;
    char buf[64];
    std::snprintf(buf, sizeof(buf), fmt, m.value);
    return buf;
}

// Multi-line UTF-8 for the optional UI panel. Labels stay stable so tests and
// the console can grep the same strings.
inline std::string format_debug_dashboard(const DebugDashboardSnapshot& s) {
    std::ostringstream out;
    out << "Model\n";
    out << "  resident: " << format_debug_text(s.resident_model, "none") << "\n";
    out << "  loaded: " << format_debug_bool(s.session_loaded) << "\n";
    out << "  n_ctx: " << format_debug_count(s.n_ctx) << "\n";
    out << "  n_gpu_layers: " << format_debug_count(s.n_gpu_layers) << "\n";
    out << "  lora: " << format_debug_text(s.lora_path, "none") << "\n";
    out << "  preloading: " << format_debug_bool(s.preloading) << "\n";
    out << "LAN\n";
    out << "  state: "
        << (s.lan_state.available ? lan_state_label(s.lan_state.value) : kDebugUnavailable) << "\n";
    out << "  port: " << format_debug_count(s.lan_port) << "\n";
    out << "  connected clients: " << format_debug_count(s.lan_connected) << "\n";
    out << "  requests: " << format_debug_count(s.lan_requests) << "\n";
    out << "  ok: " << format_debug_count(s.lan_ok) << "\n";
    out << "  failed: " << format_debug_count(s.lan_failed) << "\n";
    out << "  busy: " << format_debug_count(s.lan_busy) << "\n";
    out << "  in flight: " << format_debug_text(s.lan_in_flight, "idle") << "\n";
    out << "  last error: " << format_debug_text(s.lan_last_error, "none") << "\n";
    out << "Inference\n";
    out << "  decode tok/s: " << format_debug_fixed(s.decode_tok_s, "%.1f") << "\n";
    out << "  prompt tok/s: " << format_debug_fixed(s.prompt_tok_s, "%.1f") << "\n";
    out << "  prompt time: " << format_debug_fixed(s.t_p_eval_ms, "%.1f ms") << "\n";
    out << "  eval time: " << format_debug_fixed(s.t_eval_ms, "%.1f ms") << "\n";
    out << "  time to first token: " << format_debug_fixed(s.t_first_token_ms, "%.1f ms") << "\n";
    out << "  prompt tokens: " << format_debug_count(s.n_prompt) << "\n";
    out << "  prefill tokens: " << format_debug_count(s.n_p_eval) << "\n";
    out << "  decode tokens: " << format_debug_count(s.n_eval) << "\n";
    out << "  live decode tok/s: " << format_debug_fixed(s.live_decode_tok_s, "%.1f") << "\n";
    out << "  live tokens: " << format_debug_count(s.live_tokens) << "\n";
    out << "Admission\n";
    out << "  state: "
        << (s.admission.available ? admission_state_label(s.admission.value) : kDebugUnavailable)
        << "\n";
    out << "  queue: " << format_debug_count(s.queue_depth) << "\n";
    return out.str();
}

} // namespace xllama
