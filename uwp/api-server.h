// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// LAN HTTP endpoint (OpenAI-compatible) exposing xllama::Session on the local
// network. Opt-in, default OFF: started only when LocalState\api.flag exists.
// UWP-only (WinRT StreamSocketListener); no-op on other targets. See
// docs/api-endpoint.md and the plan in the PR description.

#pragma once

#include "xllama/local_knob.h"

#include <string>
#include <string_view>

namespace xllama {
class ApiActivity; // xllama/api_activity.h (kept out of this header: it precedes pch.h)
} // namespace xllama

namespace xllama::api {

enum class ServerState { Stopped, Starting, Running, Error };

struct ServerStatus {
    ServerState state = ServerState::Stopped;
    int port = 0;
    std::string message;
};

inline bool port_bindable(int port) {
    return port >= 1025 && port <= 49151 && port != 11443;
}

inline constexpr int kDefaultApiPort = 11434; // Ollama default port, familiar to clients

// The port LocalState\api-port.txt asks for, as the server binds it and the
// Settings page shows it: a whole integer that port_bindable accepts, else the
// default. One decision, so the page cannot show a port the server refuses.
inline int configured_api_port(std::string_view text) {
    const std::optional<int> p = ::xllama::parse_knob_int(text);
    return (p && port_bindable(*p)) ? *p : kDefaultApiPort;
}

// Bind and return; the global StreamSocketListener keeps accepting callbacks.
// Safe to call repeatedly for the same port. A different port performs a
// stop/rebind. Intended for an MTA background thread, never the UI thread.
void start_server(int port = 0);

// Stop accepting connections and release the API-owned inference Session after
// any active request leaves the single-slot mutex.
void stop_server();

// Thread-safe snapshot for the Settings UI.
ServerStatus server_status();

// Startup entrypoint used by App::OnLaunched: reads api-port.txt and calls
// start_server().
//
// Port: LocalState\api-port.txt if present, else 11434 (Ollama default port).
// Protocol subset (v1): POST /v1/chat/completions (non-streaming), GET / health.
// Concurrency: ConnectionReceived only queues a bounded thread-pool work item
// so a second LAN request can observe 503 while a streamed reply holds the
// process-wide compute permit (ADR 0008).
void run_server();

// Process-wide record of inference traffic (in-flight request, counters, last
// outcome and error) that the main page polls for its LAN API panel.
::xllama::ApiActivity& activity();

} // namespace xllama::api
