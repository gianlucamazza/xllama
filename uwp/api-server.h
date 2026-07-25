// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// LAN HTTP endpoint (OpenAI-compatible) exposing xllama::Session on the local
// network. Opt-in, default OFF: started only when LocalState\api.flag exists.
// UWP-only (WinRT StreamSocketListener); no-op on other targets. See
// docs/api-endpoint.md and the plan in the PR description.

#pragma once

#include <string>

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
// Concurrency: a single shared Session serialized by a mutex; concurrent
// requests get HTTP 503 (Ollama single-slot semantics).
void run_server();

} // namespace xllama::api
