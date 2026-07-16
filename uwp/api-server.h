// Copyright (c) 2024 Venere Labs
// SPDX-License-Identifier: MIT
//
// LAN HTTP endpoint (OpenAI-compatible) exposing xllama::Session on the local
// network. Opt-in, default OFF: started only when LocalState\api.flag exists.
// UWP-only (WinRT StreamSocketListener); no-op on other targets. See
// docs/api-endpoint.md and the plan in the PR description.

#pragma once

namespace xllama::api {

// Start the LAN HTTP listener and serve until the process exits. Intended to be
// called on a detached MTA background thread from App::OnLaunched when
// LocalState\api.flag is present. Blocks its thread; the StreamSocketListener is
// kept alive for the app lifetime and coexists with the live XAML chat UI.
//
// Port: LocalState\api-port.txt if present, else 11434 (Ollama default port).
// Protocol subset (v1): POST /v1/chat/completions (non-streaming), GET / health.
// Concurrency: a single shared Session serialized by a mutex; concurrent
// requests get HTTP 503 (Ollama single-slot semantics).
void run_server();

} // namespace xllama::api
