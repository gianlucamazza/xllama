// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// LAN HTTP endpoint (OpenAI-compatible) — implementation. UWP-only.
// A new front-end on xllama::Session: it does no inference of its own, it maps
// JSON <-> Session/chat_prompt. See api-server.h.

#include "api-server.h"

#ifdef XLLAMA_UWP

// clang-format off
// pch.h must be first: it includes <unknwn.h> before winrt/base.h (COM IUnknown
// guard). Sockets + Json are the WinRT namespaces pch.h does not already pull in.
#include "pch.h"
#include <atomic>
#include <winrt/Windows.ApplicationModel.h>
#include <winrt/Windows.Data.Json.h>
#include <winrt/Windows.Networking.Sockets.h>
#include <winrt/Windows.System.Threading.h>
// clang-format on

    #include "inference-bridge.h"
    #include "model-downloader.h"
    #include "xllama/api_activity.h"
    #include "xllama/api_connection.h"
    #include "xllama/api_discovery.h"
    #include "xllama/api_failure.h"
    #include "xllama/api_ollama.h"
    #include "xllama/api_policy.h"
    #include "xllama/api_pull_policy.h"
    #include "xllama/api_sampling.h"
    #include "xllama/chat_prompt.h"
    #include "xllama/compute_admission.h"
    #include "xllama/diffusion/request.h"
    #include "xllama/diffusion/residency.h"
    #include "xllama/http_framing.h"
    #include "xllama/local_knob.h"
    #include "xllama/model_provision.h"
    #include "xllama/path_utils.h"
    #include "xllama/personalize.h"
    #include "xllama/platform.h"
    #include "xllama/preference_capture.h"
    #include "xllama/prompt_budget.h"
    #include "xllama/reasoning_control.h"
    #include "xllama/routing_policy.h"
    #include "xllama/session.h"
    #include "xllama/session_hub.h"
    #include "xllama/session_residency.h"
    #include "xllama/sse.h"
    #include "xllama/stop_sequence_filter.h"
    #include "xllama/thinking_split.h"
    #include "xllama/tool_budget.h"
    #include "xllama/tool_request.h"
    #include "xllama/tool_template.h"
    #include "xllama/tool_transcript.h"
    #include "xllama/utf8_utils.h"

    #include <algorithm>
    #include <cctype>
    #include <charconv>
    #include <chrono>
    #include <climits>
    #include <cmath>
    #include <cstdio>
    #include <cstdlib>
    #include <ctime>
    #include <exception>
    #include <filesystem>
    #include <memory>
    #include <mutex>
    #include <optional>
    #include <string>
    #include <string_view>
    #include <unordered_map>
    #include <utility>
    #include <vector>

namespace xllama::api {
namespace {

using namespace winrt::Windows::Networking::Sockets;
using namespace winrt::Windows::Storage::Streams;
using namespace winrt::Windows::Data::Json;

// ---------------------------------------------------------------------------
// Shared single-slot inference state
//
// The Session lives in xllama::session_hub() — ONE process-wide owner shared
// with the GUI (previously each surface owned its own Session and both could
// hold a model at once, breaking the "never 2× model in RAM" invariant).
// The compute permit serializes resource-heavy jobs across UI and API; hub.mtx
// additionally protects Session identity and use. A request that finds either
// slot busy gets 503. Keep the lock order compute permit before hub.mtx.
// ---------------------------------------------------------------------------
std::atomic<uint64_t> g_req_counter{0}; // monotonic, for unique response ids

// Acquire the hub after compute admission. Busy (for example, a KV snapshot
// save) → unowned lock, caller answers 503. Startup pre-load waiting belongs
// to acquire_compute_job_or_busy(), before this mutex is touched.
std::unique_lock<std::mutex> acquire_hub_or_busy() {
    auto& hub = ::xllama::session_hub();
    return std::unique_lock<std::mutex>(hub.mtx, std::try_to_lock);
}

// Acquire the process-wide compute lease before touching the SessionHub. The
// startup pre-load is the only admitted job that API clients wait behind;
// every other occupied slot keeps the existing bounded busy response. The
// returned handle must stay alive for the whole reply, including SSE tokens.
::xllama::ComputeJobAdmission::Handle acquire_compute_job_or_busy() {
    return ::xllama::try_acquire_compute_job_while_preloading(::xllama::session_hub().preloading);
}

std::mutex g_state_mtx;
ServerStatus g_status;
uint64_t g_generation = 0; // invalidates callbacks accepted by an older listener

// Settings and autopilot can request lifecycle changes from separate worker
// threads. Serialize bind/close so only one transition owns the listener.
std::mutex g_control_mtx;

// Pulls share one writer/staging directory per model. Reject a second request
// rather than allowing overlapping writes to the same .part files.
::xllama::ApiPullGate g_pull_gate;

// Keep the listener alive for the process lifetime (callbacks fire on the WinRT
// thread pool, not on run_server's thread).
StreamSocketListener g_listener{nullptr};

// Xbox blocks binding ports in [57344, 65535] (traffic is silently dropped, per
// the UWP-on-Xbox known-issues doc); 11443 is the Device Portal. Reject those so
// an api-port.txt typo fails loudly to the default rather than binding a dead
// port. Valid app range is [1025, 49151].
int listen_port() {
    const std::string s = ::xllama::read_local_text("api-port.txt");
    const int port = configured_api_port(s);
    if (!s.empty() && ::xllama::parse_knob_int(s) != port) // unparsable or not bindable
        ::xllama::log_output("[xllama] api: api-port.txt=" + s +
                             " is not a port in [1025,49151]\\11443; using default\n");
    return port;
}

// ---------------------------------------------------------------------------
// Minimal HTTP request read
// ---------------------------------------------------------------------------
struct HttpRequest {
    std::string method;
    std::string path;
    std::string body;
    // 0 = a complete, exactly framed request. Otherwise the 4xx to answer with
    // (parse_http_head's codes, 400 for a short body, 408 past a deadline).
    int status = 400;
    std::string error = "malformed request";
    // A read still pending when a deadline passed. It is cancelled only after
    // the 408 is written: cancelling first aborts the stream, and the client
    // sees a reset instead of the status (console, 1.6.0.1234).
    winrt::Windows::Storage::Streams::DataReaderLoadOperation pending{nullptr};
};

// Reads one request: the head until "\r\n\r\n", then exactly Content-Length
// body bytes. Framing decisions are xllama::parse_http_head (ADR 0006, host
// tested); this only moves bytes and enforces the head and body deadlines, so
// a client that stalls cannot hold the thread. Blocking; called on a
// thread-pool thread, never the UI thread.
HttpRequest read_request(StreamSocket const& socket) {
    using clock = std::chrono::steady_clock;
    HttpRequest req;
    DataReader reader(socket.InputStream());
    reader.InputStreamOptions(InputStreamOptions::Partial);

    std::string data;
    // Bytes read before `deadline`; 0 at EOF, -1 when the deadline passed.
    auto load = [&](clock::time_point deadline) -> int64_t {
        const auto now = clock::now();
        if (now >= deadline)
            return -1;
        auto op = reader.LoadAsync(4096);
        if (op.wait_for(std::chrono::duration_cast<winrt::Windows::Foundation::TimeSpan>(
                deadline - now)) != winrt::Windows::Foundation::AsyncStatus::Completed) {
            req.pending = op;
            return -1;
        }
        const uint32_t got = op.GetResults();
        for (uint32_t i = 0; i < got; ++i)
            data.push_back(static_cast<char>(reader.ReadByte()));
        return got;
    };
    auto timed_out = [&req] {
        req.status = 408;
        req.error = "request not received in time";
        return req;
    };

    const auto head_deadline = clock::now() + ::xllama::kHttpHeadDeadline;
    while (::xllama::http_head_end(data) == std::string::npos &&
           data.size() < ::xllama::kHttpMaxHead) {
        const int64_t got = load(head_deadline);
        if (got < 0)
            return timed_out();
        if (got == 0)
            break;
    }
    const ::xllama::HttpHead head = ::xllama::parse_http_head(data);
    if (head.status != 0) {
        req.status = head.status;
        req.error = head.error;
        return req;
    }
    req.method = head.method;
    req.path = head.path;

    const auto body_deadline = clock::now() + ::xllama::kHttpBodyDeadline;
    while (data.size() - head.body_start < head.content_length) {
        const int64_t got = load(body_deadline);
        if (got < 0)
            return timed_out();
        if (got == 0) {
            req.error = "request body shorter than Content-Length";
            return req; // 400: nothing is dispatched on a partial body
        }
    }
    // Bytes past the declared length are not a second request: one per connection.
    req.body = data.substr(head.body_start, head.content_length);
    req.status = 0;
    req.error.clear();
    return req;
}

void write_raw(StreamSocket const& socket, const std::string& out) {
    DataWriter writer(socket.OutputStream());
    writer.UnicodeEncoding(UnicodeEncoding::Utf8);
    writer.WriteString(winrt::to_hstring(out));
    writer.StoreAsync().get();
    writer.FlushAsync().get();
    writer.DetachStream();
}

void write_response(StreamSocket const& socket, const char* status, const std::string& json) {
    std::string out = "HTTP/1.1 ";
    out += status;
    out += "\r\nContent-Type: application/json\r\nContent-Length: ";
    out += std::to_string(json.size());
    // Allow-Origin alone is not enough for browser clients: they send a custom
    // Authorization header + application/json, which triggers a CORS preflight
    // (handled separately in handle_connection).
    out += "\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n";
    out += json;
    write_raw(socket, out);
}

// The 200 head of a native Ollama stream (pull progress, chat and generate).
constexpr const char* kNdjsonResponseHead =
    "HTTP/1.1 200 OK\r\nContent-Type: application/x-ndjson\r\n"
    "Transfer-Encoding: chunked\r\nConnection: close\r\n"
    "Access-Control-Allow-Origin: *\r\n\r\n";

class PullJsonStream {
  public:
    explicit PullJsonStream(StreamSocket const& socket) {
        write_raw(socket, kNdjsonResponseHead);
        m_writer = DataWriter(socket.OutputStream());
        m_writer.UnicodeEncoding(UnicodeEncoding::Utf8);
    }

    bool write_line(const std::string& json) {
        if (m_finished)
            return false;
        const std::string payload = json + "\n";
        char size_hex[32]{};
        const auto converted =
            std::to_chars(size_hex, size_hex + sizeof(size_hex), payload.size(), 16);
        if (converted.ec != std::errc{})
            return false;
        std::string frame(size_hex, converted.ptr);
        frame += "\r\n";
        frame += payload;
        frame += "\r\n";
        try {
            m_writer.WriteString(winrt::to_hstring(frame));
            m_writer.StoreAsync().get();
            m_writer.FlushAsync().get();
            return true;
        } catch (...) {
            return false;
        }
    }

    void finish() {
        if (m_finished)
            return;
        m_finished = true;
        try {
            m_writer.WriteString(L"0\r\n\r\n");
            m_writer.StoreAsync().get();
            m_writer.FlushAsync().get();
            m_writer.DetachStream();
        } catch (...) {
        }
    }

  private:
    DataWriter m_writer{nullptr};
    bool m_finished = false;
};

std::string pull_event_json(const std::string& status, uint64_t completed = 0, uint64_t total = 0) {
    JsonObject root;
    root.Insert(L"status", JsonValue::CreateStringValue(winrt::to_hstring(status)));
    if (total > 0) {
        root.Insert(L"completed", JsonValue::CreateNumberValue(static_cast<double>(completed)));
        root.Insert(L"total", JsonValue::CreateNumberValue(static_cast<double>(total)));
    }
    return winrt::to_string(root.Stringify());
}

using ::xllama::ollama_error_json;

// Reads `response_format` for the decision in api_policy.h
// (api_response_format_supported).
bool accepts_response_format(JsonObject const& root) {
    using Kind = ::xllama::ApiResponseFormat::Kind;
    ::xllama::ApiResponseFormat format;
    if (root.HasKey(L"response_format")) {
        const auto value = root.GetNamedValue(L"response_format");
        if (value.ValueType() == JsonValueType::Null) {
            format.kind = Kind::Null;
        } else if (value.ValueType() == JsonValueType::Object) {
            const auto object = value.GetObject();
            format.kind = Kind::Object;
            format.members = object.Size();
            if (object.HasKey(L"type") &&
                object.GetNamedValue(L"type").ValueType() == JsonValueType::String) {
                format.type_is_string = true;
                format.type = winrt::to_string(object.GetNamedString(L"type"));
            }
        } else {
            format.kind = Kind::Other;
        }
    }
    return ::xllama::api_response_format_supported(format);
}

// The wire a chat reply is written in: OpenAI chat.completion(.chunk) over SSE,
// or native Ollama /api/chat and /api/generate objects over NDJSON. The request
// itself is the same canonical chat body for all three (api_ollama.h).
using ::xllama::ChatWire;
using ChatClock = std::chrono::steady_clock;

// Per-request reply state: the native wire's timings, and what the request's
// keep_alive does to the session once the reply is written.
struct ChatResponse {
    ChatWire wire = ChatWire::OpenAI;
    ChatClock::time_point started = ChatClock::now();
    double load_ms = 0; // the model load this request paid, if any
    // Set once the request holds the session (keep_alive_action); applied by
    // KeepAliveAtEnd under the same permit and hub lock.
    std::optional<::xllama::KeepAliveAction> keep_alive;
    std::chrono::milliseconds lease{0};
};

bool native_wire(const ChatResponse& response) {
    return response.wire != ChatWire::OpenAI;
}

enum class StreamHeadStatus { Written, FailedBeforeBytes, FailedUnknown };

// Writes the 200 head before ChatStream exists, so a failure can still answer
// with JSON. `FailedBeforeBytes` is reported only when the failure is provably
// before anything reached the socket (DataWriter construction or encoding);
// once StoreAsync starts, a partial head cannot be ruled out, so the caller must
// close instead of writing a second HTTP response.
StreamHeadStatus write_chat_head(StreamSocket const& socket, ChatWire wire) {
    try {
        DataWriter writer(socket.OutputStream());
        writer.UnicodeEncoding(UnicodeEncoding::Utf8);
        writer.WriteString(winrt::to_hstring(wire == ChatWire::OpenAI
                                                 ? ::xllama::sse_response_head()
                                                 : std::string(kNdjsonResponseHead)));
        try {
            writer.StoreAsync().get();
            writer.FlushAsync().get();
            writer.DetachStream();
            return StreamHeadStatus::Written;
        } catch (...) {
            return StreamHeadStatus::FailedUnknown;
        }
    } catch (...) {
        return StreamHeadStatus::FailedBeforeBytes;
    }
}

// The token counts of a chat reply's usage object, JSON and SSE alike.
struct ChatUsage {
    int prompt = 0;     // the prompt's length (n_prompt)
    int completion = 0; // generated tokens (n_eval)
    // The part of completion spent inside the reasoning (#585); 0 for a model
    // that does not reason.
    int reasoning = 0;
    // Prompt tokens the session did not prefill because its resident KV already
    // held them (#170a prefix reuse, #595): n_prompt - n_p_eval. A GGUF session
    // reuses a matching prefix on every path, tool proposals included; an ORT
    // session prefills the whole prompt, so it is 0 there.
    int cached = 0;
};

ChatUsage chat_usage(const ::xllama::InferenceResult& r, int reasoning_tokens) {
    ChatUsage u;
    u.prompt = r.n_prompt;
    u.completion = r.n_eval;
    u.reasoning = reasoning_tokens;
    u.cached = std::clamp(r.n_prompt - r.n_p_eval, 0, r.n_prompt);
    return u;
}

// OpenAI's usage shape; both details objects are always sent, with zeros when
// nothing applies, as OpenAI does.
JsonObject usage_json(const ChatUsage& u) {
    JsonObject prompt_details;
    prompt_details.Insert(L"cached_tokens", JsonValue::CreateNumberValue(u.cached));
    JsonObject completion_details;
    completion_details.Insert(L"reasoning_tokens", JsonValue::CreateNumberValue(u.reasoning));
    JsonObject usage;
    usage.Insert(L"prompt_tokens", JsonValue::CreateNumberValue(u.prompt));
    usage.Insert(L"completion_tokens", JsonValue::CreateNumberValue(u.completion));
    usage.Insert(L"total_tokens", JsonValue::CreateNumberValue(u.prompt + u.completion));
    usage.Insert(L"prompt_tokens_details", prompt_details);
    usage.Insert(L"completion_tokens_details", completion_details);
    return usage;
}

winrt::hstring utc_timestamp(std::time_t time) {
    std::tm utc{};
    char text[32]{};
    if (gmtime_s(&utc, &time) == 0)
        std::strftime(text, sizeof(text), "%Y-%m-%dT%H:%M:%SZ", &utc);
    return winrt::to_hstring(text);
}

// One native Ollama reply object. /api/chat carries |message| (role, content,
// thinking, tool_calls); /api/generate flattens it to `response` and
// `thinking`. |reason| set means the final object: done, done_reason and,
// with |result|, the token counts and durations (nanoseconds) Ollama reports.
JsonObject ollama_chat_json(const ChatResponse& response, const std::string& model,
                            JsonObject const& message, const wchar_t* reason,
                            const ::xllama::InferenceResult* result = nullptr) {
    JsonObject root;
    root.Insert(L"model", JsonValue::CreateStringValue(winrt::to_hstring(model)));
    root.Insert(L"created_at", JsonValue::CreateStringValue(utc_timestamp(std::time(nullptr))));
    if (response.wire == ChatWire::OllamaChat) {
        JsonObject out;
        out.Insert(L"role", JsonValue::CreateStringValue(L"assistant"));
        out.Insert(L"content",
                   JsonValue::CreateStringValue(message.GetNamedString(L"content", L"")));
        for (const wchar_t* key : {L"thinking", L"tool_calls"})
            if (message.HasKey(key))
                out.Insert(key, message.GetNamedValue(key));
        root.Insert(L"message", out);
    } else {
        root.Insert(L"response",
                    JsonValue::CreateStringValue(message.GetNamedString(L"content", L"")));
        if (message.HasKey(L"thinking"))
            root.Insert(L"thinking", message.GetNamedValue(L"thinking"));
    }
    root.Insert(L"done", JsonValue::CreateBooleanValue(reason != nullptr));
    if (reason)
        root.Insert(L"done_reason", JsonValue::CreateStringValue(reason));
    if (result) {
        const auto metric = [&](const wchar_t* key, double value) {
            root.Insert(key, JsonValue::CreateNumberValue(value));
        };
        const auto total = std::chrono::duration_cast<std::chrono::nanoseconds>(ChatClock::now() -
                                                                                response.started)
                               .count();
        metric(L"total_duration", static_cast<double>(total));
        metric(L"load_duration", std::round(response.load_ms * 1e6));
        metric(L"prompt_eval_count", result->n_prompt);
        metric(L"prompt_eval_duration", std::round(result->t_p_eval_ms * 1e6));
        metric(L"eval_count", result->n_eval);
        metric(L"eval_duration", std::round(result->t_eval_ms * 1e6));
    }
    return root;
}

// Ollama has no tool-call finish: a reply that proposes calls is "stop".
const wchar_t* ollama_done_reason(std::wstring_view reason) {
    return reason == L"length" ? L"length" : L"stop";
}

// ---------------------------------------------------------------------------
// Ollama keep_alive. What a lease may do is decided in session_residency.h
// (the console app's session never expires); this is only its timer. The
// lease itself lives in hub.residency under hub.mtx; the timer carries a
// snapshot (generation, id) that a reload, a newer lease, a cancel or a GUI
// turn makes stale. Lock order: compute permit -> hub mutex -> timer mutex.
// ---------------------------------------------------------------------------
using winrt::Windows::System::Threading::ThreadPoolTimer;
std::mutex g_lease_timer_mtx;
ThreadPoolTimer g_lease_timer{nullptr}; // guarded by g_lease_timer_mtx
constexpr std::chrono::milliseconds kLeaseRetry{250};

ThreadPoolTimer make_lease_timer(std::chrono::milliseconds delay, uint64_t generation, uint64_t id);

// A lease's timer fired. It takes compute admission and the hub like any other
// job, so it never releases a session a turn is using; while either is busy
// it only re-arms itself.
void expire_lease(ThreadPoolTimer const& timer, uint64_t generation, uint64_t id) {
    auto permit = ::xllama::try_acquire_compute_job();
    auto hub_lock = permit ? acquire_hub_or_busy() : std::unique_lock<std::mutex>{};
    std::lock_guard<std::mutex> timer_lock(g_lease_timer_mtx);
    if (timer != g_lease_timer)
        return; // a newer lease or a cancel replaced this timer
    if (!hub_lock.owns_lock()) {
        g_lease_timer = make_lease_timer(kLeaseRetry, generation, id);
        return;
    }
    auto& hub = ::xllama::session_hub();
    const ::xllama::SessionResidency& lease = hub.residency;
    if (lease.deadline && lease.may_expire() && lease.id == id && hub.generation == generation) {
        if (!lease.expired(generation, id, ChatClock::now())) {
            g_lease_timer = make_lease_timer(kLeaseRetry, generation, id); // fired early
            return;
        }
        ::xllama::log_output("[xllama] api: keep_alive expired, releasing " + hub.model + "\n");
        hub.reset_locked();
    }
    g_lease_timer = nullptr;
}

ThreadPoolTimer make_lease_timer(std::chrono::milliseconds delay, uint64_t generation,
                                 uint64_t id) {
    return ThreadPoolTimer::CreateTimer(
        [generation, id](ThreadPoolTimer const& timer) {
            try {
                expire_lease(timer, generation, id);
            } catch (...) {
                ::xllama::log_output("[xllama] api: keep_alive timer failed\n");
            }
        },
        std::chrono::duration_cast<winrt::Windows::Foundation::TimeSpan>(delay));
}

// Under the compute permit and hub.mtx.
void cancel_lease_timer_locked() {
    ThreadPoolTimer timer{nullptr};
    {
        std::lock_guard<std::mutex> lock(g_lease_timer_mtx);
        timer = std::exchange(g_lease_timer, nullptr);
    }
    if (timer)
        timer.Cancel();
}

// Under the compute permit and hub.mtx: what a request's keep_alive does to the
// session it used, once its reply is written.
void apply_keep_alive_locked(::xllama::KeepAliveAction action, std::chrono::milliseconds lease) {
    using Action = ::xllama::KeepAliveAction;
    auto& hub = ::xllama::session_hub();
    cancel_lease_timer_locked();
    switch (action) {
    case Action::Unload:
        if (hub.residency.owner != ::xllama::SessionOwner::Gui)
            hub.reset_locked();
        return;
    case Action::Arm:
        if (hub.residency.arm(lease, hub.generation, ChatClock::now())) {
            std::lock_guard<std::mutex> lock(g_lease_timer_mtx);
            g_lease_timer = make_lease_timer(lease, hub.generation, hub.residency.id);
        }
        return;
    case Action::Keep:
        hub.residency.cancel();
        return;
    case Action::Refuse:
        return;
    }
}

// Applies the request's keep_alive when the reply ends, a client disconnect or
// a throw included. Declared after the hub lock, so it runs before the lock
// and the compute permit are released.
struct KeepAliveAtEnd {
    const ChatResponse& response;
    ~KeepAliveAtEnd() {
        if (!response.keep_alive)
            return;
        try {
            apply_keep_alive_locked(*response.keep_alive, response.lease);
        } catch (...) {
            ::xllama::session_hub().residency.cancel();
            ::xllama::log_output("[xllama] api: keep_alive could not be applied\n");
        }
    }
};

// The console app holds the session: a client may not unload it.
constexpr const char* kSessionInUseByConsole = "model is in use by the console app";

enum class StreamUsage { Legacy, Omit, Separate };

// An absent or null options object retains the historical finish-chunk usage.
std::optional<StreamUsage> stream_usage(JsonObject const& root, bool stream) {
    if (!root.HasKey(L"stream_options"))
        return StreamUsage::Legacy;
    const auto value = root.GetNamedValue(L"stream_options");
    if (value.ValueType() == JsonValueType::Null)
        return StreamUsage::Legacy;
    if (!stream || value.ValueType() != JsonValueType::Object)
        return std::nullopt;
    // Other keys (the SDKs' include_obfuscation, future ones) are ignored, as
    // OpenAI-compatible servers do; only include_usage changes the stream.
    const auto options = value.GetObject();
    if (!options.HasKey(L"include_usage"))
        return StreamUsage::Legacy;
    const auto include = options.GetNamedValue(L"include_usage");
    if (include.ValueType() != JsonValueType::Boolean)
        return std::nullopt;
    return include.GetBoolean() ? StreamUsage::Separate : StreamUsage::Omit;
}

// Streamed chat reply (docs/api-endpoint.md). OpenAI: token deltas, finish
// choice, optional usage summary and [DONE] as SSE events. Native Ollama: one
// NDJSON object per delta, then a final `done: true` object with the counts.
// A streamed response carries no Content-Length, so either travels in HTTP/1.1
// chunks (xllama::chunk_frame, host tested) and the socket closes at the end.
//
// Plain chat emits token pieces during generation. Tool proposals buffer the
// model output until the template parser has assembled the reply.
class ChatStream {
  public:
    ChatStream(StreamSocket const& socket, std::string model, std::atomic<bool>* abort,
               StreamUsage mode, const ChatResponse& response)
        : m_model(std::move(model)), m_usage_mode(mode), m_response(response),
          m_id("chatcmpl-xllama-" + std::to_string(time(nullptr)) + "-" +
               std::to_string(++g_req_counter)),
          m_abort(abort) {
        m_writer = DataWriter(socket.OutputStream());
        m_writer.UnicodeEncoding(UnicodeEncoding::Utf8);
        // Both wires open with the assistant role and empty content, even when
        // generation produces no tokens: the client sees the reply has begun.
        JsonObject delta;
        delta.Insert(L"role", JsonValue::CreateStringValue(L"assistant"));
        delta.Insert(L"content", JsonValue::CreateStringValue(L""));
        write_message_delta(delta);
        // A long prefill writes nothing, and a client that left is noticed only
        // when a write fails: a comment every few silent seconds is that write,
        // and its failure sets |abort|, which ends the prefill (#580, #581).
        // NDJSON has no comment line, so the native wire repeats an empty
        // `done: false` object. The frame is built here; the keepalive thread
        // only writes bytes.
        if (!m_finished) {
            const std::string heartbeat =
                native_wire(m_response)
                    ? winrt::to_string(ollama_chat_json(m_response, m_model, JsonObject{}, nullptr)
                                           .Stringify()) +
                          "\n"
                    : ::xllama::sse_comment("keepalive");
            m_keepalive.emplace(
                ::xllama::kSseKeepaliveInterval,
                [this, frame = ::xllama::chunk_frame(heartbeat)] { return write_frame(frame); });
        }
    }

    ~ChatStream() {
        stop_keepalive();
    }
    ChatStream(const ChatStream&) = delete;
    ChatStream& operator=(const ChatStream&) = delete;

    // False when the role chunk could not be written: the client is gone before
    // generation even starts.
    bool started() const {
        return !m_finished;
    }

    // A token piece can end midway through a multibyte character
    // (llama_token_to_piece returns bytes, not code points). Buffer the trailing
    // bytes and emit only complete UTF-8 sequences; an invalid byte becomes
    // U+FFFD and is dropped so the rest of the reply keeps flowing.
    // |reasoning| sends the piece as delta.reasoning_content (#564). Each field
    // keeps its own split-sequence buffer, so a code point is never emitted
    // under the wrong key.
    void write_delta(std::string_view piece, bool reasoning = false) {
        if (m_finished || piece.empty())
            return;
        (reasoning ? m_pending_reasoning : m_pending).append(piece.data(), piece.size());
        try {
            drain_pending(reasoning, false);
        } catch (...) {
            stop();
        }
    }

    void write_message_delta(JsonObject const& delta) noexcept {
        if (m_finished)
            return;
        try {
            if (native_wire(m_response))
                send(winrt::to_string(
                    ollama_chat_json(m_response, m_model, delta, nullptr).Stringify()));
            else
                send_chunk(delta, JsonValue::CreateNullValue(), nullptr);
        } catch (...) {
            stop();
        }
    }

    // The stream is already 200 by the time generation fails, so the failure
    // travels as the last event instead of an HTTP status. Never throws: it is
    // the recovery path, and a second HTTP response is no longer possible.
    // Native Ollama ends on an {"error":"..."} line instead.
    void write_error(const std::string& message) noexcept {
        if (m_finished)
            return;
        try {
            end(&message);
        } catch (...) {
            stop();
        }
    }

    void finish(const wchar_t* reason, const ::xllama::InferenceResult& result,
                int reasoning_tokens) {
        if (m_finished)
            return;
        try {
            drain_pending(true, true);
            drain_pending(false, true);
        } catch (...) {
            stop();
        }
        if (m_finished)
            return;
        try {
            if (native_wire(m_response)) {
                send(winrt::to_string(ollama_chat_json(m_response, m_model, JsonObject{},
                                                       ollama_done_reason(reason), &result)
                                          .Stringify()));
                end();
                return;
            }
            const JsonObject usage = usage_json(chat_usage(result, reasoning_tokens));
            send_chunk(JsonObject{}, JsonValue::CreateStringValue(winrt::hstring(reason)),
                       m_usage_mode == StreamUsage::Legacy ? &usage : nullptr);
            if (!m_finished && m_usage_mode == StreamUsage::Separate)
                send_choices(JsonArray{}, &usage);
            end();
        } catch (...) {
            stop();
        }
    }

  private:
    // Emits one delta chunk from text already known to end on a code-point
    // boundary.
    void emit_delta(std::string_view text, bool reasoning) {
        JsonObject delta;
        const wchar_t* reasoning_key = native_wire(m_response) ? L"thinking" : L"reasoning_content";
        delta.Insert(reasoning ? reasoning_key : L"content",
                     JsonValue::CreateStringValue(winrt::to_hstring(text)));
        write_message_delta(delta);
    }

    // Emits complete code points, replaces invalid bytes and advances, and at
    // finish replaces any truncated tail. Looping is what keeps a malformed byte
    // from blocking the bytes behind it.
    void drain_pending(bool reasoning, bool finishing) {
        std::string& pending = reasoning ? m_pending_reasoning : m_pending;
        for (;;) {
            if (m_finished || pending.empty())
                return;
            const ::xllama::SseUtf8Step step = ::xllama::sse_utf8_step(pending, finishing);
            if (step.wait)
                return;
            if (step.replace) {
                pending.erase(0, 1);
                emit_delta("\xEF\xBF\xBD", reasoning);
                continue;
            }
            const std::string text = pending.substr(0, step.take);
            pending.erase(0, step.take);
            emit_delta(text, reasoning);
        }
    }

    // Joins the keepalive thread; never called from it.
    void stop_keepalive() noexcept {
        if (m_keepalive)
            m_keepalive->stop();
    }

    // One framed write, serialized with the keepalive thread. False, and the
    // reply stopped, when the client is gone.
    bool write_frame(const std::string& framed) noexcept {
        std::lock_guard<std::mutex> lock(m_write_mu);
        if (m_finished)
            return false;
        try {
            m_writer.WriteString(winrt::to_hstring(framed));
            m_writer.StoreAsync().get();
            m_writer.FlushAsync().get();
        } catch (...) {
            stop();
            return false;
        }
        if (m_keepalive)
            m_keepalive->touch();
        return true;
    }

    // A dead socket ends the reply; generation is aborted so the session lock is
    // released instead of running to EOS behind a client that is already gone.
    void stop() noexcept {
        m_finished = true;
        if (m_abort)
            m_abort->store(true);
    }

    void send_chunk(JsonObject const& delta, IJsonValue const& finish_reason,
                    JsonObject const* usage) {
        JsonObject choice;
        choice.Insert(L"index", JsonValue::CreateNumberValue(0));
        choice.Insert(L"delta", delta);
        choice.Insert(L"logprobs", JsonValue::CreateNullValue());
        choice.Insert(L"finish_reason", finish_reason);
        JsonArray choices;
        choices.Append(choice);
        send_choices(choices, usage);
    }

    void send_choices(JsonArray const& choices, JsonObject const* usage) {
        JsonObject chunk;
        chunk.Insert(L"id", JsonValue::CreateStringValue(winrt::to_hstring(m_id)));
        chunk.Insert(L"object", JsonValue::CreateStringValue(L"chat.completion.chunk"));
        chunk.Insert(L"created", JsonValue::CreateNumberValue(static_cast<double>(time(nullptr))));
        chunk.Insert(L"model", JsonValue::CreateStringValue(winrt::to_hstring(m_model)));
        chunk.Insert(L"choices", choices);
        if (usage != nullptr)
            chunk.Insert(L"usage", *usage);
        else if (m_usage_mode == StreamUsage::Separate)
            chunk.Insert(L"usage", JsonValue::CreateNullValue());
        send(winrt::to_string(chunk.Stringify()));
    }

    // One payload, which must hold no raw line break on either wire: an SSE
    // data line or an NDJSON line.
    bool send(const std::string& payload) {
        if (m_finished || !::xllama::sse_payload_is_safe(payload))
            return false;
        return write_frame(::xllama::chunk_frame(
            native_wire(m_response) ? payload + "\n" : ::xllama::sse_event(payload)));
    }

    void end(const std::string* error = nullptr) {
        if (m_finished)
            return;
        stop_keepalive(); // nothing may follow the final chunk
        std::lock_guard<std::mutex> lock(m_write_mu);
        if (m_finished)
            return;
        m_finished = true;
        try {
            m_writer.WriteString(
                winrt::to_hstring(::xllama::chat_terminal_frames(m_response.wire, error)));
            m_writer.StoreAsync().get();
            m_writer.FlushAsync().get();
            m_writer.DetachStream();
        } catch (...) {
            if (m_abort)
                m_abort->store(true);
        }
    }

    DataWriter m_writer{nullptr};
    std::string m_model;
    StreamUsage m_usage_mode;
    ChatResponse m_response; // wire and timings
    std::string m_id;
    std::string m_pending;           // trailing bytes of a split UTF-8 sequence (content)
    std::string m_pending_reasoning; // the same for reasoning_content
    std::atomic<bool>* m_abort = nullptr;
    // Written by the request thread and the keepalive thread; frames are
    // serialized by m_write_mu.
    std::atomic<bool> m_finished{false};
    std::mutex m_write_mu;
    std::optional<::xllama::SseKeepalive> m_keepalive; // last: stopped first
};

void write_cors_preflight(StreamSocket const& socket) {
    write_raw(socket, "HTTP/1.1 204 No Content\r\n"
                      "Access-Control-Allow-Origin: *\r\n"
                      "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
                      "Access-Control-Allow-Headers: Authorization, Content-Type\r\n"
                      "Access-Control-Allow-Private-Network: true\r\n"
                      "Access-Control-Max-Age: 86400\r\n"
                      "Content-Length: 0\r\nConnection: close\r\n\r\n");
}

std::string error_json(const std::string& msg, const char* code = nullptr,
                       const char* param = nullptr) {
    JsonObject err;
    err.Insert(L"message", JsonValue::CreateStringValue(winrt::to_hstring(msg)));
    err.Insert(L"type", JsonValue::CreateStringValue(L"xllama_error"));
    if (code && *code)
        err.Insert(L"code", JsonValue::CreateStringValue(winrt::to_hstring(code)));
    if (param && *param)
        err.Insert(L"param", JsonValue::CreateStringValue(winrt::to_hstring(param)));
    JsonObject root;
    root.Insert(L"error", err);
    return winrt::to_string(root.Stringify());
}

// Every request check is done when this runs: from here on a streamed reply owns
// the socket. Writes the 200 head, then opens the stream (role chunk and
// keepalive, whose failed write sets |abort|). Null when the stream could not
// start; |status| is then set and |body| is what the caller returns: JSON only
// when the head provably left no byte, otherwise empty, because a partial head
// may be on the wire and a second HTTP response cannot follow it.
std::unique_ptr<ChatStream> open_chat_stream(StreamSocket const& socket, const std::string& model,
                                             std::atomic<bool>* abort, JsonObject const& root,
                                             const ChatResponse& response, const char*& status,
                                             std::string& body) {
    // The chat handler validated the immutable options before model selection.
    const auto mode = stream_usage(root, true).value();
    body.clear();
    switch (write_chat_head(socket, response.wire)) {
    case StreamHeadStatus::Written:
        break;
    case StreamHeadStatus::FailedBeforeBytes:
        status = "500 Internal Server Error";
        body = error_json("failed to start the stream");
        return nullptr;
    case StreamHeadStatus::FailedUnknown:
        status = "500 Internal Server Error";
        return nullptr;
    }
    std::unique_ptr<ChatStream> sse;
    try {
        sse = std::make_unique<ChatStream>(socket, model, abort, mode, response);
    } catch (...) {
        status = "500 Internal Server Error";
        return nullptr; // the 200 head is already out
    }
    if (!sse->started()) {
        status = "500 Internal Server Error";
        return nullptr; // the role chunk could not be written
    }
    return sse;
}

// #591: the status for an exception that escaped a route handler, logged so a
// 5xx is diagnosable from xllama.log. Call only from inside a catch handler.
// Only a Json accessor failure is a 400; out of memory is 503, the rest 500.
::xllama::ApiFailureReply reply_for_current_exception(const char* route, const char* malformed,
                                                      const char* internal) {
    ::xllama::ApiFailure kind = ::xllama::ApiFailure::Internal;
    std::string detail;
    try {
        throw;
    } catch (winrt::hresult_error const& e) {
        const auto hr = static_cast<std::int32_t>(e.code().value);
        kind = ::xllama::classify_api_hresult(hr);
        char buf[32];
        snprintf(buf, sizeof(buf), "hresult 0x%08X", static_cast<unsigned>(hr));
        detail = buf;
        try {
            const std::string text = winrt::to_string(e.message());
            if (!text.empty())
                detail += ": " + text;
        } catch (...) {
        }
    } catch (...) {
        kind = ::xllama::classify_std_exception(std::current_exception(), detail);
    }
    ::xllama::log_output(std::string("[xllama] api: ") + route + " failed (" +
                         ::xllama::api_failure_name(kind) + "): " + detail + "\n");
    return ::xllama::api_failure_reply(kind, malformed, internal);
}

// ---------------------------------------------------------------------------
// OpenAI /v1/chat/completions
// ---------------------------------------------------------------------------

// Catalogue session policy for one model, cached: parsing the manifest (bundled
// file + LocalState override) on EVERY chat request, under hub.mtx and in front
// of the model load, is pure latency. Only called from handle_chat_locked, which
// holds hub.mtx — that lock is what serialises the statics below.
//
// A miss re-reads the file, so a model published (or uploaded) while the server
// runs is picked up on its first request; models genuinely absent from the
// catalogue re-read once per request, exactly as every request did before.
struct CatalogueSessionPolicy {
    int n_ctx = ::xllama::kDefaultNCtx;
    bool coding = false;
    bool gguf = false;
    bool embedding = false;
    // The catalogue's sampling layer: a request field it omits takes this.
    ::xllama::SamplingOverrides sampling;
};

CatalogueSessionPolicy catalogue_session_policy(const std::string& model) {
    static std::vector<::xllama::ManifestEntry> cache;
    const std::wstring wname = ::xllama::utf8_to_wstring(model);
    const ::xllama::ManifestEntry* entry = ::xllama::FindManifestEntry(cache, wname);
    if (!entry) {
        cache = ::xllama::LoadModelManifest();
        entry = ::xllama::FindManifestEntry(cache, wname);
    }
    CatalogueSessionPolicy p;
    if (entry) {
        p.gguf = entry->kind == L"gguf";
        p.n_ctx = ::xllama::catalogue_n_ctx(entry->n_ctx, p.gguf);
        p.coding = ::xllama::role_is_coding(::xllama::wstring_to_utf8(entry->role));
        p.embedding = ::xllama::role_is_embedding(::xllama::wstring_to_utf8(entry->role));
        p.sampling = entry->sampling;
    }
    return p;
}

// True when the console can serve `model` now: the resident model, or a
// LocalState\models\<id> directory with a loadable layout. Raw paths (model.txt
// may name one) are left to the session loader. The caller holds hub.mtx, so
// this must not call current_model_id() (it try_locks the same mutex).
bool model_servable_locked(const std::string& model) {
    if (model.find_first_of("\\/:") != std::string::npos)
        return true;
    if (::xllama::session_hub().model == model)
        return true;
    std::error_code ec;
    const std::filesystem::path dir =
        std::filesystem::path(::xllama::resolve_local_path("models")) / model;
    if (!std::filesystem::is_directory(dir, ec))
        return false;
    std::vector<std::string> files;
    for (const auto& f : std::filesystem::directory_iterator(dir, ec))
        files.push_back(f.path().filename().string());
    return ::xllama::model_dir_files_ready(files);
}

std::string model_not_found(const std::string& model) {
    return "model '" + model + "' is not on this console; see GET /v1/models";
}

void append_client_stops(JsonObject const& root, std::vector<std::string>& stops) {
    if (!root.HasKey(L"stop"))
        return;
    const auto sv = root.GetNamedValue(L"stop");
    if (sv.ValueType() == JsonValueType::String) {
        stops.push_back(winrt::to_string(sv.GetString()));
    } else if (sv.ValueType() == JsonValueType::Array) {
        for (auto&& e : sv.GetArray())
            if (e.ValueType() == JsonValueType::String)
                stops.push_back(winrt::to_string(e.GetString()));
    }
}

// The reply's token count. max_tokens is deprecated in favour of
// max_completion_tokens; accept both. Validated against the catalogue context
// before any model load (F01): a negative, fractional, or context-sized value
// is the client's error. Absent = default_max_tokens (512, less on a reduced
// context). A native Ollama request carried it as options.num_predict
// (api_ollama.h), and its errors name that field.
::xllama::ApiTokenLimit read_max_tokens(JsonObject const& root, int n_ctx, bool native = false) {
    for (const char* key : {"max_completion_tokens", "max_tokens"}) {
        const winrt::hstring name = winrt::to_hstring(key);
        if (root.HasKey(name))
            return ::xllama::validate_max_tokens(native ? "options.num_predict" : key,
                                                 root.GetNamedNumber(name), n_ctx);
    }
    ::xllama::ApiTokenLimit out;
    out.value = ::xllama::default_max_tokens(n_ctx);
    return out;
}

// options.num_ctx: a native Ollama option, and an xllama extension on
// /v1/chat/completions and the embedding routes. Absent leaves |value| empty.
// It may only reduce |configured|, the catalogue context
// (validate_context_override); other option keys are not read here.
std::string read_context_override(JsonObject const& root, int configured,
                                  std::optional<int>& value) {
    value.reset();
    if (!root.HasKey(L"options") ||
        root.GetNamedValue(L"options").ValueType() == JsonValueType::Null)
        return {};
    if (root.GetNamedValue(L"options").ValueType() != JsonValueType::Object)
        return "options must be an object";
    const JsonObject options = root.GetNamedObject(L"options");
    if (!options.HasKey(L"num_ctx") ||
        options.GetNamedValue(L"num_ctx").ValueType() == JsonValueType::Null)
        return {};
    if (options.GetNamedValue(L"num_ctx").ValueType() != JsonValueType::Number)
        return "options.num_ctx must be an integer";
    const double n_ctx = options.GetNamedNumber(L"num_ctx");
    std::string error = ::xllama::validate_context_override(n_ctx, configured);
    if (error.empty())
        value = static_cast<int>(n_ctx);
    return error;
}

// The chat request's model id, as handle_chat_locked resolves it.
std::string request_model(JsonObject const& root) {
    std::string model = winrt::to_string(root.GetNamedString(L"model", L""));
    return model.empty() ? ::xllama::read_local_text("model.txt") : model;
}

// Before compute admission: a context the model's catalogue entry cannot take
// is the client's error, not a reason to wait behind inference.
std::string check_context_override(JsonObject const& root) {
    std::optional<int> unused;
    const std::string model = request_model(root);
    const int configured =
        model.empty() ? ::xllama::kDefaultNCtx : ::xllama::CatalogueSessionParams(model).n_ctx;
    return read_context_override(root, configured, unused);
}

// The request's numbers, for the WinRT-free sampling table (api_sampling.h).
::xllama::ApiNumberLookup json_numbers(JsonObject const& root) {
    return [&root](std::string_view key) -> std::optional<double> {
        const winrt::hstring name = winrt::to_hstring(std::string(key));
        if (!root.HasKey(name))
            return std::nullopt;
        return root.GetNamedNumber(name);
    };
}

// Sampling numbers, validated before any model load (F01): the fields and
// ranges of xllama/api_sampling.h, which /health also reports.
std::string check_sampling(JsonObject const& root) {
    return ::xllama::validate_api_sampling(json_numbers(root));
}

// The model's values (model_sampling.h: global < model), then the sampling
// numbers that check_sampling already accepted, so a field the request omits
// keeps the model's value.
void fill_sampling(JsonObject const& root, const ::xllama::SamplingOverrides& model_sampling,
                   ::xllama::GenerateParams& gp) {
    ::xllama::apply_resolved_sampling(::xllama::resolve_sampling(model_sampling), gp);
    ::xllama::apply_api_sampling(json_numbers(root), gp);
}

// One OpenAI chat.completion body. `message` is choices[0].message.
std::string completion_json(const std::string& model, JsonObject const& message,
                            winrt::hstring const& finish_reason, const ChatUsage& usage,
                            const char*& status) {
    JsonObject choice;
    choice.Insert(L"index", JsonValue::CreateNumberValue(0));
    choice.Insert(L"message", message);
    // openai-python / LangChain deserialize into Pydantic models and choke when
    // logprobs is absent; emit it explicitly as null.
    choice.Insert(L"logprobs", JsonValue::CreateNullValue());
    choice.Insert(L"finish_reason", JsonValue::CreateStringValue(finish_reason));
    JsonArray choices;
    choices.Append(choice);

    JsonObject resp;
    const std::string id =
        "chatcmpl-xllama-" + std::to_string(time(nullptr)) + "-" + std::to_string(++g_req_counter);
    resp.Insert(L"id", JsonValue::CreateStringValue(winrt::to_hstring(id)));
    resp.Insert(L"object", JsonValue::CreateStringValue(L"chat.completion"));
    resp.Insert(L"created", JsonValue::CreateNumberValue(static_cast<double>(time(nullptr))));
    resp.Insert(L"model", JsonValue::CreateStringValue(winrt::to_hstring(model)));
    resp.Insert(L"choices", choices);
    resp.Insert(L"usage", usage_json(usage));
    status = "200 OK";
    return winrt::to_string(resp.Stringify());
}

// The non-streamed reply on the request's wire: an OpenAI chat.completion, or
// one native Ollama object with done: true.
std::string chat_reply_json(const ChatResponse& response, const std::string& model,
                            JsonObject const& message, winrt::hstring const& finish_reason,
                            const ::xllama::InferenceResult& result, int reasoning_tokens,
                            const char*& status) {
    if (!native_wire(response))
        return completion_json(model, message, finish_reason, chat_usage(result, reasoning_tokens),
                               status);
    status = "200 OK";
    return winrt::to_string(
        ollama_chat_json(response, model, message, ollama_done_reason(finish_reason), &result)
            .Stringify());
}

// Generates, parses and answers one tool proposal. With |sse| the stream is
// already open, so a failure is an SSE error event and the return is empty;
// without it, the return is the JSON response.
std::string write_tool_proposal(::xllama::GenerateParams& gp, const ::xllama::ToolRequest& request,
                                const std::vector<::xllama::ChatMessage>& messages,
                                ::xllama::Session* session, const std::string& model, bool thinking,
                                ::xllama::ApiActivity::Ticket& ticket, const char*& status,
                                std::chrono::steady_clock::time_point started,
                                ::xllama::ReasoningTokenCounter& reasoning_tokens, ChatStream* sse,
                                const std::atomic<bool>& abort, const ChatResponse& response) {
    using Clock = std::chrono::steady_clock;
    const bool native = native_wire(response);
    const bool stream = sse != nullptr;
    const auto prepared = Clock::now();
    const ::xllama::InferenceResult generated = session->generate(gp);
    const auto generated_at = Clock::now();
    ticket.tokens(generated.n_prompt, generated.n_eval);
    if (!generated.success) {
        status = "500 Internal Server Error";
        if (stream) {
            sse->write_error("generation failed: " + generated.error_msg);
            return {};
        }
        return error_json("generation failed: " + generated.error_msg);
    }
    if (stream && abort.load()) {
        // The client left mid-decode: nobody reads the parsed reply.
        status = "200 OK";
        return {};
    }
    const bool hit_cap = generated.hit_token_limit(gp.n_predict);
    const ::xllama::ToolParseResult parsed = ::xllama::parse_tool_generation(
        session->resident_llama_model(), thinking, "", messages, request.tools, request.choice,
        request.forced_name, generated.output_text, hit_cap);
    const auto parsed_at = Clock::now();
    const auto elapsed_us = [](Clock::time_point begin, Clock::time_point end) {
        return std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    };
    ::xllama::log_output(
        "[xllama] api: tool_proposal prepare_us=" + std::to_string(elapsed_us(started, prepared)) +
        " generate_us=" + std::to_string(elapsed_us(prepared, generated_at)) +
        " parse_us=" + std::to_string(elapsed_us(generated_at, parsed_at)) + "\n");
    const ::xllama::ChoiceAssembly assembly = ::xllama::assemble_choice(
        generated.output_text, parsed.content, parsed.calls, hit_cap, thinking, parsed.reasoning);
    const int n_reasoning = reasoning_tokens.count(thinking, parsed.reasoning);

    // Native Ollama content is always a string (empty beside tool calls), its
    // reasoning is `thinking`, and a call's arguments are an object.
    JsonObject message;
    if (assembly.content_null && !native) {
        if (!stream)
            message.Insert(L"content", JsonValue::CreateNullValue());
    } else {
        message.Insert(L"content",
                       JsonValue::CreateStringValue(winrt::to_hstring(assembly.content)));
    }
    if (!assembly.reasoning.empty())
        message.Insert(native ? L"thinking" : L"reasoning_content",
                       JsonValue::CreateStringValue(winrt::to_hstring(assembly.reasoning)));
    if (!assembly.tool_calls.empty()) {
        JsonArray calls;
        for (const ::xllama::ToolCall& call : assembly.tool_calls) {
            JsonObject function;
            function.Insert(L"name", JsonValue::CreateStringValue(winrt::to_hstring(call.name)));
            JsonObject tool_call;
            if (native) {
                JsonObject arguments{nullptr};
                if (!JsonObject::TryParse(winrt::to_hstring(call.arguments_json), arguments) ||
                    arguments == nullptr) {
                    const char* error = "model emitted tool arguments that are not an object";
                    status = "500 Internal Server Error";
                    if (stream) {
                        sse->write_error(error);
                        return {};
                    }
                    return error_json(error);
                }
                function.Insert(L"index", JsonValue::CreateNumberValue(calls.Size()));
                function.Insert(L"arguments", arguments);
            } else {
                function.Insert(L"arguments", JsonValue::CreateStringValue(
                                                  winrt::to_hstring(call.arguments_json)));
                if (stream)
                    tool_call.Insert(L"index", JsonValue::CreateNumberValue(calls.Size()));
                tool_call.Insert(L"id", JsonValue::CreateStringValue(winrt::to_hstring(call.id)));
            }
            tool_call.Insert(L"type", JsonValue::CreateStringValue(L"function"));
            tool_call.Insert(L"function", function);
            calls.Append(tool_call);
        }
        message.Insert(L"tool_calls", calls);
    }
    const auto finish_reason = winrt::to_hstring(assembly.finish_reason);
    if (!stream) {
        message.Insert(L"role", JsonValue::CreateStringValue(L"assistant"));
        return chat_reply_json(response, model, message, finish_reason, generated, n_reasoning,
                               status);
    }

    // The parser has assembled the whole reply: it leaves as one delta, never as
    // raw model markup.
    sse->write_message_delta(message);
    sse->finish(finish_reason.c_str(), generated, n_reasoning);
    status = "200 OK";
    return {};
}

// Client-side tool proposal. The process does not run the tool. One generate,
// constrained by the template's own grammar (ADR 0005), then its parser. KV reuse stays off: the
// schema lives in the template's system prefix, so a pinned plain-chat prefix would be wrong.
std::string handle_tool_proposal_locked(
    JsonObject const& root, ::xllama::ToolRequest& request, ::xllama::Session* session,
    const std::string& model, bool thinking, bool model_is_coding, int policy_n_ctx, int n_predict,
    const ::xllama::SamplingOverrides& model_sampling, ::xllama::ApiActivity::Ticket& ticket,
    const char*& status, StreamSocket const& socket, bool stream, const ChatResponse& response) {
    using Clock = std::chrono::steady_clock;
    const auto started = Clock::now();
    bool has_system = false;
    for (const ::xllama::ChatMessage& message : request.messages) {
        if (message.role == ::xllama::ChatRole::System) {
            has_system = true;
            break;
        }
    }
    if (!has_system) {
        ::xllama::ChatMessage system;
        system.role = ::xllama::ChatRole::System;
        system.content =
            model_is_coding ? ::xllama::kCodingSystemPrompt : ::xllama::kDefaultSystemPrompt;
        request.messages.insert(request.messages.begin(), std::move(system));
    }

    ::xllama::GenerateParams gp;
    gp.reuse_kv = false;
    gp.n_predict = n_predict;
    fill_sampling(root, model_sampling, gp);
    ::xllama::ToolTemplateFit fit;
    try {
        const void* llama_model = session->resident_llama_model();
        fit = ::xllama::fit_tool_template(
            request.messages, request.tools, policy_n_ctx, gp.n_predict,
            [&](const std::vector<::xllama::ChatMessage>& messages,
                const std::vector<::xllama::ToolSpec>& tools) {
                return ::xllama::render_tool_prompt(llama_model, thinking, "", messages, tools,
                                                    request.choice, request.forced_name);
            },
            [session](const std::string& text) { return session->count_tokens(text); });
    } catch (const std::bad_alloc&) {
        throw; // #591: the route handler answers 503, not a 400 naming bad_alloc
    } catch (const std::exception& ex) {
        status = "400 Bad Request";
        return error_json(ex.what());
    }
    if (!fit.rendered.ok) {
        status = "400 Bad Request";
        return error_json(fit.rendered.error);
    }
    if (!fit.fits) {
        status = "400 Bad Request";
        const ::xllama::ToolBudgetError overflow = ::xllama::tool_prompt_too_long(
            fit.n_tokens, fit.n_tokens_without_tools, gp.n_predict, policy_n_ctx);
        return error_json(overflow.message, overflow.code, overflow.param);
    }
    if (fit.dropped > 0)
        ::xllama::log_output("[xllama] api: dropped " + std::to_string(fit.dropped) +
                             " oldest message(s) to fit n_ctx " + std::to_string(policy_n_ctx) +
                             "\n");
    gp.prompt = std::move(fit.rendered.prompt);
    gp.stop_sequences = std::move(fit.rendered.stop_sequences);
    gp.preserved_tokens = std::move(fit.rendered.preserved_tokens);
    gp.grammar = std::move(fit.rendered.grammar); // ADR 0005
    append_client_stops(root, gp.stop_sequences);

    // Every token's piece, for usage.completion_tokens_details (#585).
    ::xllama::ReasoningTokenCounter reasoning_tokens;
    gp.on_token = [&reasoning_tokens](std::string_view piece) { reasoning_tokens.push(piece); };

    // Every request check is done, so a streamed proposal opens its stream now,
    // before generation (#594). The keepalive then watches the socket through a
    // long prefill, and a client that leaves aborts it as on plain chat (#581).
    // Generation and parse failures travel as an SSE error event from here on.
    std::atomic<bool> abort{false};
    std::unique_ptr<ChatStream> sse;
    if (stream) {
        std::string failed_body;
        sse = open_chat_stream(socket, model, &abort, root, response, status, failed_body);
        if (!sse)
            return failed_body;
        gp.abort_flag = &abort;
        try {
            write_tool_proposal(gp, request, fit.messages, session, model, thinking, ticket, status,
                                started, reasoning_tokens, sse.get(), abort, response);
        } catch (...) {
            const ::xllama::ApiFailureReply failure = reply_for_current_exception(
                "chat tool stream", "malformed request body", "generation failed");
            status = failure.status;
            sse->write_error(failure.message);
        }
        return {}; // the stream owns the response
    }
    return write_tool_proposal(gp, request, fit.messages, session, model, thinking, ticket, status,
                               started, reasoning_tokens, nullptr, abort, response);
}

// Handles one chat request under session_hub().mtx (already locked by the caller).
// |native| is the parsed Ollama request for /api/chat and /api/generate (its
// canonical body is |body|), null for /v1/chat/completions. |thinking| is the
// request's effective reasoning, decided before admission (decide_reasoning,
// reasoning_control.h); plain chat and a tool proposal both render it. Errors
// return the OpenAI error shape; handle_connection rewrites them for a native
// client.
std::string handle_chat_locked(JsonObject const& root, const std::string& body, const char*& status,
                               ::xllama::ApiActivity::Ticket& ticket, StreamSocket const& socket,
                               bool stream, ChatResponse& response,
                               const ::xllama::OllamaRequest* native, bool thinking) {
    // A tool proposal is parsed here and executed by the client. This process
    // does not run tools. Plain chat (no schema, empty tools, tool_choice none)
    // stays on ChatFormat below. A native transcript that replays tool calls
    // without offering tools again still renders through the GGUF template.
    ::xllama::ToolRequest request =
        ::xllama::parse_tool_request(body, native && native->replay_without_tools);
    if (!request.error.empty()) {
        status = "400 Bad Request";
        return error_json(request.error);
    }
    const bool proposal = request.disposition == ::xllama::ApiToolDisposition::Proposal ||
                          request.disposition == ::xllama::ApiToolDisposition::Legacy;

    const std::string model = request_model(root); // model.txt: field-test fallback
    if (model.empty()) {
        status = "400 Bad Request";
        return error_json("missing 'model' (and no LocalState\\model.txt fallback)");
    }
    ticket.model(model);
    if (!model_servable_locked(model)) {
        status = "404 Not Found"; // OpenAI: model_not_found; was a 500 with a path
        return error_json(model_not_found(model));
    }

    // An empty native request with keep_alive 0 unloads the model if the API
    // holds it; the console app's session stays (session_residency.h).
    auto& hub = ::xllama::session_hub();
    if (native && native->action == ::xllama::OllamaAction::Unload) {
        if (hub.session && hub.model == model) {
            if (::xllama::keep_alive_action(hub.residency.owner, native->keep_alive) ==
                ::xllama::KeepAliveAction::Refuse) {
                status = "409 Conflict";
                return error_json(kSessionInUseByConsole);
            }
            cancel_lease_timer_locked();
            hub.reset_locked();
        }
        status = "200 OK";
        return winrt::to_string(
            ollama_chat_json(response, model, JsonObject{}, L"unload").Stringify());
    }

    // Roles and content parts follow the tool parser's rules (tool_request.h).
    // A native load (no messages, no prompt) only loads the model; an unload
    // returned above.
    const bool load_only = native && native->action == ::xllama::OllamaAction::Load;
    std::string system, final_user;
    std::vector<::xllama::ChatTurn> history;
    if (!proposal && !load_only) {
        ::xllama::PlainChatRequest plain = ::xllama::parse_plain_chat(body);
        if (!plain.error.empty()) {
            status = "400 Bad Request";
            return error_json(plain.error);
        }
        system = std::move(plain.system);
        history = std::move(plain.history);
        final_user = std::move(plain.final_user);
        if (final_user.empty()) {
            status = "400 Bad Request";
            return error_json("no user message to complete");
        }
    }
    // Lazily (re)create the resident Session when the requested model differs
    // (hub.mtx is held by the caller; the swap invalidates the GUI's KV-reuse
    // state via hub.generation, which its next turn detects). Catalogue n_ctx
    // / role (coding) apply the same policy as the chat UI.
    ::xllama::Session* session = nullptr;
    bool model_is_coding = false;
    int policy_n_ctx = ::xllama::kDefaultNCtx;
    int n_predict = 0;
    ::xllama::SamplingOverrides model_sampling;
    {
        std::string err;
        const CatalogueSessionPolicy policy = catalogue_session_policy(model);

        // Embedding models must not be used for chat completion.
        if (policy.embedding) {
            status = "400 Bad Request";
            return error_json("model '" + model +
                              "' is an embedding model and cannot be used for chat completions; "
                              "use /api/embed or /v1/embeddings instead");
        }

        model_is_coding = policy.coding;
        model_sampling = policy.sampling;
        // ADR 0007: the GUI's resolver, adapter included (audit F03). Its n_ctx is
        // the request the session loads with (same_load pins it). The reply limit
        // is validated against it here, before any load (F01). A GGUF session may
        // open smaller (capped at its trained context, #570), so fit_prompt and the
        // reply limit are checked again below against the loaded context.
        ::xllama::SessionParams sp = ::xllama::CatalogueSessionParams(model);
        // options.num_ctx (checked before admission too) reduces the load. A
        // different n_ctx is a different load: same_load reloads the model.
        std::optional<int> context;
        if (const std::string error = read_context_override(root, sp.n_ctx, context);
            !error.empty()) {
            status = "400 Bad Request";
            return error_json(error, "invalid_value", "options.num_ctx");
        }
        if (context)
            sp.n_ctx = *context;
        policy_n_ctx = sp.n_ctx;
        const ::xllama::ApiTokenLimit limit =
            read_max_tokens(root, policy_n_ctx, native != nullptr);
        if (!limit.error.empty()) {
            status = "400 Bad Request";
            return error_json(limit.error);
        }
        n_predict = limit.value;
        const std::string sampling_error = check_sampling(root);
        if (!sampling_error.empty()) {
            status = "400 Bad Request";
            return error_json(sampling_error);
        }
        if (proposal && ::xllama::Session::backend_for(sp) != ::xllama::Backend::LlamaCpp) {
            status = "400 Bad Request";
            return error_json(
                "tools require a GGUF chat template. Select a tool-capable GGUF model or send "
                "this request without tools.",
                "tools_unsupported_for_model", "model");
        }
        // The native routes serve what /v1/models reports as ollama_chat.
        if (native &&
            !::xllama::ollama_chat_model(
                model, ::xllama::Session::backend_for(sp) == ::xllama::Backend::LlamaCpp, false)) {
            status = "400 Bad Request";
            return error_json("model '" + model +
                              "' is not served by /api/chat or /api/generate; use "
                              "/v1/chat/completions for ONNX models");
        }
        const auto before_load = ChatClock::now();
        const uint64_t loaded_generation = hub.generation;
        session = hub.ensure_locked(model, sp, &err, ::xllama::SessionOwner::Api);
        if (!session) {
            status =
                ::xllama::is_memory_refusal(err) ? "400 Bad Request" : "500 Internal Server Error";
            return error_json("session create failed: " + err);
        }
        if (hub.generation != loaded_generation)
            response.load_ms =
                std::chrono::duration<double, std::milli>(ChatClock::now() - before_load).count();
        // The request holds the session now, so its keep_alive is decided
        // against who owns it; an OpenAI request has none (no expiry).
        const auto action = ::xllama::keep_alive_action(hub.residency.owner,
                                                        native ? native->keep_alive : std::nullopt);
        if (action == ::xllama::KeepAliveAction::Refuse) {
            status = "409 Conflict";
            return error_json(kSessionInUseByConsole);
        }
        response.keep_alive = action;
        if (native && native->keep_alive)
            response.lease = *native->keep_alive;
        if (load_only) {
            status = "200 OK";
            return winrt::to_string(
                ollama_chat_json(response, model, JsonObject{}, L"load").Stringify());
        }
        // The context the session actually holds is what everything below budgets
        // against: a prompt between the cap and the request must be trimmed or
        // refused here, not fail in decode.
        if (const int loaded = session->context_length(); loaded > 0 && loaded < policy_n_ctx) {
            policy_n_ctx = loaded;
            const ::xllama::ApiTokenLimit loaded_limit =
                read_max_tokens(root, policy_n_ctx, native != nullptr);
            if (!loaded_limit.error.empty()) {
                status = "400 Bad Request";
                return error_json(loaded_limit.error);
            }
            n_predict = loaded_limit.value;
        }
    }

    if (proposal)
        return handle_tool_proposal_locked(root, request, session, model, thinking, model_is_coding,
                                           policy_n_ctx, n_predict, model_sampling, ticket, status,
                                           socket, stream, response);

    // Small instruct models degrade badly with an empty system turn (they
    // hallucinate the next role instead of answering); the chat UI always seeds
    // one. Match it when the client sends no system message — coding models get
    // the coding default so a bare LAN client does not look like general chat.
    if (system.empty())
        system = model_is_coding ? ::xllama::kCodingSystemPrompt : ::xllama::kDefaultSystemPrompt;

    const ::xllama::ChatFormat fmt = ::xllama::chat_format_for(model, thinking);
    ::xllama::GenerateParams gp;
    gp.stop_sequences = fmt.stop_sequences;
    gp.reuse_kv = false; // OpenAI chat is stateless: messages[] carry full history
    // Validated above, before the model load: the budget below needs to know how
    // much room the reply asks for.
    gp.n_predict = n_predict;

    // Same budget as the chat UI, same primitive, same tokenizer: the oldest
    // messages[] entries are dropped until the prompt plus the requested reply fit
    // n_ctx (xllama::fit_prompt). Before this, a long conversation reached
    // Session::generate and came back as a 500 — a client error reported as a
    // server one, and only after paying the tokenization.
    {
        const ::xllama::PromptFit fit = ::xllama::fit_prompt(
            fmt, system, history, final_user, policy_n_ctx, gp.n_predict,
            [session](const std::string& text) { return session->count_tokens(text); });
        if (!fit.fits) {
            // Nothing older left to drop: the final user message alone does not fit.
            // That is the client's input, hence 400.
            status = "400 Bad Request";
            return error_json(::xllama::prompt_too_long_message("final user message", fit.n_tokens,
                                                                gp.n_predict, policy_n_ctx));
        }
        if (fit.dropped > 0)
            ::xllama::log_output("[xllama] api: dropped " + std::to_string(fit.dropped) +
                                 " oldest message(s) to fit n_ctx " + std::to_string(policy_n_ctx) +
                                 "\n");
        gp.prompt = fit.prompt;
    }
    fill_sampling(root, model_sampling, gp);
    // Client-supplied stop: a string or an array of strings, added to the
    // format's own stops so clients can bound output.
    if (root.HasKey(L"stop")) {
        const auto sv = root.GetNamedValue(L"stop");
        if (sv.ValueType() == JsonValueType::String) {
            gp.stop_sequences.push_back(winrt::to_string(sv.GetString()));
        } else if (sv.ValueType() == JsonValueType::Array) {
            for (auto&& e : sv.GetArray())
                if (e.ValueType() == JsonValueType::String)
                    gp.stop_sequences.push_back(winrt::to_string(e.GetString()));
        }
    }

    // Every request check is done: from here on, a streamed reply owns the
    // socket.
    std::atomic<bool> abort{false};
    if (stream) {
        std::string failed_body;
        const std::unique_ptr<ChatStream> sse =
            open_chat_stream(socket, model, &abort, root, response, status, failed_body);
        if (!sse)
            return failed_body;
        gp.abort_flag = &abort;
        // The decode loop reports a piece before it checks the stop list, so the
        // stream withholds any suffix that could still become a stop.
        ::xllama::StopSequenceFilter stop_filter(gp.stop_sequences);
        // A thinking model's <think> block goes out as delta.reasoning_content,
        // split after the stop filter so a stop is matched on the raw text.
        std::optional<::xllama::ThinkingStreamSplitter> splitter;
        if (fmt.strip_thinking_content)
            splitter.emplace(fmt.reasoning_prefill());
        const auto send = [&sse, &splitter](std::string_view text, bool finishing) {
            if (!splitter) {
                sse->write_delta(text);
                return;
            }
            ::xllama::ThinkingSplit split = splitter->push(text);
            if (finishing) {
                const ::xllama::ThinkingSplit tail = splitter->finish();
                split.reasoning += tail.reasoning;
                split.content += tail.content;
            }
            sse->write_delta(split.reasoning, /*reasoning=*/true);
            sse->write_delta(split.content);
        };
        // #585, before the stop filter
        ::xllama::ReasoningTokenCounter reasoning_tokens(fmt.reasoning_prefill());
        gp.on_token = [&stop_filter, &send, &reasoning_tokens](std::string_view piece) {
            reasoning_tokens.push(piece);
            const std::string text = stop_filter.push(piece);
            if (!text.empty())
                send(text, false);
        };
        try {
            const ::xllama::InferenceResult r = session->generate(gp);
            // 200 is already on the wire: only the stream can report the outcome.
            // A client that disconnected aborts the loop; the partial result is
            // still a success, and the finish write is dropped by the dead socket.
            ticket.tokens(r.n_prompt, r.n_eval);
            if (!r.success) {
                status = "500 Internal Server Error";
                sse->write_error("generation failed: " + r.error_msg);
                return std::string();
            }
            // Generation can end on EOG or the token cap while a stop prefix is
            // still held; it never completed a stop, so it is ordinary text.
            send(stop_filter.finish(), /*finishing=*/true);
            const bool streamed_cap = r.hit_token_limit(gp.n_predict);
            sse->finish(streamed_cap ? L"length" : L"stop", r,
                        reasoning_tokens.count(fmt.strip_thinking_content));
            status = "200 OK";
        } catch (...) {
            // Generation or JSON serialization failed after the head was sent.
            // write_error is noexcept; the empty body keeps the outer handler
            // from writing a plain HTTP response on top of the stream.
            const ::xllama::ApiFailureReply failure = reply_for_current_exception(
                "chat stream", "malformed request body", "generation failed");
            status = failure.status;
            sse->write_error(failure.message);
        }
        return std::string(); // handle_connection must not write a second response
    }
    ::xllama::ReasoningTokenCounter reasoning_tokens(fmt.reasoning_prefill()); // #585
    gp.on_token = [&reasoning_tokens](std::string_view piece) { reasoning_tokens.push(piece); };
    const ::xllama::InferenceResult r = session->generate(gp);
    if (!r.success) {
        status = "500 Internal Server Error";
        return error_json("generation failed: " + r.error_msg);
    }
    ticket.tokens(r.n_prompt, r.n_eval);
    const ::xllama::ThinkingSplit reply = fmt.split_output(r.output_text);

    JsonObject message;
    message.Insert(L"role", JsonValue::CreateStringValue(L"assistant"));
    message.Insert(L"content", JsonValue::CreateStringValue(winrt::to_hstring(reply.content)));
    if (!reply.reasoning.empty())
        message.Insert(native ? L"thinking" : L"reasoning_content",
                       JsonValue::CreateStringValue(winrt::to_hstring(reply.reasoning)));
    const bool hit_cap = r.hit_token_limit(gp.n_predict);
    return chat_reply_json(response, model, message, hit_cap ? L"length" : L"stop", r,
                           reasoning_tokens.count(fmt.strip_thinking_content), status);
}

// The single model the server can currently serve: the resident one, else the
// LocalState\model.txt hint. Empty id if neither is set. hub.model is mutated
// under hub.mtx; read it only if we can take the lock, else fall back to the
// file hint (which touches no shared state) to avoid a data race.
std::string current_model_id() {
    auto& hub = ::xllama::session_hub();
    std::unique_lock<std::mutex> lk(hub.mtx, std::try_to_lock);
    if (lk.owns_lock() && !hub.model.empty())
        return hub.model;
    return ::xllama::read_local_text("model.txt");
}

// Catalogue ids ready to serve: every LocalState\models\<id> whose files make
// it loadable (model_dir_files_ready — base GGUF or ORT layout). The active /
// hinted model is appended if it is not already in the list (model.txt may
// name a raw path outside the catalogue). Sorted for stable output; the scan
// touches no shared state, so no hub lock is needed.
std::vector<std::string> servable_model_ids() {
    std::vector<std::string> ids;
    std::error_code ec;
    const std::filesystem::path root = ::xllama::resolve_local_path("models");
    for (const auto& entry : std::filesystem::directory_iterator(root, ec)) {
        if (!entry.is_directory(ec))
            continue;
        std::vector<std::string> files;
        for (const auto& f : std::filesystem::directory_iterator(entry.path(), ec))
            files.push_back(f.path().filename().string());
        if (::xllama::model_dir_files_ready(files))
            ids.push_back(entry.path().filename().string());
    }
    std::sort(ids.begin(), ids.end());
    const std::string current = current_model_id();
    if (!current.empty() && std::find(ids.begin(), ids.end(), current) == ids.end())
        ids.push_back(current);
    return ids;
}

// The package version (Major.Minor.Build.Revision; CI stamps the revision).
std::string package_version() {
    try {
        const auto v = winrt::Windows::ApplicationModel::Package::Current().Id().Version();
        return std::to_string(v.Major) + "." + std::to_string(v.Minor) + "." +
               std::to_string(v.Build) + "." + std::to_string(v.Revision);
    } catch (...) {
        return "unknown";
    }
}

// Whether |id|'s GGUF chat template renders tools, read from the file header
// and cached per file, size and write time: a pull or import that replaces
// the file is re-read, an unchanged one is not re-parsed on every listing.
bool gguf_model_renders_tools(const std::string& id) {
    const std::string path = ::xllama::first_gguf_in_dir(::xllama::resolve_model_path(id));
    if (path.empty())
        return false;
    std::error_code ec;
    const auto size = std::filesystem::file_size(path, ec);
    const auto written = std::filesystem::last_write_time(path, ec);
    if (ec)
        return false;
    const std::string key = path + "|" + std::to_string(size) + "|" +
                            std::to_string(written.time_since_epoch().count());
    static std::mutex mtx;
    static std::unordered_map<std::string, bool> cache;
    {
        std::lock_guard<std::mutex> lk(mtx);
        const auto hit = cache.find(key);
        if (hit != cache.end())
            return hit->second;
    }
    const bool renders = ::xllama::chat_template_renders_tools(::xllama::gguf_chat_template(path));
    std::lock_guard<std::mutex> lk(mtx);
    cache[key] = renders;
    return renders;
}

// OpenAI GET /v1/models: every servable on-device model, with xllama's
// non-standard fields (api_discovery.h; clients ignore unknown keys). Any
// listed id is valid for POST "model" — the server switches Sessions on
// demand. n_ctx is the resident session's when it is the active model, else
// the catalogue's.
std::string models_json() {
    const std::vector<::xllama::ManifestEntry> manifest = ::xllama::LoadModelManifest();
    std::string active;
    int active_n_ctx = 0;
    {
        auto& hub = ::xllama::session_hub();
        std::unique_lock<std::mutex> lk(hub.mtx, std::try_to_lock);
        if (lk.owns_lock() && hub.session) {
            active = hub.model;
            // The context the session holds, which can be below the requested
            // n_ctx (hub.params) when the backend bounds it at load.
            const int loaded = hub.session->context_length();
            active_n_ctx = loaded > 0 ? loaded : hub.params.n_ctx;
        }
    }
    if (active.empty())
        active = current_model_id();
    std::vector<::xllama::ModelInfo> models;
    for (const std::string& id : servable_model_ids()) {
        ::xllama::ModelInfo m;
        m.id = id;
        m.active = id == active;
        const ::xllama::ManifestEntry* entry =
            ::xllama::FindManifestEntry(manifest, ::xllama::utf8_to_wstring(id));
        if (entry) {
            m.display = ::xllama::wstring_to_utf8(entry->display);
            m.kind = ::xllama::wstring_to_utf8(entry->kind);
            m.role = ::xllama::wstring_to_utf8(entry->role);
            // The context a session of this entry opens with: the same rule as
            // catalogue_session_policy (ORT keeps the 8192 cap, #570).
            m.n_ctx = ::xllama::catalogue_n_ctx(entry->n_ctx, entry->kind == L"gguf");
        } else if (::xllama::model_uses_llama_backend(id)) {
            m.kind = "gguf";
        }
        if (m.active && active_n_ctx > 0)
            m.n_ctx = active_n_ctx;
        m.embedding = ::xllama::role_is_embedding(m.role);
        m.reasoning = ::xllama::model_is_thinking(id);
        m.reasoning_control = !m.embedding && ::xllama::model_supports_reasoning_control(id);
        m.tools = m.kind == "gguf" && !m.embedding && gguf_model_renders_tools(id);
        // What a chat request that sets no sampling field gets for this model.
        m.has_sampling = !m.embedding && m.kind != "diffusion";
        // What /api/chat and /api/generate accept (api_ollama.h).
        m.ollama_chat = ::xllama::ollama_chat_model(id, m.kind == "gguf", m.embedding);
        m.sampling =
            ::xllama::resolve_sampling(entry ? entry->sampling : ::xllama::SamplingOverrides{});
        models.push_back(std::move(m));
    }
    return ::xllama::models_list_json(models, static_cast<std::int64_t>(time(nullptr)));
}

// Ollama GET /api/ps under hub.mtx: zero or one resident model, its loaded
// context, who holds it ("console" or "api", session_residency.h) and, while an
// Ollama keep_alive lease runs, when it expires. A model the console app holds
// has no expiry.
std::string ps_json_locked() {
    auto& hub = ::xllama::session_hub();
    JsonArray models;
    if (hub.session) {
        JsonObject resident;
        const auto name = JsonValue::CreateStringValue(winrt::to_hstring(hub.model));
        resident.Insert(L"name", name);
        resident.Insert(L"model", name);
        const int loaded = hub.session->context_length();
        resident.Insert(L"context_length",
                        JsonValue::CreateNumberValue(loaded > 0 ? loaded : hub.params.n_ctx));
        resident.Insert(
            L"owner",
            JsonValue::CreateStringValue(
                hub.residency.owner == ::xllama::SessionOwner::Api ? L"api" : L"console"));
        if (hub.residency.deadline) {
            using WallClock = std::chrono::system_clock;
            const auto remaining =
                std::max(ChatClock::duration::zero(), *hub.residency.deadline - ChatClock::now());
            const auto expiry =
                WallClock::now() + std::chrono::duration_cast<WallClock::duration>(remaining);
            resident.Insert(L"expires_at", JsonValue::CreateStringValue(
                                               utc_timestamp(WallClock::to_time_t(expiry))));
        }
        models.Append(resident);
    }
    JsonObject root;
    root.Insert(L"models", models);
    return winrt::to_string(root.Stringify());
}

// Ollama GET /api/tags — {"models":[{name,model}]} for Ollama-probing clients.
std::string tags_json() {
    JsonArray models;
    for (const std::string& id : servable_model_ids()) {
        JsonObject m;
        m.Insert(L"name", JsonValue::CreateStringValue(winrt::to_hstring(id)));
        m.Insert(L"model", JsonValue::CreateStringValue(winrt::to_hstring(id)));
        models.Append(m);
    }
    JsonObject root;
    root.Insert(L"models", models);
    return winrt::to_string(root.Stringify());
}

enum class EmbeddingApi { Ollama, Legacy, OpenAI };

std::string embedding_error_json(EmbeddingApi api, const std::string& message) {
    // Native Ollama errors expose a string; OpenAI exposes a typed object.
    return api == EmbeddingApi::OpenAI ? error_json(message) : ollama_error_json(message);
}

std::string handle_embedding_locked(const std::string& body, const char*& status, EmbeddingApi api,
                                    ::xllama::ApiActivity::Ticket& ticket) {
    JsonObject root{nullptr};
    if (!JsonObject::TryParse(winrt::to_hstring(body), root) || root == nullptr) {
        status = "400 Bad Request";
        return embedding_error_json(api, "invalid JSON body");
    }
    std::string model = winrt::to_string(root.GetNamedString(L"model", L""));
    if (model.empty())
        model = ::xllama::read_local_text("model.txt");
    if (model.empty()) {
        status = "400 Bad Request";
        return embedding_error_json(api, "missing 'model' (and no LocalState\\model.txt fallback)");
    }

    model = ::xllama::resolve_pull_model_name(model);
    ticket.model(model);
    if (!model_servable_locked(model)) {
        status = "404 Not Found";
        return embedding_error_json(api, model_not_found(model));
    }

    std::vector<std::string> inputs;
    const wchar_t* input_key = api == EmbeddingApi::Legacy ? L"prompt" : L"input";
    if (!root.HasKey(input_key)) {
        status = "400 Bad Request";
        return embedding_error_json(api, api == EmbeddingApi::Legacy ? "missing 'prompt'"
                                                                     : "missing 'input'");
    }
    const IJsonValue input_value = root.GetNamedValue(input_key);
    if (input_value.ValueType() == JsonValueType::String) {
        inputs.push_back(winrt::to_string(input_value.GetString()));
    } else if (api != EmbeddingApi::Legacy && input_value.ValueType() == JsonValueType::Array) {
        const JsonArray arr = input_value.GetArray();
        for (uint32_t i = 0; i < arr.Size(); ++i) {
            const IJsonValue v = arr.GetAt(i);
            if (v.ValueType() != JsonValueType::String) {
                status = "400 Bad Request";
                return embedding_error_json(api, "every input array item must be a string");
            }
            inputs.push_back(winrt::to_string(v.GetString()));
        }
    } else {
        status = "400 Bad Request";
        return embedding_error_json(api, api == EmbeddingApi::Legacy
                                             ? "prompt must be a string"
                                             : "input must be a string or array of strings");
    }
    if (inputs.empty() || inputs.size() > 128) {
        status = "400 Bad Request";
        return embedding_error_json(api, inputs.empty() ? "input array must not be empty"
                                                        : "input batch is limited to 128 items");
    }
    for (const auto& input : inputs) {
        if (input.empty()) {
            status = "400 Bad Request";
            return embedding_error_json(api, "input items must not be empty");
        }
    }

    int dimensions = 0;
    if (root.HasKey(L"dimensions")) {
        const double raw = root.GetNamedNumber(L"dimensions");
        if (raw < 0 || raw > 32768 || raw != static_cast<int>(raw)) {
            status = "400 Bad Request";
            return embedding_error_json(api, "dimensions must be a non-negative integer");
        }
        dimensions = static_cast<int>(raw);
    }
    // BGE-M3 is not Matryoshka-trained: prefix truncation is not its
    // catalogue contract. Keep its native vector width on every adapter.
    if (model == "embed-bge-m3" && dimensions != 0 && dimensions != 1024) {
        status = "400 Bad Request";
        return embedding_error_json(api, "BGE-M3 dimensions must be 0 or 1024");
    }
    bool truncate = true;
    if (root.HasKey(L"truncate"))
        truncate = root.GetNamedBoolean(L"truncate");

    std::string encoding = "float";
    if (api == EmbeddingApi::OpenAI && root.HasKey(L"encoding_format")) {
        encoding = winrt::to_string(root.GetNamedString(L"encoding_format"));
        if (encoding != "float" && encoding != "base64") {
            status = "400 Bad Request";
            return embedding_error_json(api, "encoding_format must be 'float' or 'base64'");
        }
    }

    const CatalogueSessionPolicy policy = catalogue_session_policy(model);
    if (!policy.embedding || !policy.gguf) {
        status = "400 Bad Request";
        return embedding_error_json(api, "model is not a catalogue embedding GGUF");
    }
    const auto started = std::chrono::steady_clock::now();
    ::xllama::Session* session = nullptr;
    std::string err;
    ::xllama::SessionParams sp = ::xllama::CatalogueSessionParams(model); // ADR 0007
    std::optional<int> context;
    if (const std::string error = read_context_override(root, policy.n_ctx, context);
        !error.empty()) {
        status = "400 Bad Request";
        return embedding_error_json(api, error);
    }
    if (context)
        sp.n_ctx = *context;
    // A different num_ctx is a different load: ensure_locked reloads it.
    session = ::xllama::session_hub().ensure_locked(model, sp, &err, ::xllama::SessionOwner::Api);
    if (!session) {
        // A model or num_ctx that does not fit is the request's choice.
        status = ::xllama::is_memory_refusal(err) ? "400 Bad Request" : "500 Internal Server Error";
        return embedding_error_json(api, "session create failed: " + err);
    }
    const auto loaded = std::chrono::steady_clock::now();

    JsonArray embeddings;
    int prompt_tokens = 0;
    size_t embedding_width = 0;
    for (uint32_t i = 0; i < inputs.size(); ++i) {
        ::xllama::EmbeddingParams params;
        params.input = inputs[i];
        params.dimensions = dimensions;
        params.truncate = truncate;
        const ::xllama::EmbeddingResult result = session->embed(params);
        if (!result.success) {
            const bool client_error =
                result.error_msg.find("not supported") != std::string::npos ||
                result.error_msg.find("do not provide") != std::string::npos ||
                result.error_msg.find("input") != std::string::npos ||
                result.error_msg.find("dimension") != std::string::npos ||
                result.error_msg.find("ranking") != std::string::npos ||
                result.error_msg.find("encoder-only") != std::string::npos;
            status = client_error ? "400 Bad Request" : "500 Internal Server Error";
            return embedding_error_json(api, result.error_msg);
        }
        prompt_tokens += result.n_tokens;
        embedding_width = result.embedding.size();
        if (api == EmbeddingApi::OpenAI) {
            JsonObject item;
            item.Insert(L"object", JsonValue::CreateStringValue(L"embedding"));
            item.Insert(L"index", JsonValue::CreateNumberValue(i));
            if (encoding == "base64") {
                item.Insert(L"embedding", JsonValue::CreateStringValue(winrt::to_hstring(
                                              ::xllama::embedding_base64(result.embedding))));
            } else {
                JsonArray vector;
                for (float x : result.embedding)
                    vector.Append(JsonValue::CreateNumberValue(x));
                item.Insert(L"embedding", vector);
            }
            embeddings.Append(item);
        } else {
            JsonArray vector;
            for (float x : result.embedding)
                vector.Append(JsonValue::CreateNumberValue(x));
            embeddings.Append(vector);
        }
    }
    const auto finished = std::chrono::steady_clock::now();
    ticket.tokens(prompt_tokens, 0);
    ::xllama::log_output(
        ("[embedding] model=" + model + " inputs=" + std::to_string(inputs.size()) +
         " tokens=" + std::to_string(prompt_tokens) + " width=" + std::to_string(embedding_width) +
         " peak_ws_mb=" + std::to_string(::xllama::peak_working_set_mb()) + " total_ms=" +
         std::to_string(
             std::chrono::duration_cast<std::chrono::milliseconds>(finished - started).count()) +
         "\n")
            .c_str());
    const auto duration_ns = [](auto a, auto b) {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count();
    };
    const std::string json_model = api == EmbeddingApi::Legacy ? std::string{} : model;
    if (api == EmbeddingApi::Legacy) {
        JsonObject out;
        out.Insert(L"embedding", embeddings.GetAt(0).GetArray());
        status = "200 OK";
        return winrt::to_string(out.Stringify());
    }
    if (api == EmbeddingApi::OpenAI) {
        JsonObject usage;
        usage.Insert(L"prompt_tokens", JsonValue::CreateNumberValue(prompt_tokens));
        usage.Insert(L"total_tokens", JsonValue::CreateNumberValue(prompt_tokens));
        JsonObject out;
        out.Insert(L"object", JsonValue::CreateStringValue(L"list"));
        out.Insert(L"data", embeddings);
        out.Insert(L"model", JsonValue::CreateStringValue(winrt::to_hstring(json_model)));
        out.Insert(L"usage", usage);
        status = "200 OK";
        return winrt::to_string(out.Stringify());
    }
    JsonObject out;
    out.Insert(L"model", JsonValue::CreateStringValue(winrt::to_hstring(json_model)));
    out.Insert(L"embeddings", embeddings);
    out.Insert(L"total_duration", JsonValue::CreateNumberValue(duration_ns(started, finished)));
    out.Insert(L"load_duration", JsonValue::CreateNumberValue(duration_ns(started, loaded)));
    out.Insert(L"prompt_eval_count", JsonValue::CreateNumberValue(prompt_tokens));
    status = "200 OK";
    return winrt::to_string(out.Stringify());
}

// ---------------------------------------------------------------------------
// #118 — preferences / training status / images
// ---------------------------------------------------------------------------

std::string base64_encode(const std::string& in) {
    static const char kTbl[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve(((in.size() + 2) / 3) * 4);
    size_t i = 0;
    while (i + 2 < in.size()) {
        const unsigned n = (static_cast<unsigned char>(in[i]) << 16) |
                           (static_cast<unsigned char>(in[i + 1]) << 8) |
                           static_cast<unsigned char>(in[i + 2]);
        out.push_back(kTbl[(n >> 18) & 63]);
        out.push_back(kTbl[(n >> 12) & 63]);
        out.push_back(kTbl[(n >> 6) & 63]);
        out.push_back(kTbl[n & 63]);
        i += 3;
    }
    if (i < in.size()) {
        unsigned n = static_cast<unsigned char>(in[i]) << 16;
        out.push_back(kTbl[(n >> 18) & 63]);
        if (i + 1 < in.size()) {
            n |= static_cast<unsigned char>(in[i + 1]) << 8;
            out.push_back(kTbl[(n >> 12) & 63]);
            out.push_back(kTbl[(n >> 6) & 63]);
            out.push_back('=');
        } else {
            out.push_back(kTbl[(n >> 12) & 63]);
            out.push_back('=');
            out.push_back('=');
        }
    }
    return out;
}

std::string read_file_bytes(const std::string& path) {
    FILE* f = _wfopen(::xllama::utf8_to_wstring(path).c_str(), L"rb");
    if (!f)
        return {};
    std::string out;
    char buf[4096];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0)
        out.append(buf, n);
    fclose(f);
    return out;
}

// POST /v1/preferences — append one preference sample (same contract as UI rate).
std::string handle_preferences(const std::string& body, const char*& status) {
    JsonObject root{nullptr};
    if (!JsonObject::TryParse(winrt::to_hstring(body), root) || root == nullptr) {
        status = "400 Bad Request";
        return error_json("invalid JSON body");
    }
    const std::string label = winrt::to_string(root.GetNamedString(L"label", L""));
    if (!::xllama::preference_label_valid(label)) {
        status = "400 Bad Request";
        return error_json("invalid label (like|dislike|correction|implicit)");
    }
    if (!root.HasKey(L"messages") ||
        root.GetNamedValue(L"messages").ValueType() != JsonValueType::Array) {
        status = "400 Bad Request";
        return error_json("missing or non-array 'messages'");
    }
    std::vector<std::pair<std::string, std::string>> messages;
    const JsonArray arr = root.GetNamedArray(L"messages");
    for (uint32_t i = 0; i < arr.Size(); ++i) {
        const JsonObject m = arr.GetObjectAt(i);
        messages.emplace_back(winrt::to_string(m.GetNamedString(L"role", L"")),
                              winrt::to_string(m.GetNamedString(L"content", L"")));
    }
    const std::string preferred =
        winrt::to_string(root.GetNamedString(L"preferred_assistant", L""));
    const std::string line = ::xllama::format_preference_sample_jsonl(label, messages, preferred);
    if (line.empty()) {
        status = "400 Bad Request";
        return error_json("could not format preference sample");
    }
    // Ensure training/ exists.
    CreateDirectoryW(::xllama::utf8_to_wstring(::xllama::resolve_local_path("training")).c_str(),
                     nullptr);
    const std::string path = ::xllama::resolve_local_path(::xllama::kPreferenceSamplesRelPath);
    if (!::xllama::append_preference_sample_file(path, line)) {
        status = "500 Internal Server Error";
        return error_json("could not append training/samples.jsonl");
    }
    JsonObject ok;
    ok.Insert(L"ok", JsonValue::CreateBooleanValue(true));
    ok.Insert(L"label", JsonValue::CreateStringValue(winrt::to_hstring(label)));
    ok.Insert(L"path", JsonValue::CreateStringValue(L"training/samples.jsonl"));
    status = "200 OK";
    return winrt::to_string(ok.Stringify());
}

// GET /v1/training/status — result.done + progress.json + optional result.json.
std::string handle_training_status() {
    const std::string done_raw = ::xllama::read_local_text("training/result.done");
    const std::string done = ::xllama::parse_train_result_done(done_raw);
    const std::string progress = ::xllama::read_local_text("training/progress.json");
    const std::string result_path =
        ::xllama::resolve_local_path("training/out/personalized/result.json");
    const std::string result_body = read_file_bytes(result_path);

    std::string state = "idle";
    if (!progress.empty() && done.empty())
        state = "running";
    else if (done == "ok")
        state = "ok";
    else if (done == "fail")
        state = "fail";

    JsonObject root;
    root.Insert(L"state", JsonValue::CreateStringValue(winrt::to_hstring(state)));
    if (!done.empty())
        root.Insert(L"result_done", JsonValue::CreateStringValue(winrt::to_hstring(done)));
    if (!progress.empty()) {
        JsonObject prog{nullptr};
        if (JsonObject::TryParse(winrt::to_hstring(progress), prog) && prog)
            root.Insert(L"progress", prog);
        else
            root.Insert(L"progress_raw", JsonValue::CreateStringValue(winrt::to_hstring(progress)));
    }
    if (!result_body.empty()) {
        JsonObject res{nullptr};
        if (JsonObject::TryParse(winrt::to_hstring(result_body), res) && res)
            root.Insert(L"result", res);
    }
    const int samples = ::xllama::count_usable_preference_samples(
        ::xllama::resolve_local_path(::xllama::kPreferenceSamplesRelPath));
    root.Insert(L"usable_samples", JsonValue::CreateNumberValue(samples));
    return winrt::to_string(root.Stringify());
}

// POST /v1/images/generations — same guardrails as the Image dialog (steps 1–4).
std::string handle_images_locked(const std::string& body, const char*& status) {
    JsonObject root{nullptr};
    if (!JsonObject::TryParse(winrt::to_hstring(body), root) || root == nullptr) {
        status = "400 Bad Request";
        return error_json("invalid JSON body");
    }
    std::string prompt = winrt::to_string(root.GetNamedString(L"prompt", L""));
    if (prompt.empty()) {
        // OpenAI also accepts messages-like bodies; we only support prompt.
        status = "400 Bad Request";
        return error_json("missing 'prompt'");
    }
    // Clamped, never rejected, by the contract the Image dialog and the pipeline
    // share (xllama/diffusion/request.h): steps 1-4, any uint32 seed.
    namespace diff = ::xllama::diffusion;
    int steps = diff::kDefaultSteps;
    if (root.HasKey(L"steps"))
        steps = diff::interactive_steps(root.GetNamedNumber(L"steps", diff::kDefaultSteps));
    std::uint32_t seed = diff::kDefaultSeed;
    if (root.HasKey(L"seed"))
        seed = diff::request_seed(root.GetNamedNumber(L"seed", diff::kDefaultSeed));

    // Drive the same LocalState knobs as the Image dialog / autopilot.
    {
        // A half-written set would run the previous request's prompt or seed.
        if (!::xllama::write_local_text("prompt.txt", prompt) ||
            !::xllama::write_local_text("diffuse-steps.txt", std::to_string(steps)) ||
            !::xllama::write_local_text("diffuse-seed.txt", std::to_string(seed)) ||
            !::xllama::write_local_text("diffuse-model.txt", "sd-turbo-fp16")) {
            status = "500 Internal Server Error";
            return error_json("cannot write the image request to LocalState");
        }
        _wremove(
            ::xllama::utf8_to_wstring(::xllama::resolve_local_path("diffuse-cancel.flag")).c_str());
    }

    // The caller holds hub.mtx. Free the resident chat model first when the
    // pipeline would not fit next to it (#548).
    if (::xllama::diffusion::make_room_for_diffusion_locked(
            ::xllama::session_hub(), ::xllama::working_set_mb(), ::xllama::app_memory_limit_mb()))
        ::xllama::log_output(
            "[xllama] api: released the resident chat model to fit the image pipeline\n");
    const auto png_size = ::xllama::bridge::run_diffuse();

    const std::string stage = ::xllama::read_local_text("diffuse-progress.txt");
    if (!png_size) {
        status = "500 Internal Server Error";
        return error_json(std::string("image generation failed: stage=") + stage);
    }
    const std::string png_path = ::xllama::resolve_local_path("diffuse-out.png");
    const std::string png = read_file_bytes(png_path);
    if (png.size() != png_size) {
        status = "500 Internal Server Error";
        return error_json("diffuse-out.png incomplete after generation");
    }

    JsonObject item;
    item.Insert(L"b64_json", JsonValue::CreateStringValue(winrt::to_hstring(base64_encode(png))));
    item.Insert(L"path", JsonValue::CreateStringValue(L"diffuse-out.png"));
    JsonArray data;
    data.Append(item);
    JsonObject out;
    out.Insert(L"created", JsonValue::CreateNumberValue(static_cast<double>(time(nullptr))));
    out.Insert(L"data", data);
    status = "200 OK";
    return winrt::to_string(out.Stringify());
}

bool same_pull_manifest_entry(const ::xllama::ManifestEntry& a, const ::xllama::ManifestEntry& b) {
    if (a.name != b.name || a.kind != b.kind || a.role != b.role ||
        a.hf_base_url != b.hf_base_url || a.n_ctx != b.n_ctx || a.files.size() != b.files.size())
        return false;
    for (size_t i = 0; i < a.files.size(); ++i) {
        const auto& x = a.files[i];
        const auto& y = b.files[i];
        if (x.filename != y.filename || x.remote != y.remote || x.approx_bytes != y.approx_bytes ||
            x.sha256 != y.sha256 || x.role != y.role)
            return false;
    }
    return true;
}

// The /api/pull admission descriptor of a bundled catalogue entry
// (LoadModelManifest(&trust, false)): one home for handle_pull and the
// "pullable" field of GET /api/catalog. A bundled entry is part of the
// publisher-signed MSIX. Store additionally requires the catalogue's detached
// RSA signature. Device Portal overrides are not in that list and can never
// supply a pull URL or checksum.
::xllama::PullModelDescriptor pull_descriptor(const ::xllama::ManifestEntry& entry,
                                              const ::xllama::ManifestTrust& trust) {
    ::xllama::PullModelDescriptor descriptor;
    descriptor.name = ::xllama::wstring_to_utf8(entry.name);
    descriptor.kind = ::xllama::wstring_to_utf8(entry.kind);
    descriptor.role = ::xllama::wstring_to_utf8(entry.role);
    descriptor.source_url = ::xllama::wstring_to_utf8(entry.hf_base_url);
    #ifdef XLLAMA_STORE_SKU
    descriptor.trusted_catalogue = trust.trusted;
    #else
    (void)trust;
    descriptor.trusted_catalogue = true;
    #endif
    for (const auto& file : entry.files)
        descriptor.sha256_pins.push_back(::xllama::wstring_to_utf8(file.sha256));
    return descriptor;
}

// Chat and embedding inference use the merged runtime manifest. A same-name
// LocalState override that would change backend, role, URL, or pinned files
// shadows the bundled entry: /api/pull refuses it, so a pulled model is loaded
// with the settings it was pulled under, and /api/catalog reports it not pullable.
bool pull_entry_loads_as_bundled(const ::xllama::ManifestEntry& bundled,
                                 const std::vector<::xllama::ManifestEntry>& effective_manifest) {
    const auto* effective = ::xllama::FindManifestEntry(effective_manifest, bundled.name);
    return effective && same_pull_manifest_entry(bundled, *effective);
}

// GET /api/catalog (#658): the package's trusted catalogue, the list /api/pull
// downloads from, without withdrawn ids or diffusion assets
// (catalog_lists_entry). /v1/models stays "servable"; this is "downloadable".
// Fields come from the bundled entry with the rules /v1/models uses (n_ctx,
// sampling, reasoning, embedding); "installed" is membership in /v1/models and
// "pullable" is the pull admission above.
std::string catalog_list_json() {
    ::xllama::ManifestTrust trust;
    const auto manifest = ::xllama::LoadModelManifest(&trust, false);
    const auto effective_manifest = ::xllama::LoadModelManifest();
    const std::vector<std::string> servable = servable_model_ids();
    std::vector<::xllama::CatalogEntryInfo> entries;
    for (const ::xllama::ManifestEntry& entry : manifest) {
        ::xllama::CatalogEntryInfo e;
        e.id = ::xllama::wstring_to_utf8(entry.name);
        e.kind = ::xllama::wstring_to_utf8(entry.kind);
        if (!::xllama::catalog_lists_entry(e.id, e.kind))
            continue;
        e.display = ::xllama::wstring_to_utf8(entry.display);
        e.role = ::xllama::wstring_to_utf8(entry.role);
        e.n_ctx = ::xllama::catalogue_n_ctx(entry.n_ctx, entry.kind == L"gguf");
        for (const auto& file : entry.files)
            e.size_bytes += file.approx_bytes;
        e.installed = std::find(servable.begin(), servable.end(), e.id) != servable.end();
        e.pullable = ::xllama::api_pull_model_allowed(pull_descriptor(entry, trust)) &&
                     pull_entry_loads_as_bundled(entry, effective_manifest);
        e.embedding = ::xllama::role_is_embedding(e.role);
        e.reasoning = ::xllama::model_is_thinking(e.id);
        e.has_sampling = !e.embedding;
        e.sampling = ::xllama::resolve_sampling(entry.sampling);
        entries.push_back(std::move(e));
    }
    return ::xllama::catalog_json(entries);
}

void handle_pull(StreamSocket const& socket, const std::string& body, uint64_t generation) {
    JsonObject request{nullptr};
    if (!JsonObject::TryParse(winrt::to_hstring(body), request) || request == nullptr) {
        write_response(socket, "400 Bad Request", ollama_error_json("invalid JSON body"));
        return;
    }

    std::string requested_name;
    bool stream = true;
    try {
        if (request.HasKey(L"model")) {
            const auto value = request.GetNamedValue(L"model");
            if (value.ValueType() != JsonValueType::String) {
                write_response(socket, "400 Bad Request",
                               ollama_error_json("'model' must be a string"));
                return;
            }
            requested_name = winrt::to_string(value.GetString());
        }
        if (request.HasKey(L"stream")) {
            const auto value = request.GetNamedValue(L"stream");
            if (value.ValueType() != JsonValueType::Boolean) {
                write_response(socket, "400 Bad Request",
                               ollama_error_json("'stream' must be boolean"));
                return;
            }
            stream = value.GetBoolean();
        }
    } catch (...) {
        write_response(socket, "400 Bad Request", ollama_error_json("malformed pull request"));
        return;
    }
    if (requested_name.empty()) {
        write_response(socket, "400 Bad Request", ollama_error_json("missing 'model'"));
        return;
    }

    const std::string model_name = ::xllama::resolve_pull_model_name(requested_name);
    ::xllama::ManifestTrust catalogue_trust;
    const auto manifest = ::xllama::LoadModelManifest(&catalogue_trust, false);
    const auto* entry =
        ::xllama::FindManifestEntry(manifest, ::xllama::utf8_to_wstring(model_name));
    if (!entry) {
        write_response(socket, "404 Not Found",
                       ollama_error_json("model is not in the bundled catalogue"));
        return;
    }

    if (!::xllama::api_pull_model_allowed(pull_descriptor(*entry, catalogue_trust))) {
        write_response(
            socket, "403 Forbidden",
            ollama_error_json("model is not a trusted, pinned chat or embedding download"));
        return;
    }

    if (!pull_entry_loads_as_bundled(*entry, ::xllama::LoadModelManifest())) {
        write_response(socket, "403 Forbidden",
                       ollama_error_json("model is shadowed by a LocalState catalogue override"));
        return;
    }

    auto pull_guard = g_pull_gate.try_acquire();
    if (!pull_guard.owns_lock()) {
        write_response(socket, "409 Conflict",
                       ollama_error_json("another model pull is in progress"));
        return;
    }

    {
        std::lock_guard<std::mutex> state_lock(g_state_mtx);
        if (g_status.state != ServerState::Running || generation != g_generation) {
            write_response(socket, "503 Service Unavailable", ollama_error_json("server stopped"));
            return;
        }
    }

    std::unique_ptr<PullJsonStream> output;
    if (stream) {
        output = std::make_unique<PullJsonStream>(socket);
        (void)output->write_line(pull_event_json("pulling manifest"));
        (void)output->write_line(pull_event_json("pulling model"));
    }

    const std::string model_dir = ::xllama::resolve_local_path("models/" + model_name);
    bool download_ok = false;
    std::wstring download_error;
    {
        const std::wstring local_dir = ::xllama::utf8_to_wstring(model_dir);
        try {
            auto* output_stream = output.get();
            ::xllama::ModelDownloader::DownloadAsync(
                entry->hf_base_url, local_dir, entry->files,
                winrt::Windows::UI::Core::CoreDispatcher{nullptr},
                [output_stream](uint64_t done, uint64_t total) {
                    if (output_stream)
                        (void)output_stream->write_line(
                            pull_event_json("pulling model", done, total));
                },
                [&download_ok, &download_error](bool ok, std::wstring error) {
                    download_ok = ok;
                    download_error = std::move(error);
                })
                .get();
        } catch (winrt::hresult_error const& error) {
            download_ok = false;
            download_error = L"download failed (0x" +
                             std::to_wstring(static_cast<uint32_t>(error.code().value)) + L")";
        } catch (...) {
            download_ok = false;
            download_error = L"download failed";
        }
    }
    if (!download_ok) {
        const std::string message = ::xllama::wstring_to_utf8(download_error);
        if (output) {
            (void)output->write_line(ollama_error_json(message));
            output->finish();
        } else {
            write_response(socket,
                           download_error == ::xllama::ModelDownloader::kBusyError
                               ? "409 Conflict"
                               : "500 Internal Server Error",
                           ollama_error_json(message));
        }
        return;
    }

    if (output)
        (void)output->write_line(pull_event_json("loading model"));

    auto compute_permit = acquire_compute_job_or_busy();
    if (!compute_permit) {
        const std::string message = "busy";
        if (output) {
            (void)output->write_line(ollama_error_json(message));
            output->finish();
        } else {
            write_response(socket, "503 Service Unavailable", ollama_error_json(message));
        }
        return;
    }

    bool server_active = false;
    {
        std::lock_guard<std::mutex> state_lock(g_state_mtx);
        server_active = g_status.state == ServerState::Running && generation == g_generation;
    }
    if (!server_active) {
        const std::string message = "server stopped before model load";
        if (output) {
            (void)output->write_line(ollama_error_json(message));
            output->finish();
        } else {
            write_response(socket, "503 Service Unavailable", ollama_error_json(message));
        }
        return;
    }

    std::string load_error;
    ::xllama::Session* loaded = nullptr;
    {
        auto& hub = ::xllama::session_hub();
        std::lock_guard<std::mutex> hub_lock(hub.mtx); // let active inference finish, then swap
        const ::xllama::SessionParams params = ::xllama::CatalogueSessionParams(model_name);
        // The files may have just changed under a resident session of this
        // model: same id and parameters, stale weights (ADR 0007).
        hub.invalidate_locked(model_name);
        // Recheck after waiting for inference: stop/rebind may have invalidated
        // the listener while this pull was blocked on the resident session.
        {
            std::lock_guard<std::mutex> state_lock(g_state_mtx);
            server_active = g_status.state == ServerState::Running && generation == g_generation;
        }
        if (server_active)
            loaded =
                hub.ensure_locked(model_name, params, &load_error, ::xllama::SessionOwner::Api);
        else
            load_error = "server stopped before model load";
    }
    if (!loaded) {
        const std::string message = "model downloaded but could not load: " + load_error;
        if (output) {
            (void)output->write_line(ollama_error_json(message));
            output->finish();
        } else {
            write_response(socket,
                           server_active ? "500 Internal Server Error" : "503 Service Unavailable",
                           ollama_error_json(message));
        }
        return;
    }

    if (output) {
        (void)output->write_line(pull_event_json("success"));
        output->finish();
    } else {
        write_response(socket, "200 OK", pull_event_json("success"));
    }
}

// "200 OK" -> 200 for the activity record.
int http_code(const char* status) {
    return std::atoi(status);
}

// The human message of an error body: OpenAI {"error":{"message":...}} or
// Ollama {"error":"..."}. Empty when the body carries none.
std::string error_text(const std::string& json) {
    JsonObject root{nullptr};
    if (!JsonObject::TryParse(winrt::to_hstring(json), root) || root == nullptr ||
        !root.HasKey(L"error"))
        return {};
    const auto value = root.GetNamedValue(L"error");
    if (value.ValueType() == JsonValueType::String)
        return winrt::to_string(value.GetString());
    if (value.ValueType() == JsonValueType::Object)
        return winrt::to_string(value.GetObject().GetNamedString(L"message", L""));
    return {};
}

// Close the activity record of a request once its response is written.
void record_outcome(::xllama::ApiActivity::Ticket& ticket, const char* status,
                    const std::string& json) {
    const int code = http_code(status);
    ticket.done(code, code >= 400 ? error_text(json) : std::string());
}

// Status line for a framing refusal (read_request / parse_http_head).
const char* framing_status(int code) {
    switch (code) {
    case 408:
        return "408 Request Timeout";
    case 411:
        return "411 Length Required";
    case 413:
        return "413 Payload Too Large";
    case 431:
        return "431 Request Header Fields Too Large";
    default:
        return "400 Bad Request";
    }
}

// Serve one accepted socket. Runs on a thread-pool work item, never inline
// in ConnectionReceived: the listener does not raise the next connection
// until that callback returns, so an in-line SSE reply hid the permit from
// every subsequent LAN request (console 1.6.0.1284). The compute permit is
// a stack RAII handle; a client drop mid-SSE aborts generate() and this
// frame unwinds, which releases the slot.
void handle_connection(StreamSocket const& socket, uint64_t generation) {
    try {
        const HttpRequest req = read_request(socket);
        if (req.status != 0) {
            write_response(socket, framing_status(req.status), error_json(req.error));
            if (req.pending)
                req.pending.Cancel(); // after the 408 is out, see HttpRequest::pending
            return;
        }

        // CORS preflight for browser clients (Continue.dev webview, web UIs).
        if (req.method == "OPTIONS") {
            write_cors_preflight(socket);
            return;
        }

        // Health probe (also the spike gate: curl http://<ip>:<port>/ -> 200).
        if (req.method == "GET" && (req.path == "/" || req.path == "/health")) {
            write_response(socket, "200 OK",
                           ::xllama::health_json(package_version(),
                                                 ::xllama::build_api_capabilities(),
                                                 ::xllama::build_llama_cpp()));
            return;
        }

        // Model discovery: clients probe /v1/models (OpenAI SDK) and /api/tags
        // (Ollama-aware) before completing.
        if (req.method == "GET" && req.path == "/v1/models") {
            write_response(socket, "200 OK", models_json());
            return;
        }
        if (req.method == "GET" && req.path == "/api/tags") {
            write_response(socket, "200 OK", tags_json());
            return;
        }
        // What a client may offer to download (#658); not a servable list.
        if (req.method == "GET" && req.path == "/api/catalog") {
            write_response(socket, "200 OK", catalog_list_json());
            return;
        }

        // Reads the hub only: busy while a turn holds it, as Ollama clients
        // poll this between requests.
        if (req.method == "GET" && req.path == "/api/ps") {
            std::unique_lock<std::mutex> lk = acquire_hub_or_busy();
            if (!lk.owns_lock()) {
                write_response(socket, ::xllama::kApiBusyStatus, ollama_error_json("busy"));
                return;
            }
            const std::string json = ps_json_locked();
            lk.unlock();
            write_response(socket, "200 OK", json);
            return;
        }

        if (req.method == "POST" && req.path == "/api/pull") {
            handle_pull(socket, req.body, generation);
            return;
        }

        // OpenAI chat and the native Ollama text routes share one handler: the
        // native body is checked and translated into the chat body before
        // compute admission (api_ollama.h), and only the reply wire differs.
        const bool native_chat = req.path == "/api/chat";
        const bool native_generate = req.path == "/api/generate";
        if (req.method == "POST" &&
            (req.path == "/v1/chat/completions" || native_chat || native_generate)) {
            ChatResponse response;
            std::optional<::xllama::OllamaRequest> native;
            const auto failure_json = [&](const std::string& message) {
                return native ? ollama_error_json(message) : error_json(message);
            };
            if (native_chat || native_generate) {
                response.wire = native_chat ? ChatWire::OllamaChat : ChatWire::OllamaGenerate;
                native = ::xllama::parse_ollama_request(
                    req.body,
                    native_chat ? ::xllama::OllamaRoute::Chat : ::xllama::OllamaRoute::Generate);
                if (!native->error.empty()) {
                    write_response(socket, "400 Bad Request", failure_json(native->error));
                    return;
                }
            }
            const std::string& body = native ? native->body : req.body;
            JsonObject root{nullptr};
            if (!JsonObject::TryParse(winrt::to_hstring(body), root) || root == nullptr) {
                write_response(socket, "400 Bad Request", failure_json("invalid JSON body"));
                return;
            }
            if (root.HasKey(L"stream") &&
                root.GetNamedValue(L"stream").ValueType() != JsonValueType::Boolean) {
                write_response(socket, "400 Bad Request", error_json("'stream' must be boolean"));
                return;
            }
            if (!accepts_response_format(root)) {
                write_response(socket, "400 Bad Request",
                               error_json("response_format must be an object with only type 'text'",
                                          "unsupported_response_format", "response_format"));
                return;
            }
            if (const std::string error = check_context_override(root); !error.empty()) {
                write_response(socket, "400 Bad Request",
                               native ? ollama_error_json(error)
                                      : error_json(error, "invalid_value", "options.num_ctx"));
                return;
            }
            // reasoning_effort, chat_template_kwargs.enable_thinking (and Ollama
            // `think`, mapped onto them): a switch the model cannot honour is
            // the client's error, so it is refused here, not after a load.
            const ::xllama::ReasoningDecision reasoning = ::xllama::decide_reasoning(
                request_model(root), ::xllama::read_reasoning_fields(body));
            if (!reasoning.error.empty()) {
                write_response(socket, "400 Bad Request",
                               native
                                   ? ollama_error_json(reasoning.error)
                                   : error_json(reasoning.error, reasoning.code, reasoning.param));
                return;
            }
            const bool stream = native ? native->stream : root.GetNamedBoolean(L"stream", false);
            // Like response_format: a malformed option is the client's error
            // (400), not a reason to wait behind inference or report busy.
            if (!stream_usage(root, stream)) {
                const std::string message = "stream_options requires stream:true and an object "
                                            "whose include_usage, if present, is a boolean";
                write_response(socket, "400 Bad Request",
                               native ? ollama_error_json(message)
                                      : error_json(message, "invalid_value", "stream_options"));
                return;
            }
            auto compute_permit = acquire_compute_job_or_busy();
            if (!compute_permit) {
                activity().busy();
                write_response(socket, ::xllama::kApiBusyStatus, failure_json("busy"));
                return;
            }
            std::unique_lock<std::mutex> lk = acquire_hub_or_busy();
            if (!lk.owns_lock()) {
                activity().busy();
                write_response(socket, "503 Service Unavailable", failure_json("busy"));
                return;
            }
            // stop/rebind may have happened after this callback was queued. Do
            // not let an old listener recreate the lazily-owned Session.
            bool active_generation = false;
            {
                std::lock_guard<std::mutex> state_lock(g_state_mtx);
                active_generation =
                    g_status.state == ServerState::Running && generation == g_generation;
            }
            if (!active_generation) {
                write_response(socket, "503 Service Unavailable", failure_json("server stopped"));
                return;
            }
            // handle_chat_locked touches WinRT JSON accessors that throw on
            // wrong-typed fields; guarantee the client always gets a response
            // rather than a silently dropped socket.
            auto ticket = activity().begin(native_generate ? "generate" : "chat");
            const char* status = "200 OK";
            std::string json;
            {
                const KeepAliveAtEnd keep_alive{response};
                try {
                    json = handle_chat_locked(root, body, status, ticket, socket, stream, response,
                                              native ? &*native : nullptr, reasoning.thinking);
                    // Native Ollama errors are {"error":"<text>"}.
                    if (native && status[0] != '2' && !json.empty())
                        json = ollama_error_json(error_text(json));
                } catch (...) {
                    const ::xllama::ApiFailureReply failure = reply_for_current_exception(
                        "chat", "malformed request body", "internal error");
                    status = failure.status;
                    json = failure_json(failure.message);
                }
            }
            // A new admission can replace the activity ticket. Finish accounting
            // before releasing ownership, then deliver the request-owned body.
            record_outcome(ticket, status, json);
            lk.unlock();
            compute_permit.reset();
            // A streamed reply is already on the socket; an empty body means the
            // stream (or its error event) was the response.
            if (stream && json.empty())
                return;
            write_response(socket, status, json);
            return;
        }

        EmbeddingApi embedding_api;
        const bool is_embedding_route =
            req.method == "POST" && (req.path == "/api/embed" || req.path == "/api/embeddings" ||
                                     req.path == "/v1/embeddings");
        if (is_embedding_route) {
            embedding_api = req.path == "/api/embed"        ? EmbeddingApi::Ollama
                            : req.path == "/api/embeddings" ? EmbeddingApi::Legacy
                                                            : EmbeddingApi::OpenAI;
            auto compute_permit = acquire_compute_job_or_busy();
            if (!compute_permit) {
                activity().busy();
                write_response(socket, ::xllama::kApiBusyStatus,
                               embedding_error_json(embedding_api, "busy"));
                return;
            }
            std::unique_lock<std::mutex> lk = acquire_hub_or_busy();
            if (!lk.owns_lock()) {
                activity().busy();
                write_response(socket, "503 Service Unavailable",
                               embedding_error_json(embedding_api, "busy"));
                return;
            }
            bool active = false;
            {
                std::lock_guard<std::mutex> state_lock(g_state_mtx);
                active = g_status.state == ServerState::Running && generation == g_generation;
            }
            if (!active) {
                write_response(socket, "503 Service Unavailable",
                               embedding_error_json(embedding_api, "server stopped"));
                return;
            }
            auto ticket = activity().begin("embeddings");
            const char* status = "200 OK";
            std::string json;
            try {
                json = handle_embedding_locked(req.body, status, embedding_api, ticket);
            } catch (...) {
                const ::xllama::ApiFailureReply failure = reply_for_current_exception(
                    "embeddings", "malformed embedding request", "internal error");
                status = failure.status;
                json = embedding_error_json(embedding_api, failure.message);
            }
            // Record before writing: a client that disconnects makes
            // write_response throw, and the request was still served.
            record_outcome(ticket, status, json);
            write_response(socket, status, json);
            return;
        }

        // #118: preference capture (no inference lock needed).
        if (req.method == "POST" && req.path == "/v1/preferences") {
            const char* status = "200 OK";
            std::string json;
            try {
                json = handle_preferences(req.body, status);
            } catch (...) {
                const ::xllama::ApiFailureReply failure = reply_for_current_exception(
                    "preferences", "malformed request body", "internal error");
                status = failure.status;
                json = error_json(failure.message);
            }
            write_response(socket, status, json);
            return;
        }

        // #118: training status (read-only files).
        if (req.method == "GET" && req.path == "/v1/training/status") {
            write_response(socket, "200 OK", handle_training_status());
            return;
        }

        // #118: image generation — shares compute admission and the hub slot.
        if (req.method == "POST" && req.path == "/v1/images/generations") {
            auto compute_permit = acquire_compute_job_or_busy();
            if (!compute_permit) {
                activity().busy();
                write_response(socket, ::xllama::kApiBusyStatus, error_json("busy"));
                return;
            }
            std::unique_lock<std::mutex> lk = acquire_hub_or_busy();
            if (!lk.owns_lock()) {
                activity().busy();
                write_response(socket, ::xllama::kApiBusyStatus, error_json("busy"));
                return;
            }
            bool active = false;
            {
                std::lock_guard<std::mutex> state_lock(g_state_mtx);
                active = g_status.state == ServerState::Running && generation == g_generation;
            }
            if (!active) {
                write_response(socket, "503 Service Unavailable", error_json("server stopped"));
                return;
            }
            auto ticket = activity().begin("images");
            const char* status = "200 OK";
            std::string json;
            try {
                json = handle_images_locked(req.body, status);
            } catch (...) {
                const ::xllama::ApiFailureReply failure = reply_for_current_exception(
                    "images", "malformed request body", "image generation failed");
                status = failure.status;
                json = error_json(failure.message);
            }
            // Record before writing: a client that disconnects makes
            // write_response throw, and the request was still served.
            record_outcome(ticket, status, json);
            write_response(socket, status, json);
            return;
        }

        write_response(socket, "404 Not Found", error_json("no such endpoint"));
    } catch (winrt::hresult_error const& e) {
        char buf[256];
        snprintf(buf, sizeof(buf), "[xllama] api: connection hresult 0x%08X\n",
                 static_cast<unsigned>(e.code().value));
        ::xllama::log_output(buf);
    } catch (const std::exception& e) {
        ::xllama::log_output(std::string("[xllama] api: connection error: ") + e.what() + "\n");
    } catch (...) {
        ::xllama::log_output("[xllama] api: connection unknown error\n");
    }
}

// Return from ConnectionReceived immediately after queueing. A write on this
// thread is only the bounded-gate 503; every other reply runs on the work item.
void dispatch_connection(StreamSocket const& socket, uint64_t generation) {
    auto slot = ::xllama::process_api_connection_gate().try_acquire();
    if (!slot) {
        try {
            activity().busy();
            write_response(socket, ::xllama::kApiBusyStatus, error_json("busy"));
        } catch (...) {
        }
        return;
    }
    auto held = std::make_shared<::xllama::ApiConnectionGate::Handle>(std::move(slot));
    try {
        winrt::Windows::System::Threading::ThreadPool::RunAsync(
            [socket, generation, held](winrt::Windows::Foundation::IAsyncAction const&) {
                auto client = activity().client();
                handle_connection(socket, generation);
                (void)held;
                (void)client;
            });
    } catch (...) {
        try {
            activity().busy();
            write_response(socket, ::xllama::kApiBusyStatus, error_json("busy"));
        } catch (...) {
        }
    }
}

void stop_server_locked(bool write_log) {
    StreamSocketListener listener{nullptr};
    {
        std::lock_guard<std::mutex> lock(g_state_mtx);
        listener = g_listener;
        g_listener = nullptr;
        ++g_generation;
        g_status = {ServerState::Stopped, 0, {}};
    }
    if (listener) {
        try {
            listener.Close();
        } catch (...) {
        }
    }
    // Deliberately NOT resetting the hub here: the Session is process-shared
    // now, and the GUI may be the one using the resident model. A model loaded
    // solely via the API stays resident, and a keep_alive lease on it still
    // runs; the GUI takes it over or swaps it on next use.
    if (write_log)
        ::xllama::log_output("[xllama] api: stopped\n");
}

} // namespace

ServerStatus server_status() {
    std::lock_guard<std::mutex> lock(g_state_mtx);
    return g_status;
}

void stop_server() {
    std::lock_guard<std::mutex> control_lock(g_control_mtx);
    stop_server_locked(true);
}

void start_server(int requested_port) {
    std::lock_guard<std::mutex> control_lock(g_control_mtx);
    const int port = requested_port == 0 ? listen_port() : requested_port;
    if (!port_bindable(port)) {
        stop_server_locked(false);
        std::lock_guard<std::mutex> lock(g_state_mtx);
        g_status = {ServerState::Error, port, "port outside bindable range"};
        return;
    }

    const ServerStatus current = server_status();
    if (current.state == ServerState::Running && current.port == port)
        return;
    if (current.state == ServerState::Running || current.state == ServerState::Starting)
        stop_server_locked(false);

    uint64_t generation = 0;
    {
        std::lock_guard<std::mutex> lock(g_state_mtx);
        generation = ++g_generation;
        g_status = {ServerState::Starting, port, {}};
    }
    try {
        StreamSocketListener listener;
        listener.ConnectionReceived(
            [generation](StreamSocketListener const&,
                         StreamSocketListenerConnectionReceivedEventArgs const& a) {
                dispatch_connection(a.Socket(), generation);
            });
        listener.BindServiceNameAsync(winrt::to_hstring(std::to_string(port))).get();
        {
            std::lock_guard<std::mutex> lock(g_state_mtx);
            g_listener = listener;
            g_status = {ServerState::Running, port, {}};
        }
        ::xllama::log_output("[xllama] api: listening on 0.0.0.0:" + std::to_string(port) +
                             " (OpenAI-compat /v1/chat/completions)\n");
    } catch (winrt::hresult_error const& e) {
        char buf[256];
        snprintf(buf, sizeof(buf), "[xllama] api: listener bind failed 0x%08X\n",
                 static_cast<unsigned>(e.code().value));
        {
            std::lock_guard<std::mutex> lock(g_state_mtx);
            g_listener = nullptr;
            g_status = {ServerState::Error, port, buf};
        }
        ::xllama::log_output(buf);
    } catch (const std::exception& e) {
        {
            std::lock_guard<std::mutex> lock(g_state_mtx);
            g_listener = nullptr;
            g_status = {ServerState::Error, port, e.what()};
        }
        ::xllama::log_output(std::string("[xllama] api: fatal: ") + e.what() + "\n");
    }
}

void run_server() {
    start_server();
}

::xllama::ApiActivity& activity() {
    static ::xllama::ApiActivity instance;
    return instance;
}

} // namespace xllama::api

#else // !XLLAMA_UWP

    #include "xllama/api_activity.h"

namespace xllama::api {
::xllama::ApiActivity& activity() {
    static ::xllama::ApiActivity instance;
    return instance;
}
void start_server(int) {}
void stop_server() {}
ServerStatus server_status() {
    return {};
}
void run_server() {}
} // namespace xllama::api

#endif // XLLAMA_UWP
