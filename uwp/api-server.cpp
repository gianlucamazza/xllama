// Copyright (c) 2024 Venere Labs
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
#include <winrt/Windows.Data.Json.h>
#include <winrt/Windows.Networking.Sockets.h>
// clang-format on

    #include "xllama/chat_prompt.h"
    #include "xllama/path_utils.h"
    #include "xllama/platform.h"
    #include "xllama/session.h"

    #include <cctype>
    #include <cstdio>
    #include <ctime>
    #include <future>
    #include <memory>
    #include <mutex>
    #include <string>
    #include <vector>

namespace xllama::api {
namespace {

using namespace winrt::Windows::Networking::Sockets;
using namespace winrt::Windows::Storage::Streams;
using namespace winrt::Windows::Data::Json;

// ---------------------------------------------------------------------------
// Shared single-slot inference state
//
// The server owns ONE Session, created lazily on the first request and reused.
// g_mtx serializes both (re)creation and generate(): xllama::Session::generate()
// is not concurrent (session.h:74). A request that finds the slot busy gets 503.
// ---------------------------------------------------------------------------
std::mutex g_mtx;
std::unique_ptr<::xllama::Session> g_session;
std::string g_model; // model id currently loaded into g_session

// Keep the listener alive for the process lifetime (callbacks fire on the WinRT
// thread pool, not on run_server's thread).
StreamSocketListener g_listener{nullptr};

constexpr int kDefaultPort = 11434; // Ollama default port, familiar to clients

// Read and trim a small LocalState text file; empty string if absent.
std::string read_local_text(const char* name) {
    std::string out;
    const std::string p = ::xllama::resolve_local_path(name);
    FILE* f = _wfopen(winrt::to_hstring(p).c_str(), L"r");
    if (!f)
        return out;
    char buf[512];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0)
        out.append(buf, n);
    fclose(f);
    while (!out.empty() && (out.back() == '\n' || out.back() == '\r' || out.back() == ' '))
        out.pop_back();
    return out;
}

// Xbox blocks binding ports in [57344, 65535] (traffic is silently dropped, per
// the UWP-on-Xbox known-issues doc); 11443 is the Device Portal. Reject those so
// an api-port.txt typo fails loudly to the default rather than binding a dead
// port. Valid app range is [1025, 49151].
bool port_bindable(int p) {
    return p >= 1025 && p <= 49151 && p != 11443;
}

int listen_port() {
    const std::string s = read_local_text("api-port.txt");
    if (!s.empty()) {
        const int p = std::atoi(s.c_str());
        if (port_bindable(p))
            return p;
        ::xllama::log_output("[xllama] api: api-port.txt=" + s +
                             " out of bindable range [1025,49151]\\11443; using default\n");
    }
    return kDefaultPort;
}

// ---------------------------------------------------------------------------
// Minimal HTTP request read
// ---------------------------------------------------------------------------
struct HttpRequest {
    std::string method;
    std::string path;
    std::string body;
    bool ok = false;
};

std::string to_lower(std::string s) {
    for (char& c : s)
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

// Reads request line + headers + Content-Length body from the socket. Blocking
// (LoadAsync().get()); called on a thread-pool thread, never the UI thread.
HttpRequest read_request(StreamSocket const& socket) {
    HttpRequest req;
    DataReader reader(socket.InputStream());
    reader.InputStreamOptions(InputStreamOptions::Partial);

    std::string data;
    size_t header_end = std::string::npos;
    // Pull chunks until the header terminator appears (cap to avoid abuse).
    while (header_end == std::string::npos && data.size() < 64 * 1024) {
        const uint32_t got = reader.LoadAsync(4096).get();
        if (got == 0)
            break;
        for (uint32_t i = 0; i < got; ++i)
            data.push_back(static_cast<char>(reader.ReadByte()));
        header_end = data.find("\r\n\r\n");
    }
    if (header_end == std::string::npos)
        return req;

    // Request line: METHOD SP PATH SP HTTP/x.y
    const size_t line_end = data.find("\r\n");
    const std::string line = data.substr(0, line_end);
    const size_t sp1 = line.find(' ');
    const size_t sp2 = (sp1 == std::string::npos) ? std::string::npos : line.find(' ', sp1 + 1);
    if (sp1 == std::string::npos || sp2 == std::string::npos)
        return req;
    req.method = line.substr(0, sp1);
    req.path = line.substr(sp1 + 1, sp2 - sp1 - 1);

    // Content-Length (case-insensitive header scan).
    size_t content_length = 0;
    {
        const std::string headers = to_lower(data.substr(line_end + 2, header_end - line_end - 2));
        const size_t cl = headers.find("content-length:");
        if (cl != std::string::npos) {
            content_length = static_cast<size_t>(std::atoll(headers.c_str() + cl + 15));
            if (content_length > 8 * 1024 * 1024)
                content_length = 8 * 1024 * 1024; // hard cap
        }
    }

    const size_t body_start = header_end + 4;
    while (data.size() - body_start < content_length) {
        const uint32_t got = reader.LoadAsync(4096).get();
        if (got == 0)
            break;
        for (uint32_t i = 0; i < got; ++i)
            data.push_back(static_cast<char>(reader.ReadByte()));
    }
    req.body = data.substr(body_start, content_length);
    req.ok = true;
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

// CORS preflight: browsers OPTIONS non-simple requests before the real POST.
void write_cors_preflight(StreamSocket const& socket) {
    write_raw(socket, "HTTP/1.1 204 No Content\r\n"
                      "Access-Control-Allow-Origin: *\r\n"
                      "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
                      "Access-Control-Allow-Headers: Authorization, Content-Type\r\n"
                      "Access-Control-Allow-Private-Network: true\r\n"
                      "Access-Control-Max-Age: 86400\r\n"
                      "Content-Length: 0\r\nConnection: close\r\n\r\n");
}

std::string error_json(const std::string& msg) {
    JsonObject err;
    err.Insert(L"message", JsonValue::CreateStringValue(winrt::to_hstring(msg)));
    err.Insert(L"type", JsonValue::CreateStringValue(L"xllama_error"));
    JsonObject root;
    root.Insert(L"error", err);
    return winrt::to_string(root.Stringify());
}

// ---------------------------------------------------------------------------
// OpenAI /v1/chat/completions
// ---------------------------------------------------------------------------

// Turn the OpenAI messages[] into (system, history, final_user) for render_prompt.
void split_messages(JsonArray const& messages, std::string& system,
                    std::vector<::xllama::ChatTurn>& history, std::string& final_user) {
    std::string cur_user;
    bool have_user = false;
    for (uint32_t i = 0; i < messages.Size(); ++i) {
        const JsonObject m = messages.GetObjectAt(i);
        const std::string role = winrt::to_string(m.GetNamedString(L"role", L""));
        const std::string content = winrt::to_string(m.GetNamedString(L"content", L""));
        if (role == "system") {
            if (!system.empty())
                system += "\n";
            system += content;
        } else if (role == "user") {
            if (have_user) // consecutive user turns: close the previous unanswered
                history.push_back({cur_user, ""});
            cur_user = content;
            have_user = true;
        } else if (role == "assistant") {
            if (have_user) {
                history.push_back({cur_user, content});
                have_user = false;
            }
        }
    }
    if (have_user)
        final_user = cur_user; // trailing user turn is the one we answer
}

// Handles one chat request under g_mtx (already locked by the caller).
std::string handle_chat_locked(const std::string& body, const char*& status) {
    JsonObject root{nullptr};
    if (!JsonObject::TryParse(winrt::to_hstring(body), root) || root == nullptr) {
        status = "400 Bad Request";
        return error_json("invalid JSON body");
    }

    std::string model = winrt::to_string(root.GetNamedString(L"model", L""));
    if (model.empty())
        model = read_local_text("model.txt"); // field-test fallback
    if (model.empty()) {
        status = "400 Bad Request";
        return error_json("missing 'model' (and no LocalState\\model.txt fallback)");
    }

    if (!root.HasKey(L"messages")) {
        status = "400 Bad Request";
        return error_json("missing 'messages'");
    }
    std::string system, final_user;
    std::vector<::xllama::ChatTurn> history;
    split_messages(root.GetNamedArray(L"messages"), system, history, final_user);

    // Lazily (re)create the Session when the requested model differs.
    if (!g_session || g_model != model) {
        std::string err;
        ::xllama::SessionParams sp;
        sp.model_path = model;
        sp.n_ctx = 2048;
        g_session = ::xllama::Session::create(sp, &err);
        g_model = model;
        if (!g_session) {
            g_model.clear();
            status = "500 Internal Server Error";
            return error_json("session create failed: " + err);
        }
    }

    const ::xllama::ChatFormat fmt = ::xllama::chat_format_for(model);
    ::xllama::GenerateParams gp;
    gp.prompt = fmt.render_prompt(system, history, final_user);
    gp.stop_sequences = fmt.stop_sequences;
    gp.reuse_kv = false; // OpenAI chat is stateless: messages[] carry full history
    // max_tokens is deprecated in favour of max_completion_tokens; accept both.
    gp.n_predict = 512;
    if (root.HasKey(L"max_completion_tokens"))
        gp.n_predict = static_cast<int>(root.GetNamedNumber(L"max_completion_tokens"));
    else if (root.HasKey(L"max_tokens"))
        gp.n_predict = static_cast<int>(root.GetNamedNumber(L"max_tokens"));
    if (root.HasKey(L"temperature"))
        gp.temperature = static_cast<float>(root.GetNamedNumber(L"temperature"));
    if (root.HasKey(L"top_p"))
        gp.top_p = static_cast<float>(root.GetNamedNumber(L"top_p"));

    const ::xllama::InferenceResult r = g_session->generate(gp);
    if (!r.success) {
        status = "500 Internal Server Error";
        return error_json("generation failed: " + r.error_msg);
    }
    const std::string content = fmt.postprocess_output(r.output_text);

    JsonObject message;
    message.Insert(L"role", JsonValue::CreateStringValue(L"assistant"));
    message.Insert(L"content", JsonValue::CreateStringValue(winrt::to_hstring(content)));
    JsonObject choice;
    choice.Insert(L"index", JsonValue::CreateNumberValue(0));
    choice.Insert(L"message", message);
    // openai-python / LangChain deserialize into Pydantic models and choke when
    // logprobs is absent; emit it explicitly as null.
    choice.Insert(L"logprobs", JsonValue::CreateNullValue());
    choice.Insert(L"finish_reason",
                  JsonValue::CreateStringValue(r.ended_with_stop ? L"stop" : L"length"));
    JsonArray choices;
    choices.Append(choice);

    JsonObject usage;
    usage.Insert(L"prompt_tokens", JsonValue::CreateNumberValue(r.n_p_eval));
    usage.Insert(L"completion_tokens", JsonValue::CreateNumberValue(r.n_eval));
    usage.Insert(L"total_tokens", JsonValue::CreateNumberValue(r.n_p_eval + r.n_eval));

    JsonObject resp;
    resp.Insert(L"id", JsonValue::CreateStringValue(L"chatcmpl-xllama"));
    resp.Insert(L"object", JsonValue::CreateStringValue(L"chat.completion"));
    resp.Insert(L"created", JsonValue::CreateNumberValue(static_cast<double>(time(nullptr))));
    resp.Insert(L"model", JsonValue::CreateStringValue(winrt::to_hstring(model)));
    resp.Insert(L"choices", choices);
    resp.Insert(L"usage", usage);
    status = "200 OK";
    return winrt::to_string(resp.Stringify());
}

// The single model the server can currently serve: the loaded one, else the
// LocalState\model.txt hint. Empty id if neither is set. g_model is mutated
// under g_mtx by chat requests; read it only if we can take the lock, else fall
// back to the file hint (which touches no shared state) to avoid a data race.
std::string current_model_id() {
    std::unique_lock<std::mutex> lk(g_mtx, std::try_to_lock);
    if (lk.owns_lock() && !g_model.empty())
        return g_model;
    return read_local_text("model.txt");
}

// OpenAI GET /v1/models — {"object":"list","data":[{id,object,created,owned_by}]}.
std::string models_json() {
    JsonArray data;
    const std::string id = current_model_id();
    if (!id.empty()) {
        JsonObject m;
        m.Insert(L"id", JsonValue::CreateStringValue(winrt::to_hstring(id)));
        m.Insert(L"object", JsonValue::CreateStringValue(L"model"));
        m.Insert(L"created", JsonValue::CreateNumberValue(static_cast<double>(time(nullptr))));
        m.Insert(L"owned_by", JsonValue::CreateStringValue(L"xllama"));
        data.Append(m);
    }
    JsonObject root;
    root.Insert(L"object", JsonValue::CreateStringValue(L"list"));
    root.Insert(L"data", data);
    return winrt::to_string(root.Stringify());
}

// Ollama GET /api/tags — {"models":[{name,model}]} for Ollama-probing clients.
std::string tags_json() {
    JsonArray models;
    const std::string id = current_model_id();
    if (!id.empty()) {
        JsonObject m;
        m.Insert(L"name", JsonValue::CreateStringValue(winrt::to_hstring(id)));
        m.Insert(L"model", JsonValue::CreateStringValue(winrt::to_hstring(id)));
        models.Append(m);
    }
    JsonObject root;
    root.Insert(L"models", models);
    return winrt::to_string(root.Stringify());
}

void handle_connection(StreamSocket const& socket) {
    try {
        const HttpRequest req = read_request(socket);
        if (!req.ok) {
            write_response(socket, "400 Bad Request", error_json("malformed request"));
            return;
        }

        // CORS preflight for browser clients (Continue.dev webview, web UIs).
        if (req.method == "OPTIONS") {
            write_cors_preflight(socket);
            return;
        }

        // Health probe (also the spike gate: curl http://<ip>:<port>/ -> 200).
        if (req.method == "GET" && (req.path == "/" || req.path == "/health")) {
            write_response(socket, "200 OK", "{\"status\":\"ok\",\"service\":\"xllama\"}");
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

        if (req.method == "POST" && req.path == "/v1/chat/completions") {
            std::unique_lock<std::mutex> lk(g_mtx, std::try_to_lock);
            if (!lk.owns_lock()) {
                write_response(socket, "503 Service Unavailable", error_json("busy"));
                return;
            }
            const char* status = "200 OK";
            const std::string json = handle_chat_locked(req.body, status);
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

} // namespace

void run_server() {
    try {
        const int port = listen_port();
        g_listener = StreamSocketListener();
        g_listener.ConnectionReceived([](StreamSocketListener const&,
                                         StreamSocketListenerConnectionReceivedEventArgs const& a) {
            handle_connection(a.Socket());
        });
        g_listener.BindServiceNameAsync(winrt::to_hstring(std::to_string(port))).get();
        ::xllama::log_output("[xllama] api: listening on 0.0.0.0:" + std::to_string(port) +
                             " (OpenAI-compat /v1/chat/completions)\n");

        // Block forever: the listener stays bound for the app lifetime.
        std::promise<void>().get_future().wait();
    } catch (winrt::hresult_error const& e) {
        char buf[256];
        snprintf(buf, sizeof(buf), "[xllama] api: listener bind failed 0x%08X\n",
                 static_cast<unsigned>(e.code().value));
        ::xllama::log_output(buf);
    } catch (const std::exception& e) {
        ::xllama::log_output(std::string("[xllama] api: fatal: ") + e.what() + "\n");
    }
}

} // namespace xllama::api

#else // !XLLAMA_UWP

namespace xllama::api {
void run_server() {}
} // namespace xllama::api

#endif // XLLAMA_UWP
