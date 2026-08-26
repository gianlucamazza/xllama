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
#include <winrt/Windows.Data.Json.h>
#include <winrt/Windows.Networking.Sockets.h>
// clang-format on

    #include "inference-bridge.h"
    #include "model-downloader.h"
    #include "xllama/api_policy.h"
    #include "xllama/chat_prompt.h"
    #include "xllama/model_provision.h"
    #include "xllama/path_utils.h"
    #include "xllama/personalize.h"
    #include "xllama/platform.h"
    #include "xllama/preference_capture.h"
    #include "xllama/prompt_budget.h"
    #include "xllama/routing_policy.h"
    #include "xllama/session.h"
    #include "xllama/session_hub.h"
    #include "xllama/utf8_utils.h"

    #include <algorithm>
    #include <cctype>
    #include <cstdio>
    #include <ctime>
    #include <filesystem>
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
// The Session lives in xllama::session_hub() — ONE process-wide owner shared
// with the GUI (previously each surface owned its own Session and both could
// hold a model at once, breaking the "never 2× model in RAM" invariant).
// hub.mtx serializes (re)creation and generate(): xllama::Session::generate()
// is not concurrent (session.h). A request that finds the hub busy — another
// API request OR a GUI turn — gets 503, exactly the old busy semantics
// widened to the whole process.
// ---------------------------------------------------------------------------
std::atomic<uint64_t> g_req_counter{0}; // monotonic, for unique response ids

// Acquire the hub for a request. Busy (another request or a GUI turn) →
// unowned lock, caller answers 503. But if the holder is the session
// PRE-LOAD (app just became Ready), wait briefly instead: the client's very
// first request used to bounce with 503 while the model was warming up
// (observed on-console). Bounded: ≤15 s, and only while preloading is set.
std::unique_lock<std::mutex> acquire_hub_or_busy() {
    auto& hub = ::xllama::session_hub();
    std::unique_lock<std::mutex> lk(hub.mtx, std::try_to_lock);
    for (int i = 0; i < 150 && !lk.owns_lock() && hub.preloading.load(); ++i) {
        Sleep(100);
        (void)lk.try_lock();
    }
    return lk;
}

std::mutex g_state_mtx;
ServerStatus g_status;
uint64_t g_generation = 0; // invalidates callbacks accepted by an older listener

// Settings and autopilot can request lifecycle changes from separate worker
// threads. Serialize bind/close so only one transition owns the listener.
std::mutex g_control_mtx;

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
    bool chunked = false; // Transfer-Encoding: chunked (unsupported framing)
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
    // Drop the query string so "/v1/models?foo=1" still routes.
    if (const size_t q = req.path.find('?'); q != std::string::npos)
        req.path.erase(q);

    // Header scan (case-insensitive). We only support Content-Length framing;
    // chunked is flagged so the caller can reject it cleanly instead of parsing
    // an empty body.
    size_t content_length = 0;
    {
        const std::string headers = to_lower(data.substr(line_end + 2, header_end - line_end - 2));
        if (headers.find("transfer-encoding:") != std::string::npos &&
            headers.find("chunked") != std::string::npos) {
            req.chunked = true;
            req.ok = true;
            return req;
        }
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
        p.n_ctx = ::xllama::resolve_n_ctx(entry->n_ctx);
        p.coding = ::xllama::role_is_coding(::xllama::wstring_to_utf8(entry->role));
        p.gguf = entry->kind == L"gguf";
    }
    return p;
}

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

// Handles one chat request under session_hub().mtx (already locked by the caller).
std::string handle_chat_locked(const std::string& body, const char*& status) {
    JsonObject root{nullptr};
    if (!JsonObject::TryParse(winrt::to_hstring(body), root) || root == nullptr) {
        status = "400 Bad Request";
        return error_json("invalid JSON body");
    }

    // The endpoint does not execute tools. Reject them explicitly instead of
    // silently ignoring a client's requested capability, which could make a
    // local model appear more trusted than the runtime actually is.
    const ::xllama::ApiToolFields tool_fields{root.HasKey(L"tools"), root.HasKey(L"functions"),
                                              root.HasKey(L"tool_choice")};
    if (::xllama::api_tool_execution_requested(tool_fields)) {
        status = "400 Bad Request";
        return error_json("tool execution is not supported; tools and functions are disabled");
    }

    std::string model = winrt::to_string(root.GetNamedString(L"model", L""));
    if (model.empty())
        model = read_local_text("model.txt"); // field-test fallback
    if (model.empty()) {
        status = "400 Bad Request";
        return error_json("missing 'model' (and no LocalState\\model.txt fallback)");
    }

    if (!root.HasKey(L"messages") ||
        root.GetNamedValue(L"messages").ValueType() != JsonValueType::Array) {
        status = "400 Bad Request";
        return error_json("missing or non-array 'messages'");
    }
    std::string system, final_user;
    std::vector<::xllama::ChatTurn> history;
    split_messages(root.GetNamedArray(L"messages"), system, history, final_user);
    if (final_user.empty()) {
        status = "400 Bad Request";
        return error_json("no user message to complete");
    }
    // Lazily (re)create the resident Session when the requested model differs
    // (hub.mtx is held by the caller; the swap invalidates the GUI's KV-reuse
    // state via hub.generation, which its next turn detects). Catalogue n_ctx
    // / role (coding) apply the same policy as the chat UI.
    ::xllama::Session* session = nullptr;
    bool model_is_coding = false;
    int policy_n_ctx = ::xllama::kDefaultNCtx;
    {
        std::string err;
        const CatalogueSessionPolicy policy = catalogue_session_policy(model);
        model_is_coding = policy.coding;
        policy_n_ctx = policy.n_ctx;
        ::xllama::SessionParams sp;
        sp.model_path = model;
        sp.n_ctx = policy.n_ctx;
        if (policy.gguf)
            sp.backend = ::xllama::Backend::LlamaCpp;
        session = ::xllama::session_hub().ensure_locked(model, sp, &err);
        if (!session) {
            status = "500 Internal Server Error";
            return error_json("session create failed: " + err);
        }
    }

    // Small instruct models degrade badly with an empty system turn (they
    // hallucinate the next role instead of answering); the chat UI always seeds
    // one. Match it when the client sends no system message — coding models get
    // the coding default so a bare LAN client does not look like general chat.
    if (system.empty())
        system = model_is_coding ? ::xllama::kCodingSystemPrompt : ::xllama::kDefaultSystemPrompt;

    const ::xllama::ChatFormat fmt = ::xllama::chat_format_for(model);
    ::xllama::GenerateParams gp;
    gp.stop_sequences = fmt.stop_sequences;
    gp.reuse_kv = false; // OpenAI chat is stateless: messages[] carry full history
    // max_tokens is deprecated in favour of max_completion_tokens; accept both.
    // Parsed BEFORE the prompt is built: the budget below needs to know how much
    // room the reply asks for.
    gp.n_predict = 512;
    if (root.HasKey(L"max_completion_tokens"))
        gp.n_predict = static_cast<int>(root.GetNamedNumber(L"max_completion_tokens"));
    else if (root.HasKey(L"max_tokens"))
        gp.n_predict = static_cast<int>(root.GetNamedNumber(L"max_tokens"));

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
            return error_json("prompt too long: the final user message needs " +
                              std::to_string(fit.n_tokens) + " tokens plus " +
                              std::to_string(gp.n_predict) + " for the reply, but n_ctx is " +
                              std::to_string(policy_n_ctx));
        }
        if (fit.dropped > 0)
            ::xllama::log_output("[xllama] api: dropped " + std::to_string(fit.dropped) +
                                 " oldest message(s) to fit n_ctx " + std::to_string(policy_n_ctx) +
                                 "\n");
        gp.prompt = fit.prompt;
    }
    if (root.HasKey(L"temperature"))
        gp.temperature = static_cast<float>(root.GetNamedNumber(L"temperature"));
    if (root.HasKey(L"top_p"))
        gp.top_p = static_cast<float>(root.GetNamedNumber(L"top_p"));
    if (root.HasKey(L"seed")) // reproducibility for a research endpoint
        gp.seed = static_cast<uint32_t>(root.GetNamedNumber(L"seed", -1.0));
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

    const ::xllama::InferenceResult r = session->generate(gp);
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
    // "length" only when we actually hit the token cap; a model that ends on its
    // EOS token before the cap stops naturally, which ended_with_stop (a textual
    // stop-sequence match) does not capture — deduce it from the token count.
    const bool hit_cap = r.n_eval >= gp.n_predict;
    choice.Insert(L"finish_reason", JsonValue::CreateStringValue(hit_cap ? L"length" : L"stop"));
    JsonArray choices;
    choices.Append(choice);

    JsonObject usage;
    usage.Insert(L"prompt_tokens", JsonValue::CreateNumberValue(r.n_p_eval));
    usage.Insert(L"completion_tokens", JsonValue::CreateNumberValue(r.n_eval));
    usage.Insert(L"total_tokens", JsonValue::CreateNumberValue(r.n_p_eval + r.n_eval));

    JsonObject resp;
    // Unique per response (id is keyed by clients / trace dedup). hub.mtx is held.
    const std::string id =
        "chatcmpl-xllama-" + std::to_string(time(nullptr)) + "-" + std::to_string(++g_req_counter);
    resp.Insert(L"id", JsonValue::CreateStringValue(winrt::to_hstring(id)));
    resp.Insert(L"object", JsonValue::CreateStringValue(L"chat.completion"));
    resp.Insert(L"created", JsonValue::CreateNumberValue(static_cast<double>(time(nullptr))));
    resp.Insert(L"model", JsonValue::CreateStringValue(winrt::to_hstring(model)));
    resp.Insert(L"choices", choices);
    resp.Insert(L"usage", usage);
    status = "200 OK";
    return winrt::to_string(resp.Stringify());
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
    return read_local_text("model.txt");
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

// OpenAI GET /v1/models — {"object":"list","data":[{id,object,created,owned_by}]}
// listing every servable on-device model. The non-standard "active" flag marks
// the model the lazily-owned Session currently has loaded (clients ignore
// unknown fields); any listed id is valid for POST "model" — the server
// switches Sessions on demand.
std::string models_json() {
    JsonArray data;
    const std::string current = current_model_id();
    for (const std::string& id : servable_model_ids()) {
        JsonObject m;
        m.Insert(L"id", JsonValue::CreateStringValue(winrt::to_hstring(id)));
        m.Insert(L"object", JsonValue::CreateStringValue(L"model"));
        m.Insert(L"created", JsonValue::CreateNumberValue(static_cast<double>(time(nullptr))));
        m.Insert(L"owned_by", JsonValue::CreateStringValue(L"xllama"));
        if (id == current)
            m.Insert(L"active", JsonValue::CreateBooleanValue(true));
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
    const std::string done_raw = read_local_text("training/result.done");
    const std::string done = ::xllama::parse_train_result_done(done_raw);
    const std::string progress = read_local_text("training/progress.json");
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
    int steps = 1;
    if (root.HasKey(L"steps"))
        steps = static_cast<int>(root.GetNamedNumber(L"steps", 1));
    if (steps < 1)
        steps = 1;
    if (steps > 4)
        steps = 4; // UI Image dialog max
    int seed = 42;
    if (root.HasKey(L"seed"))
        seed = static_cast<int>(root.GetNamedNumber(L"seed", 42));
    if (seed < 0)
        seed = 42;

    // Drive the same LocalState knobs as the Image dialog / autopilot.
    {
        auto write = [](const char* name, const std::string& v) {
            FILE* f = _wfopen(::xllama::utf8_to_wstring(::xllama::resolve_local_path(name)).c_str(),
                              L"wb");
            if (f) {
                fwrite(v.data(), 1, v.size(), f);
                fclose(f);
            }
        };
        write("prompt.txt", prompt);
        write("diffuse-steps.txt", std::to_string(steps));
        write("diffuse-seed.txt", std::to_string(seed));
        write("diffuse-model.txt", "sd-turbo-fp16");
        _wremove(
            ::xllama::utf8_to_wstring(::xllama::resolve_local_path("diffuse-cancel.flag")).c_str());
    }

    ::xllama::bridge::run_diffuse();

    const std::string stage = read_local_text("diffuse-progress.txt");
    if (stage != "done") {
        status = "500 Internal Server Error";
        return error_json(std::string("image generation failed: stage=") + stage);
    }
    const std::string png_path = ::xllama::resolve_local_path("diffuse-out.png");
    const std::string png = read_file_bytes(png_path);
    if (png.empty()) {
        status = "500 Internal Server Error";
        return error_json("diffuse-out.png missing after generation");
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

void handle_connection(StreamSocket const& socket, uint64_t generation) {
    try {
        const HttpRequest req = read_request(socket);
        if (!req.ok) {
            write_response(socket, "400 Bad Request", error_json("malformed request"));
            return;
        }
        if (req.chunked) {
            write_response(
                socket, "411 Length Required",
                error_json("chunked transfer-encoding not supported; send Content-Length"));
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
            std::unique_lock<std::mutex> lk = acquire_hub_or_busy();
            if (!lk.owns_lock()) {
                write_response(socket, "503 Service Unavailable", error_json("busy"));
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
                write_response(socket, "503 Service Unavailable", error_json("server stopped"));
                return;
            }
            // handle_chat_locked touches WinRT JSON accessors that throw on
            // wrong-typed fields; guarantee the client always gets a response
            // rather than a silently dropped socket.
            const char* status = "200 OK";
            std::string json;
            try {
                json = handle_chat_locked(req.body, status);
            } catch (...) {
                status = "400 Bad Request";
                json = error_json("malformed request body");
            }
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
                status = "400 Bad Request";
                json = error_json("malformed request body");
            }
            write_response(socket, status, json);
            return;
        }

        // #118: training status (read-only files).
        if (req.method == "GET" && req.path == "/v1/training/status") {
            write_response(socket, "200 OK", handle_training_status());
            return;
        }

        // #118: image generation — shares the single-slot mutex with chat.
        if (req.method == "POST" && req.path == "/v1/images/generations") {
            std::unique_lock<std::mutex> lk = acquire_hub_or_busy();
            if (!lk.owns_lock()) {
                write_response(socket, "503 Service Unavailable", error_json("busy"));
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
            const char* status = "200 OK";
            std::string json;
            try {
                json = handle_images_locked(req.body, status);
            } catch (...) {
                status = "500 Internal Server Error";
                json = error_json("image generation failed");
            }
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
    // solely via the API simply stays resident; the GUI swaps it on next use.
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
                handle_connection(a.Socket(), generation);
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

} // namespace xllama::api

#else // !XLLAMA_UWP

namespace xllama::api {
void start_server(int) {}
void stop_server() {}
ServerStatus server_status() {
    return {};
}
void run_server() {}
} // namespace xllama::api

#endif // XLLAMA_UWP
