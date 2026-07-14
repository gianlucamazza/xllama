// Copyright (c) 2024 Venere Labs
// SPDX-License-Identifier: MIT

#ifdef XLLAMA_UWP

// clang-format off
#include "pch.h"
#include "model-downloader.h"
// clang-format on

    #include "xllama/platform.h"
    #include "xllama/utf8_utils.h"

    #include <winrt/Windows.Data.Json.h>

    #include <filesystem>

using namespace winrt;
using namespace winrt::Windows::Foundation;
using namespace winrt::Windows::Storage;
using namespace winrt::Windows::Storage::Streams;
using namespace winrt::Windows::Web::Http;
using namespace winrt::Windows::Web::Http::Filters;
using namespace winrt::Windows::UI::Core;

namespace xllama {

static constexpr wchar_t kCompleteMarker[] = L".complete";

bool ModelDownloader::IsComplete(std::wstring const& local_dir) {
    std::filesystem::path p(local_dir);
    p /= kCompleteMarker;
    return std::filesystem::exists(p);
}

void ModelDownloader::Invalidate(std::wstring const& local_dir) {
    std::filesystem::path p(local_dir);
    p /= kCompleteMarker;
    std::error_code ec;
    std::filesystem::remove(p, ec);
}

IAsyncAction ModelDownloader::DownloadAsync(std::wstring hf_repo_url, std::wstring local_dir,
                                            std::vector<ModelFile> files, CoreDispatcher dispatcher,
                                            std::function<void(uint64_t, uint64_t)> on_progress,
                                            std::function<void(bool, std::wstring)> on_done) {
    // Compute total bytes for progress display.
    uint64_t total_bytes = 0;
    for (auto const& f : files)
        total_bytes += f.approx_bytes;

    co_await resume_background();

    HttpBaseProtocolFilter filter;
    filter.AllowAutoRedirect(true);
    HttpClient client(filter);

    // Default User-Agent is blocked by some CDNs; set something reasonable.
    client.DefaultRequestHeaders().UserAgent().TryParseAdd(L"xllama/0.1 WinRT");

    uint64_t bytes_done = 0;
    constexpr uint32_t kBufSize = 256 * 1024; // 256 KB read buffer

    for (auto const& f : files) {
        auto url_str = hf_repo_url + L"/" + (f.remote.empty() ? f.filename : f.remote);
        Uri uri(url_str);

        // --- Send HTTP request ------------------------------------------------
        HttpRequestMessage req(HttpMethod::Get(), uri);
        HttpResponseMessage resp{nullptr};
        std::wstring send_err;
        try {
            resp = co_await client.SendRequestAsync(req, HttpCompletionOption::ResponseHeadersRead);
        } catch (...) {
            send_err = L"Network error downloading " + f.filename;
        }
        if (!send_err.empty()) {
            co_await resume_foreground(dispatcher);
            on_done(false, send_err);
            co_return;
        }

        if (!resp.IsSuccessStatusCode()) {
            auto msg = L"HTTP " + std::to_wstring(static_cast<int>(resp.StatusCode())) + L" for " +
                       f.filename;
            co_await resume_foreground(dispatcher);
            on_done(false, msg);
            co_return;
        }

        // --- Open target file -------------------------------------------------
        // filename may carry a relative subpath (e.g. "unet/model.onnx" for the
        // diffusion components); create intermediate folders as needed.
        StorageFolder folder{nullptr};
        StorageFile out_file{nullptr};
        std::wstring open_err;
        try {
            folder = co_await StorageFolder::GetFolderFromPathAsync(local_dir);
            std::wstring leaf = f.filename;
            size_t sep;
            while ((sep = leaf.find_first_of(L"/\\")) != std::wstring::npos) {
                folder = co_await folder.CreateFolderAsync(winrt::hstring(leaf.substr(0, sep)),
                                                           CreationCollisionOption::OpenIfExists);
                leaf = leaf.substr(sep + 1);
            }
            out_file = co_await folder.CreateFileAsync(winrt::hstring(leaf),
                                                       CreationCollisionOption::ReplaceExisting);
        } catch (...) {
            open_err = L"Cannot create file " + f.filename + L" in " + local_dir;
        }
        if (!open_err.empty()) {
            co_await resume_foreground(dispatcher);
            on_done(false, open_err);
            co_return;
        }

        // --- Open output stream -----------------------------------------------
        IRandomAccessStream out_stream{nullptr};
        std::wstring stream_err;
        try {
            out_stream = co_await out_file.OpenAsync(FileAccessMode::ReadWrite);
        } catch (...) {
            stream_err = L"Cannot open " + f.filename + L" for write";
        }
        if (!stream_err.empty()) {
            co_await resume_foreground(dispatcher);
            on_done(false, stream_err);
            co_return;
        }

        // --- Stream response body to file in chunks ---------------------------
        auto content_stream = co_await resp.Content().ReadAsInputStreamAsync();
        auto out_writer = DataWriter(out_stream.GetOutputStreamAt(0));
        Buffer buf(kBufSize);

        bool read_failed = false;
        std::wstring read_err;
        for (;;) {
            IBuffer read_buf{nullptr};
            try {
                read_buf =
                    co_await content_stream.ReadAsync(buf, kBufSize, InputStreamOptions::Partial);
            } catch (...) {
                read_err = L"Read error on " + f.filename;
                read_failed = true;
            }
            if (read_failed)
                break;

            if (read_buf.Length() == 0)
                break;

            out_writer.WriteBuffer(read_buf);
            co_await out_writer.StoreAsync();

            bytes_done += read_buf.Length();

            // Progress: throttle dispatcher calls to avoid flooding UI thread.
            // ~every 512 KB is fine for a 403 MB file.
            if (bytes_done % (512 * 1024) < kBufSize) {
                auto snap_done = bytes_done;
                auto snap_total = total_bytes;
                co_await resume_foreground(dispatcher);
                on_progress(snap_done, snap_total);
                co_await resume_background();
            }
        }

        if (read_failed) {
            co_await resume_foreground(dispatcher);
            on_done(false, read_err);
            co_return;
        }

        co_await out_writer.FlushAsync();
        out_writer.DetachStream();
        out_stream = nullptr; // release ref → stream closed via RAII

        log_output(("[downloader] " + ::xllama::wstring_to_utf8(f.filename) + " done (" +
                    std::to_string(bytes_done) + " bytes total so far)")
                       .c_str());
    }

    // Write .complete marker.
    {
        bool marker_ok = true;
        StorageFolder mfolder{nullptr};
        StorageFile marker{nullptr};
        try {
            mfolder = co_await StorageFolder::GetFolderFromPathAsync(local_dir);
            marker = co_await mfolder.CreateFileAsync(kCompleteMarker,
                                                      CreationCollisionOption::ReplaceExisting);
        } catch (...) {
            marker_ok = false;
        }
        if (marker_ok && marker) {
            try {
                co_await FileIO::WriteTextAsync(marker, L"ok");
            } catch (...) {
                marker_ok = false;
            }
        }
        if (!marker_ok) {
            // Non-fatal: worst case we re-download next time.
            log_output("[downloader] WARNING: could not write .complete marker");
        }
    }

    co_await resume_foreground(dispatcher);
    on_done(true, L"");
}

// ---------------------------------------------------------------------------
// Model catalogue (models/manifest.json)
// ---------------------------------------------------------------------------

namespace {

// Parse a manifest JSON document into entries; returns empty on any shape error.
std::vector<ManifestEntry> parse_manifest(winrt::hstring const& text) {
    using winrt::Windows::Data::Json::JsonObject;
    std::vector<ManifestEntry> out;
    JsonObject root{nullptr};
    if (!JsonObject::TryParse(text, root))
        return out;
    if (!root.HasKey(L"models"))
        return out;
    for (auto const& item : root.GetNamedArray(L"models")) {
        auto obj = item.GetObject();
        ManifestEntry e;
        e.name = obj.GetNamedString(L"name", L"");
        if (e.name.empty())
            continue;
        e.display = obj.GetNamedString(L"display", winrt::hstring(e.name));
        e.kind = obj.GetNamedString(L"kind", L"ort-genai");
        e.hf_base_url = obj.GetNamedString(L"hf_base_url", L"");
        if (obj.HasKey(L"files")) {
            for (auto const& f : obj.GetNamedArray(L"files")) {
                auto fo = f.GetObject();
                ModelFile mf;
                mf.filename = fo.GetNamedString(L"filename", L"");
                mf.remote = fo.GetNamedString(L"remote", L"");
                mf.approx_bytes = (uint64_t)fo.GetNamedNumber(L"approx_bytes", 0);
                if (!mf.filename.empty())
                    e.files.push_back(std::move(mf));
            }
        }
        out.push_back(std::move(e));
    }
    return out;
}

std::vector<ManifestEntry> read_manifest_file(std::wstring const& path) {
    FILE* fp = _wfopen(path.c_str(), L"rb");
    if (!fp)
        return {};
    std::string bytes;
    char buf[4096];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), fp)) > 0)
        bytes.append(buf, n);
    fclose(fp);
    return parse_manifest(winrt::hstring(utf8_to_wstring(bytes)));
}

namespace {

bool dir_has_ort_or_gguf(const std::filesystem::path& dir) {
    std::error_code ec;
    if (std::filesystem::exists(dir / L"genai_config.json", ec))
        return true;
    for (const auto& ent : std::filesystem::directory_iterator(dir, ec)) {
        if (!ent.is_regular_file(ec))
            continue;
        auto ext = ent.path().extension().wstring();
        if (ext == L".gguf")
            return true;
    }
    return false;
}

} // namespace

bool IsModelProvisioned(std::wstring const& model_name) {
    if (model_name.empty())
        return false;

    auto local = ApplicationData::Current().LocalFolder();
    std::wstring local_dir = std::wstring(local.Path().c_str()) + L"\\models\\" + model_name;
    if (ModelDownloader::IsComplete(local_dir))
        return true;
    if (dir_has_ort_or_gguf(std::filesystem::path(local_dir)))
        return true;

    try {
        auto pkg = winrt::Windows::ApplicationModel::Package::Current();
        std::wstring installed =
            std::wstring(pkg.InstalledPath().c_str()) + L"\\models\\" + model_name;
        if (dir_has_ort_or_gguf(std::filesystem::path(installed)))
            return true;
    } catch (...) {
    }

    // USB cache written by EnsureModelAsync when a removable model is found.
    std::wstring cache_path = std::wstring(local.Path().c_str()) + L"\\usb_model_root.txt";
    FILE* fp = _wfopen(cache_path.c_str(), L"r");
    if (fp) {
        wchar_t root[512] = {};
        if (fgetws(root, static_cast<int>(std::size(root)), fp)) {
            std::wstring usb = root;
            while (!usb.empty() && (usb.back() == L'\n' || usb.back() == L'\r' || usb.back() == L' '))
                usb.pop_back();
            if (!usb.empty() && usb.back() == L'\\')
                usb.pop_back();
            if (!usb.empty()) {
                std::wstring usb_dir = usb + L"\\xllama\\models\\" + model_name;
                if (dir_has_ort_or_gguf(std::filesystem::path(usb_dir)))
                    return true;
            }
        }
        fclose(fp);
    }
    return false;
}

std::vector<ManifestEntry> LoadModelManifest() {
    // 1. Bundled catalogue (base).
    auto pkg = winrt::Windows::ApplicationModel::Package::Current();
    auto entries =
        read_manifest_file(std::wstring(pkg.InstalledPath().c_str()) + L"\\models\\manifest.json");
    // 2. LocalState override (uploadable via Device Portal, no reinstall),
    // merged PER ENTRY: a same-name entry replaces the bundled one, a new name
    // is appended. The override no longer hides bundled entries it does not
    // mention — a stale override used to shadow the whole catalogue (found
    // 2026-07-10: an Exp-2-era single-entry override made sd-turbo-fp16 and
    // the GGUF entries invisible after the catalogue grew).
    auto local = winrt::Windows::Storage::ApplicationData::Current().LocalFolder();
    auto override_entries =
        read_manifest_file(std::wstring(local.Path().c_str()) + L"\\manifest.json");
    if (!override_entries.empty()) {
        log_output("[manifest] merging LocalState\\manifest.json override (per-entry)\n");
        for (auto& oe : override_entries) {
            bool replaced = false;
            for (auto& e : entries) {
                if (e.name == oe.name) {
                    e = std::move(oe);
                    replaced = true;
                    break;
                }
            }
            if (!replaced)
                entries.push_back(std::move(oe));
        }
    }
    if (!entries.empty())
        return entries;
    // 3. Built-in fallback — the historical hardcoded catalogue, so the app
    // never starts with an empty model list even on a broken deployment.
    log_output("[manifest] WARNING: no manifest.json found, using built-in fallback\n");
    ManifestEntry e;
    e.name = L"smollm2-360m-cpu-int4";
    e.display = L"SmolLM2 360M (CPU int4)";
    e.hf_base_url = L"https://github.com/gianlucamazza/xllama/"
                    L"releases/download/models-v1";
    // NOTE: no special_tokens_map.json — the HF repo does not have one (the old
    // hardcoded list requested it and got HTTP 404, breaking every download).
    e.files = {
        {L"genai_config.json", L"", 2'000},
        {L"tokenizer.json", L"", 3'600'000},
        {L"tokenizer_config.json", L"", 1'000},
        {L"model.onnx", L"", 417'404'408},
    };
    return {e};
}

} // namespace xllama

#endif // XLLAMA_UWP
