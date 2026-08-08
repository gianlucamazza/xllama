// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#ifdef XLLAMA_UWP

// clang-format off
#include "pch.h"
#include "model-downloader.h"
// clang-format on

    #include "xllama/manifest_merge.h"
    #include "xllama/model_provision.h"
    #include "xllama/platform.h"
    #include "xllama/utf8_utils.h"

    #include <winrt/Windows.Data.Json.h>

    #include <chrono>
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

namespace {

// On-disk size of local_dir/filename (subdir-aware). 0 if missing/unreadable.
uint64_t local_file_size(const std::filesystem::path& local_dir, const std::wstring& rel) {
    std::error_code ec;
    const auto p = local_dir / rel;
    if (!std::filesystem::is_regular_file(p, ec))
        return 0;
    auto sz = std::filesystem::file_size(p, ec);
    return ec ? 0 : static_cast<uint64_t>(sz);
}

// True when a previous download/WDP upload left a usable file: exact approx_bytes
// match when known; otherwise any non-empty file (loose USB/WDP).
bool file_already_complete(const std::filesystem::path& local_dir, const ModelFile& f) {
    const uint64_t have = local_file_size(local_dir, f.filename);
    if (have == 0)
        return false;
    if (f.approx_bytes == 0)
        return true;
    return have == f.approx_bytes;
}

bool http_status_retryable(int code) {
    return code == 429 || code >= 500;
}

} // namespace

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
    constexpr int kMaxAttempts = 3;
    const std::filesystem::path local_dir_fs(local_dir);

    for (auto const& f : files) {
        // Skip HTTP when a complete copy is already on disk (WDP upload or a
        // prior partial EnsureModel that left the correct size). Avoids HF 504
        // loops and re-fetching ~1.5 GB weights.
        if (file_already_complete(local_dir_fs, f)) {
            const uint64_t have = local_file_size(local_dir_fs, f.filename);
            bytes_done += (f.approx_bytes > 0 ? f.approx_bytes : have);
            log_output(("[downloader] skip " + ::xllama::wstring_to_utf8(f.filename) +
                        " (already present, " + std::to_string(have) + " bytes)")
                           .c_str());
            auto snap_done = bytes_done;
            auto snap_total = total_bytes;
            co_await resume_foreground(dispatcher);
            on_progress(snap_done, snap_total);
            co_await resume_background();
            continue;
        }

        auto url_str = hf_repo_url + L"/" + (f.remote.empty() ? f.filename : f.remote);
        Uri uri(url_str);

        std::wstring last_err;
        bool file_ok = false;

        for (int attempt = 1; attempt <= kMaxAttempts && !file_ok; ++attempt) {
            if (attempt > 1) {
                const int backoff_s = 1 << (attempt - 2); // 1s, 2s
                log_output(("[downloader] retry " + std::to_string(attempt) + "/" +
                            std::to_string(kMaxAttempts) + " for " +
                            ::xllama::wstring_to_utf8(f.filename) + " after " +
                            std::to_string(backoff_s) + "s")
                               .c_str());
                co_await winrt::resume_after(std::chrono::seconds(backoff_s));
            }

            // --- Send HTTP request --------------------------------------------
            HttpRequestMessage req(HttpMethod::Get(), uri);
            HttpResponseMessage resp{nullptr};
            last_err.clear();
            // co_await is illegal inside a catch block (MSVC C2304): flag the
            // failure and await/report after the try/catch.
            bool net_failed = false;
            try {
                resp = co_await client.SendRequestAsync(req,
                                                        HttpCompletionOption::ResponseHeadersRead);
            } catch (...) {
                last_err = L"Network error downloading " + f.filename;
                net_failed = true;
            }
            if (net_failed) {
                if (attempt < kMaxAttempts)
                    continue;
                co_await resume_foreground(dispatcher);
                on_done(false, last_err);
                co_return;
            }

            if (!resp.IsSuccessStatusCode()) {
                const int code = static_cast<int>(resp.StatusCode());
                last_err = L"HTTP " + std::to_wstring(code) + L" for " + f.filename;
                if (http_status_retryable(code) && attempt < kMaxAttempts)
                    continue;
                co_await resume_foreground(dispatcher);
                on_done(false, last_err);
                co_return;
            }

            // --- Open target file ---------------------------------------------
            // filename may carry a relative subpath (e.g. "unet/model.onnx");
            // create intermediate folders as needed.
            //
            // Download into "<leaf>.part" and rename over the target only after
            // the full body is written. ReplaceExisting directly on the live file
            // truncated a good model.onnx first and secured bytes later — any
            // mid-download failure (or a locked mmap'd model) left a corrupt
            // 464 MB file that OgaCreateModel then failed to parse.
            StorageFolder folder{nullptr};
            StorageFile out_file{nullptr};
            std::wstring leaf;
            bool create_failed = false;
            try {
                folder = co_await StorageFolder::GetFolderFromPathAsync(local_dir);
                leaf = f.filename;
                size_t sep;
                while ((sep = leaf.find_first_of(L"/\\")) != std::wstring::npos) {
                    folder = co_await folder.CreateFolderAsync(
                        winrt::hstring(leaf.substr(0, sep)), CreationCollisionOption::OpenIfExists);
                    leaf = leaf.substr(sep + 1);
                }
                out_file = co_await folder.CreateFileAsync(
                    winrt::hstring(leaf + L".part"), CreationCollisionOption::ReplaceExisting);
            } catch (...) {
                last_err = L"Cannot create file " + f.filename + L" in " + local_dir;
                create_failed = true;
            }
            if (create_failed) {
                co_await resume_foreground(dispatcher);
                on_done(false, last_err);
                co_return;
            }

            IRandomAccessStream out_stream{nullptr};
            bool open_failed = false;
            try {
                out_stream = co_await out_file.OpenAsync(FileAccessMode::ReadWrite);
            } catch (...) {
                last_err = L"Cannot open " + f.filename + L" for write";
                open_failed = true;
            }
            if (open_failed) {
                co_await resume_foreground(dispatcher);
                on_done(false, last_err);
                co_return;
            }

            // --- Stream response body to file in chunks -----------------------
            auto content_stream = co_await resp.Content().ReadAsInputStreamAsync();
            auto out_writer = DataWriter(out_stream.GetOutputStreamAt(0));
            Buffer buf(kBufSize);

            const uint64_t bytes_before = bytes_done;
            bool read_failed = false;
            for (;;) {
                IBuffer read_buf{nullptr};
                try {
                    read_buf = co_await content_stream.ReadAsync(buf, kBufSize,
                                                                 InputStreamOptions::Partial);
                } catch (...) {
                    last_err = L"Read error on " + f.filename;
                    read_failed = true;
                }
                if (read_failed)
                    break;

                if (read_buf.Length() == 0)
                    break;

                out_writer.WriteBuffer(read_buf);
                co_await out_writer.StoreAsync();

                bytes_done += read_buf.Length();

                if (bytes_done % (512 * 1024) < kBufSize) {
                    auto snap_done = bytes_done;
                    auto snap_total = total_bytes;
                    co_await resume_foreground(dispatcher);
                    on_progress(snap_done, snap_total);
                    co_await resume_background();
                }
            }

            if (read_failed) {
                // Drop partial file; retry from scratch.
                try {
                    co_await out_writer.FlushAsync();
                } catch (...) {
                }
                out_writer.DetachStream();
                out_stream = nullptr;
                bytes_done = bytes_before;
                try {
                    co_await out_file.DeleteAsync();
                } catch (...) {
                }
                if (attempt < kMaxAttempts)
                    continue;
                co_await resume_foreground(dispatcher);
                on_done(false, last_err);
                co_return;
            }

            co_await out_writer.FlushAsync();
            out_writer.DetachStream();
            out_stream = nullptr;

            // Atomically promote the completed .part over the target. The old
            // file is only replaced once every byte is on disk; a locked target
            // fails the rename here without having destroyed the existing model.
            bool rename_failed = false;
            try {
                co_await out_file.RenameAsync(winrt::hstring(leaf),
                                              NameCollisionOption::ReplaceExisting);
            } catch (...) {
                last_err = L"Cannot replace " + f.filename + L" (target in use?)";
                rename_failed = true;
            }
            if (rename_failed) {
                try {
                    co_await out_file.DeleteAsync();
                } catch (...) {
                }
                co_await resume_foreground(dispatcher);
                on_done(false, last_err);
                co_return;
            }

            log_output(("[downloader] " + ::xllama::wstring_to_utf8(f.filename) + " done (" +
                        std::to_string(bytes_done) + " bytes total so far)")
                           .c_str());
            file_ok = true;
        }

        if (!file_ok) {
            co_await resume_foreground(dispatcher);
            on_done(false, last_err.empty() ? L"Download failed for " + f.filename : last_err);
            co_return;
        }
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
        e.lora = obj.GetNamedString(L"lora", L"");
        e.lora_scale = obj.GetNamedNumber(L"lora_scale", 1.0);
        // 0 / absent → shipping default at session open (resolve_n_ctx).
        e.n_ctx = static_cast<int>(obj.GetNamedNumber(L"n_ctx", 0));
        // 0 / absent → keep Settings / UI n_predict (global default 512).
        e.n_predict = static_cast<int>(obj.GetNamedNumber(L"n_predict", 0));
        e.role = obj.GetNamedString(L"role", L"");
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

// Directory-relative paths of every regular file under `dir` (recursive, so
// diffusion subdir files like unet/model.onnx are seen). The .complete marker is
// bookkeeping, not a model file — omit it. Empty if the dir is absent.
std::vector<std::wstring> list_relative_files(const std::filesystem::path& dir) {
    std::vector<std::wstring> out;
    std::error_code ec;
    if (!std::filesystem::is_directory(dir, ec))
        return out;
    for (const auto& ent : std::filesystem::recursive_directory_iterator(dir, ec)) {
        if (!ent.is_regular_file(ec))
            continue;
        // Lexical relative path — NOT std::filesystem::relative(), which calls
        // weakly_canonical() on both operands and fails inside the Xbox
        // AppContainer on inaccessible parent segments (same class of failure as
        // the documented ORT external-data gotcha). When it failed here, every
        // file was skipped, the list came back empty, and EnsureModel declared a
        // fully-present GPU model "missing" — triggering a destructive
        // re-download that truncated a good model.onnx.
        auto rel = ent.path().lexically_relative(dir);
        if (rel.empty())
            continue;
        std::wstring w = rel.wstring();
        if (w == L".complete")
            continue;
        out.push_back(std::move(w));
    }
    return out;
}

// Is `dir` provisioned? Loose mode (empty expected) keeps the historical
// any-gguf/genai_config check; expected mode requires the manifest's current file
// set to be present (so a stale quant no longer counts as provisioned).
bool dir_provisioned(const std::filesystem::path& dir, const std::vector<std::wstring>& expected) {
    if (expected.empty())
        return dir_has_ort_or_gguf(dir);
    return dir_satisfies_expected_files(list_relative_files(dir), expected);
}

} // namespace

bool IsModelProvisioned(std::wstring const& model_name) {
    return IsModelProvisioned(model_name, {});
}

bool IsModelProvisioned(std::wstring const& model_name,
                        std::vector<std::wstring> const& expected_files) {
    if (model_name.empty())
        return false;

    auto local = ApplicationData::Current().LocalFolder();
    std::wstring local_dir = std::wstring(local.Path().c_str()) + L"\\models\\" + model_name;
    // The .complete marker is a fast-path ONLY in loose mode: in expected mode a
    // stale-quant dir can still carry a valid old marker — exactly what defeats
    // auto-upgrade — so the expected-file scan is authoritative there.
    if (expected_files.empty() && ModelDownloader::IsComplete(local_dir))
        return true;
    if (dir_provisioned(std::filesystem::path(local_dir), expected_files))
        return true;

    try {
        auto pkg = winrt::Windows::ApplicationModel::Package::Current();
        std::wstring installed =
            std::wstring(pkg.InstalledPath().c_str()) + L"\\models\\" + model_name;
        if (dir_provisioned(std::filesystem::path(installed), expected_files))
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
            while (!usb.empty() &&
                   (usb.back() == L'\n' || usb.back() == L'\r' || usb.back() == L' '))
                usb.pop_back();
            if (!usb.empty() && usb.back() == L'\\')
                usb.pop_back();
            if (!usb.empty()) {
                std::wstring usb_dir = usb + L"\\xllama\\models\\" + model_name;
                if (dir_provisioned(std::filesystem::path(usb_dir), expected_files))
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
        merge_manifest_entries(entries, std::move(override_entries));
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
