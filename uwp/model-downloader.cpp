// Copyright (c) 2024 Venere Labs
// SPDX-License-Identifier: MIT

#ifdef XLLAMA_UWP

// clang-format off
#include "pch.h"
#include "model-downloader.h"
// clang-format on

#include "xllama/platform.h"
#include "xllama/utf8_utils.h"

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

IAsyncAction ModelDownloader::DownloadAsync(
    std::wstring hf_repo_url,
    std::wstring local_dir,
    std::vector<ModelFile> files,
    CoreDispatcher dispatcher,
    std::function<void(uint64_t, uint64_t)> on_progress,
    std::function<void(bool, std::wstring)> on_done)
{
    // Compute total bytes for progress display.
    uint64_t total_bytes = 0;
    for (auto const& f : files) total_bytes += f.approx_bytes;

    co_await resume_background();

    HttpBaseProtocolFilter filter;
    filter.AllowAutoRedirect(true);
    HttpClient client(filter);

    // Default User-Agent is blocked by some CDNs; set something reasonable.
    client.DefaultRequestHeaders().UserAgent().TryParseAdd(L"xllama/0.1 WinRT");

    uint64_t bytes_done = 0;
    constexpr uint32_t kBufSize = 256 * 1024; // 256 KB read buffer

    for (auto const& f : files) {
        auto url_str = hf_repo_url + L"/" + f.filename;
        Uri uri(url_str);

        HttpRequestMessage req(HttpMethod::Get(), uri);
        HttpResponseMessage resp{nullptr};
        try {
            resp = co_await client.SendRequestAsync(req, HttpCompletionOption::ResponseHeadersRead);
        } catch (...) {
            auto msg = L"Network error downloading " + f.filename;
            co_await resume_foreground(dispatcher);
            on_done(false, msg);
            co_return;
        }

        if (!resp.IsSuccessStatusCode()) {
            auto msg = L"HTTP " + std::to_wstring(static_cast<int>(resp.StatusCode()))
                       + L" for " + f.filename;
            co_await resume_foreground(dispatcher);
            on_done(false, msg);
            co_return;
        }

        // Open target file in local_dir.
        StorageFolder folder{nullptr};
        StorageFile out_file{nullptr};
        try {
            folder = co_await StorageFolder::GetFolderFromPathAsync(local_dir);
            out_file = co_await folder.CreateFileAsync(
                f.filename, CreationCollisionOption::ReplaceExisting);
        } catch (...) {
            auto msg = L"Cannot create file " + f.filename + L" in " + local_dir;
            co_await resume_foreground(dispatcher);
            on_done(false, msg);
            co_return;
        }

        IRandomAccessStream out_stream{nullptr};
        try {
            out_stream = co_await out_file.OpenAsync(FileAccessMode::ReadWrite);
        } catch (...) {
            auto msg = L"Cannot open " + f.filename + L" for write";
            co_await resume_foreground(dispatcher);
            on_done(false, msg);
            co_return;
        }

        // Stream response body to file in chunks.
        auto content_stream = co_await resp.Content().ReadAsInputStreamAsync();
        auto out_writer = DataWriter(out_stream.GetOutputStreamAt(0));
        Buffer buf(kBufSize);

        for (;;) {
            IBuffer read_buf{nullptr};
            try {
                read_buf = co_await content_stream.ReadAsync(
                    buf, kBufSize, InputStreamOptions::Partial);
            } catch (...) {
                auto msg = L"Read error on " + f.filename;
                co_await resume_foreground(dispatcher);
                on_done(false, msg);
                co_return;
            }

            if (read_buf.Length() == 0) break;

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

        co_await out_writer.FlushAsync();
        out_writer.DetachStream();
        out_stream = nullptr; // release ref → stream closed via RAII

        log_output(("[downloader] " + ::xllama::wstring_to_utf8(f.filename)
                    + " done (" + std::to_string(bytes_done) + " bytes total so far)")
                       .c_str());
    }

    // Write .complete marker.
    try {
        StorageFolder folder = co_await StorageFolder::GetFolderFromPathAsync(local_dir);
        StorageFile marker = co_await folder.CreateFileAsync(
            kCompleteMarker, CreationCollisionOption::ReplaceExisting);
        co_await FileIO::WriteTextAsync(marker, L"ok");
    } catch (...) {
        // Non-fatal: worst case we re-download next time.
        log_output("[downloader] WARNING: could not write .complete marker");
    }

    co_await resume_foreground(dispatcher);
    on_done(true, L"");
}

} // namespace xllama

#endif // XLLAMA_UWP
