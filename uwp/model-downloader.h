// Copyright (c) 2024 Venere Labs
// SPDX-License-Identifier: MIT

#pragma once

#ifdef XLLAMA_UWP

    #include "pch.h"

    #include <functional>
    #include <string>
    #include <vector>

namespace xllama {

struct ModelFile {
    std::wstring filename;
    uint64_t approx_bytes; // 0 = unknown; used for progress display only
};

// Async download of an ONNX GenAI model from a Hugging Face repository to
// ApplicationData LocalFolder. Callbacks are invoked on the UI thread.
class ModelDownloader {
  public:
    // Returns true if all files have been downloaded (marker file present).
    static bool IsComplete(std::wstring const& local_dir);

    // Remove the .complete marker to force a re-download on next launch.
    static void Invalidate(std::wstring const& local_dir);

    // Download all files in |files| from |hf_repo_url|/<filename> to
    // |local_dir|/<filename>. |local_dir| must already exist.
    // on_progress(bytes_done, bytes_total): called periodically; bytes_total
    //   is the sum of approx_bytes across files (may be 0 if unknown).
    // on_done(success, error_message): called exactly once when finished.
    static winrt::Windows::Foundation::IAsyncAction DownloadAsync(
        std::wstring hf_repo_url,
        std::wstring local_dir,
        std::vector<ModelFile> files,
        winrt::Windows::UI::Core::CoreDispatcher dispatcher,
        std::function<void(uint64_t, uint64_t)> on_progress,
        std::function<void(bool, std::wstring)> on_done);
};

// File manifest for SmolLM2-360M-Instruct-ort-genai-int4-cpu (merged model.onnx).
inline std::vector<ModelFile> SmolLM2_360M_Files() {
    return {
        {L"genai_config.json", 2'000},
        {L"tokenizer.json", 2'400'000},
        {L"tokenizer_config.json", 3'000},
        {L"special_tokens_map.json", 1'000},
        {L"model.onnx", 422'000'000}, // ~403 MB merged
    };
}

} // namespace xllama

#endif // XLLAMA_UWP
