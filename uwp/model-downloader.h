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
    std::wstring
        filename;        // local path under the model dir; may contain subdirs ("unet/model.onnx")
    std::wstring remote; // asset name in the download source (release assets are flat);
                         // empty = same as filename
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
    static winrt::Windows::Foundation::IAsyncAction
    DownloadAsync(std::wstring hf_repo_url, std::wstring local_dir, std::vector<ModelFile> files,
                  winrt::Windows::UI::Core::CoreDispatcher dispatcher,
                  std::function<void(uint64_t, uint64_t)> on_progress,
                  std::function<void(bool, std::wstring)> on_done);
};

// One entry of the model catalogue (models/manifest.json). An empty hf_base_url
// means the model cannot be auto-downloaded (USB/Device-Portal provisioning only).
// kind selects the consumer AND the backend: "ort-genai" (default, chat picker,
// ORT GenAI backend), "diffusion" (image dialog; hidden from the chat picker), or
// "gguf" (chat picker, llama.cpp backend). The gguf path is CPU-only on Xbox (no
// EP routing — llama.cpp UWP build is CPU-only) but KV-reuse IS enabled via a
// persistent llama_context (turn-2 prefill 4.07×, see docs/benchmarks.md).
struct ManifestEntry {
    std::wstring name;
    std::wstring display;
    std::wstring kind{L"ort-genai"};
    std::wstring hf_base_url;
    std::vector<ModelFile> files;
};

// Load the model catalogue: InstalledPath\models\manifest.json (bundled) is
// the base; LocalState\manifest.json (uploadable via Device Portal, no
// reinstall) is merged PER ENTRY on top — same-name entries replace the
// bundled ones, new names are appended, unmentioned bundled entries stay.
// Falls back to a built-in single-entry catalogue (the historical hardcoded
// SmolLM2-360M) if neither parses, so the app never starts with an empty list.
std::vector<ManifestEntry> LoadModelManifest();

// Find an entry by model dir name; nullptr-like (empty name) if absent.
inline const ManifestEntry* FindManifestEntry(const std::vector<ManifestEntry>& m,
                                              const std::wstring& name) {
    for (const auto& e : m)
        if (e.name == name)
            return &e;
    return nullptr;
}

// True when the model dir is usable without a download: .complete marker, a
// WDP/USB upload (genai_config.json or *.gguf present), bundled in the MSIX,
// or on removable storage at xllama\models\<name>.
bool IsModelProvisioned(std::wstring const& model_name);

} // namespace xllama

#endif // XLLAMA_UWP
