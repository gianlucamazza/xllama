// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
#pragma once

#include "chat-history.h"
#include "xllama/diffusion/png_writer.h"
#include "xllama/model_file_plan.h"
#include "xllama/personalize.h"
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <share.h>
#include <winrt/Windows.Storage.FileProperties.h>

namespace audit_fixtures {
using namespace winrt::Windows::Storage;
namespace fs = std::filesystem;
using Json = nlohmann::json;
constexpr wchar_t good[] = L"audit-ok-8c6af4f3e94d";
constexpr wchar_t bad[] = L"audit-missing-8c6af4f3e94d";
constexpr wchar_t signature[] = L"xllama-audit-8c6af4f3e94d";
inline auto local() {
    return ApplicationData::Current().LocalFolder();
}
inline fs::path root() {
    return fs::path(local().Path().c_str());
}
inline void require(bool ok, const char* msg) {
    if (!ok)
        throw std::runtime_error(msg);
}
inline std::string read(const fs::path& file) {
    std::ifstream input(file, std::ios::binary);
    return {std::istreambuf_iterator<char>(input), {}};
}
template <class F> void receipt(const char* filename, F fn) {
    Json result = Json::object();
    try {
        fn(result);
        result["success"] = true;
    } catch (winrt::hresult_error const& e) {
        result["success"] = false;
        result["error"] = winrt::to_string(e.message());
        result["hresult"] = e.code().value;
    } catch (std::exception const& e) {
        result["success"] = false;
        result["error"] = e.what();
    }
    ::xllama::replace_file_bytes(root() / filename, result.dump(2));
}
inline void state() {
    receipt("audit-state.json", [](Json& r) {
        using namespace ::xllama;
        auto home = root() / "audit-state-run";
        fs::create_directories(home);
        ui::ChatHistory h((home / "chats").string());
        ui::Conversation conv{"id", "old", {}};
        require(h.Save(conv), "first save");
        fs::create_directory(home / "chats/id.json.tmp");
        conv.title = "new";
        require(!h.Save(conv) && h.Load("id").title == "old", "interrupted conversation");
        fs::remove(home / "chats/id.json.tmp");
        fs::create_directory(home / "chats/index.json.tmp");
        require(!h.Save(conv), "interrupt index");
        fs::remove(home / "chats/index.json.tmp");
        h.LoadIndex();
        require(h.Index().size() == 1 && h.Index()[0].title == "new", "index recovery");
        r["chat"] = "old file survived; pending index reconciled";
        std::ofstream(home / "manifest.json")
            << R"({"models":[{"name":"alias-a"},{"name":"alias-b"}]})";
        std::ofstream(home / "merged.gguf") << "fixture generation";
        std::string error;
        if (!publish_personalized_model(home, home / "merged.gguf", &error))
            throw std::runtime_error(error);
        const auto saved = Json::parse(read(home / "manifest.json"));
        require(saved.at("models").size() == 3 && saved.at("models")[0].at("name") == "alias-a" &&
                    saved.at("models")[1].at("name") == "alias-b",
                "local overrides lost");
        const auto live_manifest = home / "manifest.json";
        FILE* held = _wfsopen(live_manifest.c_str(), L"rb", _SH_DENYWR);
        require(held != nullptr, "open the manifest under a sharing lock");
        std::ofstream(home / "merged.gguf") << "locked generation";
        const auto refused = !publish_personalized_model(home, home / "merged.gguf", &error);
        std::fclose(held);
        require(refused, "manifest replacement must be refused while locked");
        require(recover_personalized_publication(home, &error), "retry publication recovery");
        require(Json::parse(read(live_manifest)) == saved &&
                    read(home / "models/personalized/model.gguf") == "fixture generation",
                "failed manifest swap changed model or overrides");
        r["manifest_sharing_failure"] = "refused and previous generation recovered";
        std::ofstream(home / "manifest.json") << "{bad";
        std::ofstream(home / "merged.gguf") << "failed generation";
        require(!publish_personalized_model(home, home / "merged.gguf", &error) &&
                    read(home / "manifest.json") == "{bad" &&
                    read(home / "models/personalized/model.gguf") == "fixture generation",
                "failed publish");
        r["local_overrides"] = "preserved both existing entries; malformed manifest refused";
        const auto image = home / "test.png";
        const auto bytes = diffusion::publish_png_rgb(image, 1, 1, {1, 2, 3});
        require(bytes && bytes == fs::file_size(image), "png length");
        auto prior = read(image);
        fs::create_directory(home / "test.png.tmp");
        require(!diffusion::publish_png_rgb(image, 1, 1, {3, 2, 1}) && read(image) == prior,
                "png failure");
        fs::remove(home / "test.png.tmp");
        r["png"] = "complete length matched; previous image survived failed publication";
        constexpr uint64_t mib = 1024 * 1024;
        auto need = download_peak_bytes({{96 * mib, 96 * mib, 0}});
        require(need == 192 * mib && !check_download_space(need, 170 * mib).ok, "update space");
        r["space"] = "96 MiB update refused with 170 MiB free";
    });
}
inline StorageFolder drive() {
    auto drives = KnownFolders::RemovableDevices().GetFoldersAsync().get();
    require(drives.Size() == 1, "expected exactly one visible removable volume");
    return drives.GetAt(0);
}
inline StorageFolder model_root(StorageFolder disk) {
    return disk.CreateFolderAsync(L"xllama", CreationCollisionOption::OpenIfExists)
        .get()
        .CreateFolderAsync(L"models", CreationCollisionOption::OpenIfExists)
        .get();
}
inline void usb() {
    receipt("audit-usb.json", [](Json& r) {
        auto disk = drive();
        r["visible_volumes"] = 1;
        auto input = local().GetFolderAsync(L"usb-input").get();
        auto entry = Json::parse(read(root() / "usb-entry.json"));
        // Check available capacity of the selected removable unit before writing.
        auto keys = winrt::single_threaded_vector<winrt::hstring>({L"System.FreeSpace"});
        auto props = disk.Properties().RetrievePropertiesAsync(keys).get();
        auto free = props.TryLookup(L"System.FreeSpace");
        require(free && winrt::unbox_value<uint64_t>(free) > 600ull * 1024 * 1024,
                "USB free capacity unavailable or insufficient");
        auto models = model_root(disk);
        Json entries = Json::array();
        for (auto name : {good, bad}) {
            auto dst = models.CreateFolderAsync(name, CreationCollisionOption::FailIfExists).get();
            auto marker = dst.CreateFileAsync(L"review-owner.json").get();
            FileIO::WriteTextAsync(marker, signature).get();
            for (const auto& pin : entry.at("files")) {
                auto file = winrt::to_hstring(pin.at("filename").get<std::string>());
                if (std::wstring(name) == bad && file == L"model.onnx")
                    continue;
                input.GetFileAsync(file)
                    .get()
                    .CopyAsync(dst, file, NameCollisionOption::FailIfExists)
                    .get();
            }
            auto alias = entry;
            alias["name"] = winrt::to_string(winrt::hstring(name));
            alias["hf_base_url"] = "";
            entries.push_back(alias);
        }
        require(::xllama::replace_file_bytes(root() / "manifest.json",
                                             Json({{"models", entries}}).dump()),
                "fixture manifest");
        r["aliases"] = 2;
    });
}
inline void clean() {
    receipt("audit-clean.json", [](Json& r) {
        auto models = model_root(drive());
        for (auto name : {good, bad}) {
            auto item = models.TryGetItemAsync(name).get();
            if (!item)
                continue;
            auto dir = item.as<StorageFolder>();
            auto marker = dir.GetFileAsync(L"review-owner.json").get();
            require(FileIO::ReadTextAsync(marker).get() == signature, "fixture ownership mismatch");
            dir.DeleteAsync(StorageDeleteOption::PermanentDelete).get();
            require(!models.TryGetItemAsync(name).get(), "USB fixture still present");
        }
        r["owned_fixture_folders_remaining"] = 0;
    });
}
} // namespace audit_fixtures
