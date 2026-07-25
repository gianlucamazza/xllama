// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include "xllama/path_utils.h"
#include "xllama/session.h"
#include "xllama/session_hub.h"

#include <doctest/doctest.h>
#include <filesystem>
#include <fstream>

TEST_CASE("SessionHub: failed create leaves the hub empty and bumps generation") {
    auto& hub = xllama::session_hub();
    std::lock_guard<std::mutex> lk(hub.mtx);
    const uint64_t g0 = hub.generation;
    xllama::SessionParams sp;
    sp.model_path = "/nonexistent/hub-model.gguf";
    std::string err;
    CHECK(hub.ensure_locked("hub-model", sp, &err) == nullptr);
    CHECK(hub.session == nullptr);
    CHECK(hub.model.empty());
    // The attempted swap must invalidate any holder's generation even though
    // creation failed — its old session pointer is gone either way.
    CHECK(hub.generation == g0 + 1);
    hub.reset_locked(); // no-op when already empty
    CHECK(hub.generation == g0 + 1);
}

TEST_CASE("resolve_max_length: the #130 ladder has one home") {
    using xllama::resolve_max_length;
    // override < 0: saturate to n_ctx — what Session always requests.
    CHECK(resolve_max_length(2048, 1289, 256, -1) == 2048);
    CHECK(resolve_max_length(3072, 10, 96, -1) == 3072);
    // override == 0: derive min(n_ctx, prompt + n_predict) — bench default,
    // keeps every historical row's meaning.
    CHECK(resolve_max_length(2048, 1289, 256, 0) == 1545); // the shipping default
                                                           // that landed in the valley
    CHECK(resolve_max_length(2048, 1800, 512, 0) == 2048); // clamped to context
    // override > 0: explicit, clamped to (n_prompt, n_ctx].
    CHECK(resolve_max_length(2048, 1289, 256, 1801) == 1801);
    CHECK(resolve_max_length(2048, 1289, 256, 100) == 1290);  // floor: prompt+1
    CHECK(resolve_max_length(2048, 1289, 256, 9999) == 2048); // ceiling: n_ctx
}

TEST_CASE("Session::create rejects non-existent model path") {
    xllama::SessionParams sp;
    sp.model_path = "/nonexistent/path/model.gguf";
    std::string err;
    auto s = xllama::Session::create(sp, &err);
    CHECK(s == nullptr);
    CHECK(!err.empty());
}

TEST_CASE("Session::create rejects empty model path") {
    xllama::SessionParams sp;
    sp.model_path = "";
    std::string err;
    auto s = xllama::Session::create(sp, &err);
    CHECK(s == nullptr);
    CHECK(!err.empty());
}

TEST_CASE("model_uses_llama_backend detects .gguf suffix and dir layout") {
    using xllama::model_uses_llama_backend;

    CHECK(model_uses_llama_backend("foo.gguf"));
    CHECK(model_uses_llama_backend("/abs/path/model.Q4_K.gguf"));
    CHECK_FALSE(model_uses_llama_backend("smollm2-360m"));
    CHECK_FALSE(model_uses_llama_backend("some-ort-model-dir"));

    // Temp dir containing a .gguf file (simulates catalogue layout on disk)
    auto tmp = std::filesystem::temp_directory_path() / "xllama-gguf-layout-test";
    std::filesystem::create_directories(tmp);
    {
        std::ofstream f((tmp / "model.gguf").string());
        f << "dummy";
    }
    CHECK(model_uses_llama_backend(tmp.string()));

    // Dir without .gguf should be false
    auto tmp2 = std::filesystem::temp_directory_path() / "xllama-no-gguf";
    std::filesystem::create_directories(tmp2);
    CHECK_FALSE(model_uses_llama_backend(tmp2.string()));

    // Cleanup
    std::error_code ec;
    std::filesystem::remove_all(tmp, ec);
    std::filesystem::remove_all(tmp2, ec);
}

TEST_CASE("first_gguf_in_dir prefers model.gguf over adapter.gguf") {
    using xllama::first_gguf_in_dir;
    auto tmp = std::filesystem::temp_directory_path() / "xllama-gguf-prefer-test";
    std::error_code ec;
    std::filesystem::remove_all(tmp, ec);
    std::filesystem::create_directories(tmp);
    {
        std::ofstream a((tmp / "adapter.gguf").string());
        a << "adapter";
        std::ofstream m((tmp / "model.gguf").string());
        m << "model";
    }
    const std::string picked = first_gguf_in_dir(tmp.string(), "adapter.gguf");
    CHECK(picked.find("model.gguf") != std::string::npos);
    // Without exclude, model.gguf still preferred by name
    const std::string picked2 = first_gguf_in_dir(tmp.string());
    CHECK(picked2.find("model.gguf") != std::string::npos);
    std::filesystem::remove_all(tmp, ec);
}

TEST_CASE("Session::create fails when LoRA path is invalid (llama)") {
    // Real base not required: create fails at model load first if path is junk.
    // Here we only assert empty model + lora still rejects cleanly.
    xllama::SessionParams sp;
    sp.model_path = "/nonexistent/base.gguf";
    sp.lora_path = "/nonexistent/adapter.gguf";
    sp.backend = xllama::Backend::LlamaCpp;
    std::string err;
    auto s = xllama::Session::create(sp, &err);
    CHECK(s == nullptr);
    CHECK(!err.empty());
}

TEST_CASE("Session::create Auto with explicit Backend::LlamaCpp on GGUF layout") {
    // Even without real weights the dispatch must select the llama path.
    // We only assert that it does not succeed via the ORT path and that an error is produced.
    auto tmp = std::filesystem::temp_directory_path() / "xllama-session-gguf-auto";
    std::filesystem::create_directories(tmp);
    {
        std::ofstream f((tmp / "mini.gguf").string());
        f << "not-a-real-gguf";
    }

    xllama::SessionParams sp;
    sp.model_path = tmp.string();
    sp.backend = xllama::Backend::Auto; // should resolve to Llama via layout

    std::string err;
    auto s = xllama::Session::create(sp, &err);
    // On Linux this build only has llama; we expect a load failure from the llama code path.
    CHECK(s == nullptr);
    CHECK(!err.empty());

    std::error_code ec;
    std::filesystem::remove_all(tmp, ec);
}
