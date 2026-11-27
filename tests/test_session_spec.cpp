// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include <doctest/doctest.h>

#include "xllama/session_hub.h"
#include "xllama/session_spec.h"

#include <cstdlib>
#include <filesystem>
#include <string>

using xllama::CatalogueLoad;
using xllama::kGgufNThreadsKnobAbsent;
using xllama::resolve_gguf_n_threads;
using xllama::resolve_session_params;
using xllama::same_load;

TEST_CASE("resolve_session_params: one resolver for every surface (ADR 0007)") {
    // Not in the catalogue: defaults, backend left to the loader.
    const auto unknown = resolve_session_params("custom", CatalogueLoad{}, "D:\\m\\custom", -1);
    CHECK(unknown.model_path == "custom");
    CHECK(unknown.n_ctx == xllama::kDefaultNCtx);
    CHECK(unknown.n_threads == 0);
    CHECK(unknown.backend == xllama::Backend::Auto);
    CHECK(unknown.lora_path.empty());

    // An ORT entry: catalogue context, no llama.cpp fields.
    CatalogueLoad ort;
    ort.found = true;
    ort.n_ctx = 3072;
    ort.lora = "ignored.gguf";
    const auto o = resolve_session_params("smollm2", ort, "D:\\m\\smollm2", 4);
    CHECK(o.n_ctx == 3072);
    CHECK(o.n_gpu_layers == 0);
    CHECK(o.n_threads == 0);
    CHECK(o.lora_path.empty());

    // A GGUF entry with an adapter: the GUI and the API used to disagree here (F03).
    CatalogueLoad g;
    g.found = true;
    g.gguf = true;
    g.n_ctx = 1; // clamped like every session open
    g.gpu_layers = 32;
    g.lora = "adapter.gguf";
    g.lora_scale = 0.5f;
    const auto dir = std::filesystem::path("models/smollm2-lora-rt").make_preferred();
    const auto a = resolve_session_params("smollm2-lora-rt", g, dir.string(), -1);
    CHECK(a.backend == xllama::Backend::LlamaCpp);
    CHECK(a.n_ctx == xllama::kMinSessionNCtx);
    CHECK(a.n_gpu_layers == 32);
    CHECK(a.lora_path == (dir / "adapter.gguf").string());
    CHECK(a.lora_scale == 0.5f);
    CHECK(a.n_threads == 0);
    // Trailing separators, a bare relative directory and the empty directory.
    CHECK(resolve_session_params("x", g, (dir / "").string(), 0).lora_path == a.lora_path);
    CHECK(resolve_session_params("x", g, "models/", 0).lora_path == "models/adapter.gguf");
    CHECK(resolve_session_params("x", g, "models", 0).lora_path ==
          (std::filesystem::path("models") / "adapter.gguf").string());
    CHECK(resolve_session_params("x", g, "", 0).lora_path == "adapter.gguf");
    // The operator knob wins, 0 included.
    CHECK(resolve_session_params("x", g, "D:\\m\\x\\", 0).n_gpu_layers == 0);
    CHECK(resolve_session_params("x", g, "D:\\m\\x\\", 0).n_threads == 0);
    CHECK(resolve_session_params("x", g, "D:\\m\\x\\", 0, 4).n_threads == 4);
}

TEST_CASE("resolve_gguf_n_threads: absent, 0, or invalid keep auto") {
    CHECK(resolve_gguf_n_threads(kGgufNThreadsKnobAbsent) == 0);
    CHECK(resolve_gguf_n_threads(4) == 4);
    CHECK(resolve_gguf_n_threads(1) == 1);
    CHECK(resolve_gguf_n_threads(0) == 0);
    CHECK(resolve_gguf_n_threads(-5) == 0);
    // Above ggml's thread-array size the CPU backend overruns its fixed arrays.
    CHECK(resolve_gguf_n_threads(xllama::kGgufNThreadsMax) == xllama::kGgufNThreadsMax);
    CHECK(resolve_gguf_n_threads(xllama::kGgufNThreadsMax + 1) == 0);
    CHECK(resolve_gguf_n_threads(100000) == 0);
    CatalogueLoad g;
    g.found = true;
    g.gguf = true;
    CHECK(resolve_session_params("m", g, "D:\\m", -1).n_threads == 0);
    CHECK(resolve_session_params("m", g, "D:\\m", -1, kGgufNThreadsKnobAbsent).n_threads == 0);
    CHECK(resolve_session_params("m", g, "D:\\m", -1, 6).n_threads == 6);
    CHECK(resolve_session_params("m", g, "D:\\m", -1, 0).n_threads == 0);
    // GPU-layer resolution is unchanged when a thread knob is present.
    g.gpu_layers = 32;
    CHECK(resolve_session_params("m", g, "D:\\m", -1, 4).n_gpu_layers == 32);
    CHECK(resolve_session_params("m", g, "D:\\m", 0, 4).n_gpu_layers == 0);
}

TEST_CASE("same_load: an adapter, context or GPU split is a different model in RAM") {
    CatalogueLoad g;
    g.found = true;
    g.gguf = true;
    g.lora = "adapter.gguf";
    const auto base = resolve_session_params("m", g, "D:\\m", -1);
    auto other = base;
    CHECK(same_load(base, other));
    other.n_threads = 6; // a tuning knob, not the load
    other.n_batch = 256;
    CHECK(same_load(base, other));
    CHECK(base.prefix_snapshots);
    auto no_snap = base;
    no_snap.prefix_snapshots = false;
    CHECK_FALSE(same_load(base, no_snap));
    for (int field = 0; field < 5; ++field) {
        auto changed = base;
        switch (field) {
        case 0:
            changed.lora_path.clear();
            break;
        case 1:
            changed.lora_scale = 0.25f;
            break;
        case 2:
            changed.n_ctx = 1024;
            break;
        case 3:
            changed.n_gpu_layers = 8;
            break;
        case 4:
            changed.model_path = "other";
            break;
        }
        CHECK_MESSAGE(!same_load(base, changed), field);
    }
}

TEST_CASE("SessionHub reuses only the same load (opt-in: XLLAMA_TEST_MODEL)") {
    const char* model_env = std::getenv("XLLAMA_TEST_MODEL");
    if (!model_env)
        return;
    xllama::SessionHub hub;
    std::lock_guard<std::mutex> lock(hub.mtx);
    xllama::SessionParams sp;
    sp.model_path = model_env;
    sp.n_ctx = 512;
    std::string err;
    xllama::Session* first = hub.ensure_locked("m", sp, &err);
    REQUIRE_MESSAGE(first, err);
    const uint64_t gen = hub.generation;
    CHECK(hub.ensure_locked("m", sp, &err) == first);
    CHECK(hub.generation == gen);
    sp.n_ctx = 1024; // the embedding num_ctx case: a different load
    REQUIRE(hub.ensure_locked("m", sp, &err) != nullptr);
    CHECK(hub.generation > gen);
    CHECK(hub.session->context_length() == 1024);
    const uint64_t gen2 = hub.generation;
    hub.invalidate_locked("other"); // another model's files changed: no effect
    CHECK(hub.generation == gen2);
    hub.invalidate_locked("m"); // this model's files changed: stale even if equal
    CHECK(hub.session == nullptr);
}
