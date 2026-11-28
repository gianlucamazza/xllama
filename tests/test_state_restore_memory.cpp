// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
// Isolated process: count only payload-sized allocations on the restore caller's thread.
#include "xllama/chat_prompt.h"
#include "xllama/session.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <new>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace {
struct Payloads {
    std::size_t state, checkpoint, refuse;
    bool injected = false;
    std::size_t live = 0, peak = 0, checkpoint_allocations = 0;
    std::array<std::pair<void*, std::size_t>, 4> blocks{};
};
thread_local Payloads* observing = nullptr;

void expect(bool ok, const std::string& why) {
    if (!ok)
        throw std::runtime_error(why);
}
} // namespace

void* operator new(std::size_t bytes) {
    auto* sample = observing;
    const bool payload = sample && (bytes == sample->state || bytes == sample->checkpoint);
    if (payload && bytes == sample->refuse && !sample->injected) {
        sample->injected = true;
        throw std::bad_alloc();
    }
    void* ptr = std::malloc(bytes == 0 ? 1 : bytes);
    if (!ptr)
        throw std::bad_alloc();
    if (payload) {
        auto slot = std::find_if(sample->blocks.begin(), sample->blocks.end(),
                                 [](auto block) { return !block.first; });
        if (slot == sample->blocks.end()) {
            std::fputs("too many simultaneously live restore payloads\n", stderr);
            std::abort();
        }
        *slot = {ptr, bytes};
        sample->live += bytes;
        sample->peak = std::max(sample->peak, sample->live);
        sample->checkpoint_allocations += (bytes == sample->checkpoint);
    }
    return ptr;
}

void operator delete(void* ptr) noexcept {
    if (observing && ptr)
        for (auto& block : observing->blocks)
            if (block.first == ptr) {
                observing->live -= block.second;
                block = {};
                break;
            }
    std::free(ptr);
}

void operator delete(void* ptr, std::size_t) noexcept {
    ::operator delete(ptr);
}

static void check(const char* model, const char* path, bool chat) {
    xllama::SessionParams sp;
    sp.model_path = model;
    sp.n_threads = 1;
    sp.n_ctx = 256;
    std::string error;
    auto source = xllama::Session::create(sp, &error);
    expect(bool(source), error);
    xllama::GenerateParams prompt;
    prompt.prompt =
        "abcdefghijklmnopabcdefghijklmnopabcdefghijklmnopabcdefghijklmnopabcdefghijklmnop";
    if (chat) {
        prompt.prompt = xllama::chat_format_for(model).render_prompt(
            "You are a helpful assistant.", {}, "Count up to ten, using digits and commas.");
        prompt.prompt += "1, 2, 3, 4";
    }
    prompt.n_predict = 3;
    prompt.temperature = 0;
    expect(source->generate(prompt).success, "seed prefill failed");
    expect(source->save_state(path, &error), error);
    std::ifstream snapshot(path, std::ios::binary);
    std::string line, magic;
    std::size_t state_bytes = 0, token_count = 0, checkpoint_bytes = 0, prefix = 0;
    std::getline(snapshot, line);
    std::istringstream(line) >> magic >> state_bytes >> token_count;
    snapshot.seekg(static_cast<std::streamoff>(state_bytes + token_count * sizeof(int32_t)),
                   std::ios::cur);
    std::getline(snapshot, line);
    std::istringstream(line) >> magic >> prefix >> checkpoint_bytes;
    expect(magic == "xllama-ckpt" && prefix > 0 && checkpoint_bytes > 0 &&
               state_bytes > checkpoint_bytes,
           "expected a hybrid state and partial checkpoint");
    prompt.prompt.back() = chat ? '5' : 'a'; // Fork inside the saved head's slack.
    const auto expected = source->generate(prompt);
    expect(expected.success && !expected.output_text.empty() && expected.n_eval > 0 &&
               expected.n_p_eval < expected.n_prompt,
           "no resident prefix reuse");
    source.reset();
    {
        auto cold = xllama::Session::create(sp, &error);
        expect(bool(cold), error);
        const auto fresh = cold->generate(prompt);
        expect(fresh.success && fresh.n_eval == expected.n_eval &&
                   fresh.output_text == expected.output_text,
               "resident prefix differs from cold");
    }
    for (auto refuse : {std::size_t{0}, state_bytes, checkpoint_bytes}) {
        auto target = xllama::Session::create(sp, &error);
        expect(bool(target), error);
        target->load_state("",
                           &error); // Initialize the context before measuring serialized buffers.
        if (refuse == state_bytes)
            expect(target->generate(prompt).success, "failed to establish prior session state");
        const auto epoch = target->kv_epoch();
        Payloads sample{state_bytes, checkpoint_bytes, refuse};
        observing = &sample;
        bool loaded = false;
        try {
            loaded = target->load_state(path, &error);
        } catch (...) {
            observing = nullptr;
            throw;
        }
        observing = nullptr;
        expect(sample.injected == (refuse != 0), "injected allocation was not reached");
        if (refuse == 0) {
            std::printf("state_bytes=%zu checkpoint_bytes=%zu checkpoint_allocations=%zu "
                        "peak_live_payload_bytes=%zu\n",
                        state_bytes, checkpoint_bytes, sample.checkpoint_allocations, sample.peak);
            expect(sample.checkpoint_allocations == 1 && sample.peak == state_bytes,
                   "serialized state and checkpoint overlap or a checkpoint was copied");
        }
        if (refuse == state_bytes) {
            expect(!loaded && target->kv_epoch() == epoch,
                   "failed read changed the resident state");
            const auto prior = target->generate(prompt);
            expect(prior.success && prior.n_p_eval < prior.n_prompt &&
                       prior.output_text == expected.output_text,
                   "prior KV was not reusable");
            loaded = target->load_state(path, &error);
        }
        expect(loaded, error);
        const auto result = target->generate(prompt);
        expect(result.success && result.output_text == expected.output_text &&
                   result.n_eval == expected.n_eval,
               "restored greedy output differs");
        if (refuse != checkpoint_bytes)
            expect(result.n_p_eval == expected.n_p_eval, "restored head was not used");
        else
            expect(result.n_p_eval == result.n_prompt,
                   "failed optional checkpoint is still in use");
        if (refuse == 0) {
            std::atomic<bool> stop{false};
            auto interrupt = prompt;
            interrupt.abort_flag = &stop;
            interrupt.on_token = [&](std::string_view) { stop = true; };
            const auto cancelled = target->generate(interrupt);
            expect(cancelled.user_cancelled(stop), "decoded turn did not cancel");
            const auto retry = target->generate(prompt);
            expect(retry.success && retry.output_text == expected.output_text,
                   "a cancelled restored turn changed the next answer");
        }
    }
}

int main(int argc, char** argv) {
    if (argc != 3 && !(argc == 4 && std::string_view(argv[3]) == "--chat")) {
        std::fprintf(stderr, "usage: %s MODEL SNAPSHOT [--chat]\n", argv[0]);
        return 1;
    }
    try {
        check(argv[1], argv[2], argc == 4);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "hybrid state restore: %s\n", e.what());
        return 1;
    }
}
