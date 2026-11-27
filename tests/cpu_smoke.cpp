// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
#include "../src/bridge/decode_loop.h"
#include "xllama/inference.h"
#include "xllama/llama_raii.h"
#include "xllama/session.h"

#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>

using nlohmann::json;

static json capture(const char* path) {
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    xllama::LlamaModelPtr model(llama_model_load_from_file(path, mp));
    if (!model)
        throw std::runtime_error("model load failed");
    auto cp = llama_context_default_params();
    cp.n_ctx = 128;
    cp.n_threads = cp.n_threads_batch = 1;
    cp.n_batch = cp.n_ubatch = 32;
    xllama::LlamaContextPtr ctx(llama_init_from_model(model.get(), cp));
    if (!ctx)
        throw std::runtime_error("context init failed");
    const auto* vocab = llama_model_get_vocab(model.get());
    const auto width = llama_vocab_n_tokens(vocab);
    std::vector<llama_token> prompt(16);
    const int n = llama_tokenize(vocab, "abcd", 4, prompt.data(), prompt.size(), true, true);
    if (n <= 0)
        throw std::runtime_error("tokenization failed");
    prompt.resize(n);
    if (!xllama::prefill_chunked(ctx.get(), prompt.data(), n))
        throw std::runtime_error("prefill failed");
    json result = {{"prompt_ids", prompt}};
    auto record = [&]() {
        const float* row = llama_get_logits_ith(ctx.get(), -1);
        if (!row)
            throw std::runtime_error("missing logit row");
        result["positions"].push_back(llama_memory_seq_pos_max(llama_get_memory(ctx.get()), 0));
        result["rows"].push_back(std::vector<float>(row, row + width));
    };
    record();
    xllama::LlamaSamplerPtr sampler(llama_sampler_init_greedy());
    xllama::DecodeLoopParams p;
    p.ctx = ctx.get();
    p.sampler = sampler.get();
    p.vocab = vocab;
    p.n_predict = 4;
    p.on_accepted = [&](llama_token id) {
        result["generated_ids"].push_back(id);
        record();
    };
    std::string text;
    if (xllama::decode_loop(p, text).n_generated != p.n_predict)
        throw std::runtime_error("short decode");
    return result;
}

// The fixture repeats n-grams and exercises both accepted and rejected drafts.
// Extending the prompt buffer during decode must preserve prefill accounting.
static void check_prompt_lookup(const char* path) {
    xllama::InferenceParams p;
    p.model_path = path;
    p.prompt = "abcdabcdabcd";
    p.n_ctx = 128;
    p.n_threads = 1;
    p.n_predict = 64;
    p.greedy = true;
    const auto plain = xllama::run_inference(p);
    p.prompt_lookup = true;
    const auto lookup = xllama::run_inference(p);
    if (!plain.success || !lookup.success || plain.n_eval != p.n_predict ||
        lookup.n_eval != plain.n_eval || lookup.output_text != plain.output_text ||
        plain.n_p_eval != 13 || lookup.n_p_eval != plain.n_p_eval ||
        lookup.n_prompt != plain.n_prompt || lookup.n_spec_accepted <= 0 ||
        lookup.n_drafted <= lookup.n_spec_accepted)
        throw std::runtime_error(
            "prompt lookup changed generation, prompt counts or draft coverage");
}

static json turns(const char* path) {
    xllama::SessionParams sp;
    sp.model_path = path;
    sp.backend = xllama::Backend::LlamaCpp;
    sp.n_ctx = 128;
    sp.n_threads = 1;
    sp.n_batch = sp.n_ubatch = 32;
    std::string error;
    auto session = xllama::Session::create(sp, &error);
    if (!session)
        throw std::runtime_error(error);
    xllama::GenerateParams p;
    p.n_predict = 4;
    p.temperature = 0;
    p.reuse_kv = true;
    auto turn = [&](const char* prompt, bool reset) -> json {
        p.prompt = prompt;
        p.reset_kv = reset;
        p.expected_kv_epoch = session->kv_epoch();
        const auto r = session->generate(p);
        if (!r.success || r.n_eval != p.n_predict || r.gpu_layers != 0)
            throw std::runtime_error("session generation failed: " + r.error_msg);
        return {{"text", r.output_text}, {"n_prompt", r.n_prompt}, {"n_p_eval", r.n_p_eval}};
    };
    json result;
    result.push_back(turn("abcd", true));
    result.push_back(turn("mf", false));
    result.push_back(turn("mf", true));
    return result;
}

int main(int argc, char** argv) {
    if (argc != 3)
        return 2;
    try {
        auto result = capture(argv[1]);
        result["turns"] = turns(argv[1]);
        check_prompt_lookup(argv[1]);
        std::ofstream stream(argv[2]);
        stream.exceptions(std::ios::failbit | std::ios::badbit);
        stream << result.dump(2) << '\n';
        stream.close();
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
