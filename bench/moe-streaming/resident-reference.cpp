// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include "chat.h"
#include "common.h"
#include "ggml-backend.h"
#include "xllama/llama_raii.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {
namespace fs = std::filesystem;
using TokenIds = std::vector<llama_token>;

constexpr int kContext = 512;
constexpr int kMaxPromptTokens = 128;
constexpr int kContinuationTokens = 9;
constexpr int kExportSteps[] = {1, 2, 3, 5, 9};

void write_bytes(const fs::path& path, const char* data, std::streamsize size) {
    std::ofstream file;
    file.exceptions(std::ios::failbit | std::ios::badbit);
    file.open(path, std::ios::binary);
    file.write(data, size);
    file.close();
}

void write_ids(std::ostream& out, const TokenIds& ids) {
    out << '[';
    for (std::size_t i = 0; i < ids.size(); ++i) {
        if (i > 0)
            out << ", ";
        out << ids[i];
    }
    out << ']';
}

const float* decode(llama_context* ctx, llama_token* ids, int count, int vocab_size) {
    if (llama_decode(ctx, llama_batch_get_one(ids, count)) != 0)
        throw std::runtime_error("resident llama_decode failed");
    const float* row = llama_get_logits_ith(ctx, -1);
    if (!row ||
        !std::all_of(row, row + vocab_size, [](float value) { return std::isfinite(value); }))
        throw std::runtime_error("missing or nonfinite logit row");
    return row;
}

void run_case(llama_model* model, const common_chat_templates* templates, const fs::path& prompt,
              const fs::path& output) {
    std::ifstream input(prompt, std::ios::binary);
    if (!input)
        throw std::runtime_error("cannot open prompt: " + prompt.string());
    common_chat_templates_inputs chat;
    common_chat_msg message;
    message.role = "user";
    message.content.assign(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
    chat.messages.push_back(std::move(message));
    const std::string rendered = common_chat_templates_apply(templates, chat).prompt;
    const auto* vocab = llama_model_get_vocab(model);
    auto tokens = common_tokenize(vocab, rendered, true, true);
    const llama_token bos = llama_vocab_bos(vocab);
    if (tokens.empty() || tokens.size() > kMaxPromptTokens || tokens.front() != bos ||
        std::count(tokens.begin(), tokens.end(), bos) != 1)
        throw std::runtime_error("expected 1 to 128 prompt tokens and exactly one leading BOS");

    auto params = llama_context_default_params();
    params.n_ctx = kContext;
    params.n_batch = kContext;
    params.n_ubatch = kContext;
    params.n_threads = 1;
    params.n_threads_batch = 1;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    params.offload_kqv = false;
    xllama::LlamaContextPtr ctx(llama_init_from_model(model, params));
    if (!ctx)
        throw std::runtime_error("cannot create resident context");

    fs::create_directory(output);
    const int vocab_size = llama_vocab_n_tokens(vocab);
    const auto row_bytes = static_cast<std::streamsize>(vocab_size) * sizeof(float);
    const auto save_row = [&](int step, const float* row) {
        write_bytes(output / ("row-" + std::to_string(step) + ".bin"),
                    reinterpret_cast<const char*>(row), row_bytes);
    };
    const float* row =
        decode(ctx.get(), tokens.data(), static_cast<int>(tokens.size()), vocab_size);
    save_row(0, row);
    TokenIds continuation;
    for (int step = 1; step <= kContinuationTokens; ++step) {
        llama_token token = static_cast<llama_token>(std::max_element(row, row + vocab_size) - row);
        continuation.push_back(token);
        row = decode(ctx.get(), &token, 1, vocab_size);
        if (std::find(std::begin(kExportSteps), std::end(kExportSteps), step) !=
            std::end(kExportSteps))
            save_row(step, row);
    }
    write_bytes(output / "rendered.txt", rendered.data(), rendered.size());
    std::ofstream metadata;
    metadata.exceptions(std::ios::failbit | std::ios::badbit);
    metadata.open(output / "tokens.json");
    metadata << "{\n  \"vocab_size\": " << vocab_size << ",\n  \"bos_token_id\": " << bos
             << ",\n  \"prompt_token_ids\": ";
    write_ids(metadata, tokens);
    metadata << ",\n  \"teacher_forced_token_ids\": ";
    write_ids(metadata, continuation);
    metadata << "\n}\n";
    metadata.close();
}
} // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "usage: resident-reference MODEL.gguf NEW_OUTPUT_DIR PROMPT.txt...\n";
        return 2;
    }
    try {
        const fs::path output = argv[2];
        if (fs::exists(output))
            throw std::runtime_error("output directory already exists");
        const char* disable_fusion = std::getenv("GGML_CPU_DISABLE_FUSION");
        if (!disable_fusion || std::string_view(disable_fusion) != "1")
            throw std::runtime_error("set GGML_CPU_DISABLE_FUSION=1 before launching");
        ggml_backend_load_all();
        auto params = llama_model_default_params();
        ggml_backend_dev_t devices[] = {nullptr};
        params.devices = devices;
        params.n_gpu_layers = 0;
        params.load_mode = LLAMA_LOAD_MODE_NONE;
        params.lazy_mode = LLAMA_LAZY_MODE_OFF;
        params.use_extra_bufts = false;
        xllama::LlamaModelPtr model(llama_model_load_from_file(argv[1], params));
        if (!model)
            throw std::runtime_error("cannot load resident GGUF");
        auto templates = common_chat_templates_init(model.get(), "");
        const char* model_template = llama_model_chat_template(model.get(), nullptr);
        if (!model_template || !common_chat_templates_was_explicit(templates.get()))
            throw std::runtime_error("checkpoint has no explicit chat template");
        fs::create_directories(output);
        const std::string_view template_bytes(model_template);
        write_bytes(output / "chat-template.jinja", template_bytes.data(), template_bytes.size());
        for (int i = 3; i < argc; ++i) {
            const fs::path prompt = argv[i];
            const fs::path case_output = output / prompt.stem();
            if (fs::exists(case_output))
                throw std::runtime_error("duplicate prompt stem: " + prompt.stem().string());
            run_case(model.get(), templates.get(), prompt, case_output);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "resident reference: " << error.what() << '\n';
        return 1;
    }
}
