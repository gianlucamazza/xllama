// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
// Lane B engine — see include/xllama/device_train.h for the contract and
// docs/training-architecture.md §Lane B for the design + memory budget.

#include "xllama/device_train.h"

#include "xllama/chat_prompt.h"
#include "xllama/llama_raii.h"
#include "xllama/platform.h"
#include "xllama/session.h"
#include "xllama/training.h"

#include "ggml-opt.h"
#include "ggml.h"
#include "gguf.h"
#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <vector>

namespace xllama {
namespace {

#ifdef XLLAMA_DEVICE_TRAIN
void status(const DeviceTrainCallbacks& cb, const std::string& line) {
    if (cb.on_status)
        cb.on_status(line);
}
#endif

#ifdef XLLAMA_DEVICE_TRAIN
std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
        case '"':
            out += "\\\"";
            break;
        case '\\':
            out += "\\\\";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\r':
            out += "\\r";
            break;
        case '\t':
            out += "\\t";
            break;
        default:
            out.push_back(c);
            break;
        }
    }
    return out;
}
#endif

// --- minimal JSONL chat-sample extraction ({"label","messages":[{role,content}]}) ---

bool jsonl_extract_string(const std::string& line, const std::string& key, size_t from,
                          std::string& out, size_t* end_pos = nullptr) {
    const std::string needle = "\"" + key + "\"";
    size_t k = line.find(needle, from);
    if (k == std::string::npos)
        return false;
    size_t colon = line.find(':', k + needle.size());
    if (colon == std::string::npos)
        return false;
    size_t q = line.find('"', colon + 1);
    if (q == std::string::npos)
        return false;
    out.clear();
    size_t i = q + 1;
    while (i < line.size()) {
        char c = line[i++];
        if (c == '"')
            break;
        if (c == '\\' && i < line.size()) {
            char e = line[i++];
            switch (e) {
            case 'n':
                out.push_back('\n');
                break;
            case 't':
                out.push_back('\t');
                break;
            case 'r':
                out.push_back('\r');
                break;
            case 'u':
                // Keep it simple: preserve the escape verbatim (rare in our data).
                out += "\\u";
                break;
            default:
                out.push_back(e);
                break;
            }
        } else {
            out.push_back(c);
        }
    }
    if (end_pos)
        *end_pos = i;
    return true;
}

struct ChatSample {
    std::string label;               // empty when the line has no label (toy datasets)
    std::string preferred_assistant; // corrected target for label=correction
    std::vector<std::pair<std::string, std::string>> messages; // role, content
};

bool parse_chat_sample_line(const std::string& line, ChatSample& out) {
    out = ChatSample{};
    jsonl_extract_string(line, "label", 0, out.label);
    jsonl_extract_string(line, "preferred_assistant", 0, out.preferred_assistant);
    size_t pos = line.find("\"messages\"");
    if (pos == std::string::npos)
        return false;
    for (;;) {
        std::string role;
        size_t after_role = 0;
        if (!jsonl_extract_string(line, "role", pos, role, &after_role))
            break;
        std::string content;
        size_t after_content = 0;
        if (!jsonl_extract_string(line, "content", after_role, content, &after_content))
            break;
        out.messages.emplace_back(role, content);
        pos = after_content;
    }
    return !out.messages.empty();
}

// Render one chat sample into a training document with the model's template:
// completed history turns + final user turn + assistant generation header +
// the target assistant text + the closing delimiter.
std::string render_sample(const ChatFormat& fmt, const ChatSample& sample) {
    std::vector<ChatTurn> history;
    std::string pending_user;
    std::string final_user;
    std::string final_assistant;
    for (const auto& [role, content] : sample.messages) {
        if (role == "user") {
            pending_user = content;
        } else if (role == "assistant" && !pending_user.empty()) {
            history.push_back(ChatTurn{pending_user, content});
            pending_user.clear();
        }
    }
    if (history.empty())
        return {};
    final_user = history.back().user;
    final_assistant = history.back().assistant;
    if (sample.label == "correction" && !sample.preferred_assistant.empty())
        final_assistant = sample.preferred_assistant;
    history.pop_back();
    // Same default system prompt as the chat UI / API endpoint.
    std::string doc = fmt.render_prompt("You are a helpful AI assistant.", history, final_user);
    doc += final_assistant;
    doc += fmt.turn_close;
    return doc;
}

#ifdef XLLAMA_DEVICE_TRAIN

// --- param filter trampoline (llama_opt_param_filter has userdata) ---

bool param_filter_cb(const struct ggml_tensor* tensor, void* userdata) {
    const auto* patterns = static_cast<const std::vector<std::string>*>(userdata);
    return device_train_tensor_matches(ggml_get_name(tensor), *patterns);
}

// --- constant-LR optimizer params ---

struct OptPars {
    float lr = 2e-4f;
};

ggml_opt_optimizer_params opt_pars_cb(void* userdata) {
    ggml_opt_optimizer_params p = ggml_opt_get_default_optimizer_params(nullptr);
    const auto* op = static_cast<const OptPars*>(userdata);
    p.adamw.alpha = op->lr;
    p.sgd.alpha = op->lr;
    return p;
}

// --- epoch progress trampoline ---
// ggml_opt_epoch_callback carries no userdata, so the active run parks its
// callback state here. run_device_train_job is documented not concurrency-safe.

struct ProgressState {
    const DeviceTrainCallbacks* cb = nullptr;
    int epoch = 0;
    int epochs = 0;
};

ProgressState g_progress; // NOLINT: single active run by contract

void epoch_progress_cb(bool train, ggml_opt_context_t /*opt_ctx*/, ggml_opt_dataset_t /*dataset*/,
                       ggml_opt_result_t result, int64_t ibatch, int64_t ibatch_max,
                       int64_t /*t_start_us*/) {
    if (!train || !g_progress.cb || !g_progress.cb->on_progress)
        return;
    DeviceTrainProgress p;
    p.stage = TrainStage::Train;
    p.epoch = g_progress.epoch;
    p.epochs = g_progress.epochs;
    p.ibatch = ibatch;
    p.ibatch_max = ibatch_max;
    double unc = 0.0;
    ggml_opt_result_loss(result, &p.loss, &unc);
    g_progress.cb->on_progress(p);
}

// --- prepare stage: selective f32 upcast of the base GGUF ---
// llama_opt only trains GGML_TYPE_F32 tensors (llama_set_param skips the
// rest), so every tensor selected by param_filter must be stored as f32 in
// the GGUF the train loop loads. Everything else keeps its original type
// (frozen forward works on f16/quantized weights).

bool prepare_mixed_gguf(const std::string& base_gguf, const std::vector<std::string>& patterns,
                        const std::string& out_path, const DeviceTrainCallbacks& cb,
                        std::string* err) {
    ggml_context* data_ctx = nullptr;
    gguf_init_params iparams{/*no_alloc=*/false, /*ctx=*/&data_ctx};
    gguf_context* src = gguf_init_from_file(base_gguf.c_str(), iparams);
    if (!src || !data_ctx) {
        if (err)
            *err = "prepare: cannot read base GGUF: " + base_gguf;
        if (src)
            gguf_free(src);
        return false;
    }

    gguf_context* dst = gguf_init_empty();
    gguf_set_kv(dst, src); // carry all metadata (arch, hparams, tokenizer, …)

    // Fail fast on filters the current llama.cpp pin cannot train (see
    // device_train_unsupported_reason) instead of aborting inside ggml.
    int last_block = -1;
    {
        const int64_t arch_key = gguf_find_key(src, "general.architecture");
        if (arch_key >= 0) {
            const std::string arch = gguf_get_val_str(src, arch_key);
            const int64_t bc_key = gguf_find_key(src, (arch + ".block_count").c_str());
            if (bc_key >= 0)
                last_block = static_cast<int>(gguf_get_val_u32(src, bc_key)) - 1;
        }
    }
    if (last_block < 0) {
        if (err)
            *err = "prepare: cannot determine the model's last transformer block";
        ggml_free(data_ctx);
        gguf_free(src);
        gguf_free(dst);
        return false;
    }

    std::vector<std::unique_ptr<float[]>> upcast_buffers;
    // Meta-only context for the f32 twins; data points into upcast_buffers.
    const size_t n_tensors = static_cast<size_t>(gguf_get_n_tensors(src));
    ggml_init_params mparams{/*mem_size=*/(n_tensors + 2) * ggml_tensor_overhead(),
                             /*mem_buffer=*/nullptr, /*no_alloc=*/true};
    ggml_context* meta_ctx = ggml_init(mparams);

    int64_t upcast_count = 0;
    int64_t trainable_elems = 0;
    // Iterate the GGUF tensor index, not data_ctx: gguf_init_from_file also
    // creates a raw data-blob tensor in the context that is not a weight.
    for (int64_t i = 0; i < gguf_get_n_tensors(src); ++i) {
        const char* name = gguf_get_tensor_name(src, i);
        ggml_tensor* t = ggml_get_tensor(data_ctx, name);
        if (!t)
            continue;
        const bool trainable = device_train_tensor_matches(name, patterns);
        if (trainable && last_block >= 0) {
            const std::string reason = device_train_unsupported_reason(name, last_block);
            if (!reason.empty()) {
                if (err)
                    *err = std::string("prepare: param_filter selects ") + name + ": " + reason;
                ggml_free(meta_ctx);
                ggml_free(data_ctx);
                gguf_free(src);
                gguf_free(dst);
                return false;
            }
        }
        if (!trainable || t->type == GGML_TYPE_F32) {
            if (trainable)
                trainable_elems += ggml_nelements(t);
            gguf_add_tensor(dst, t);
            continue;
        }
        const ggml_type_traits* traits = ggml_get_type_traits(t->type);
        if (!traits || !traits->to_float) {
            if (err)
                *err = std::string("prepare: no f32 conversion for tensor type of ") + name;
            ggml_free(meta_ctx);
            ggml_free(data_ctx);
            gguf_free(src);
            gguf_free(dst);
            return false;
        }
        const int64_t ne = ggml_nelements(t);
        upcast_buffers.emplace_back(new float[ne]);
        float* buf = upcast_buffers.back().get();
        traits->to_float(t->data, buf, ne);

        ggml_tensor* f32 = ggml_new_tensor(meta_ctx, GGML_TYPE_F32, ggml_n_dims(t), t->ne);
        ggml_set_name(f32, name);
        f32->data = buf;
        gguf_add_tensor(dst, f32);
        ++upcast_count;
        trainable_elems += ne;
    }

    bool ok = true;
    if (trainable_elems == 0) {
        if (err)
            *err = "prepare: param_filter matches no tensors in the base GGUF";
        ok = false;
    } else if (!gguf_write_to_file(dst, out_path.c_str(), /*only_meta=*/false)) {
        if (err)
            *err = "prepare: cannot write " + out_path;
        ok = false;
    } else {
        std::ostringstream os;
        os << "prepare: " << out_path << " (" << upcast_count << " tensors upcast to f32, "
           << trainable_elems << " trainable elements)";
        status(cb, os.str());
    }

    ggml_free(meta_ctx);
    ggml_free(data_ctx);
    gguf_free(src);
    gguf_free(dst);
    return ok;
}

std::string resolve_base_gguf(const std::string& base_model, std::string* err) {
    namespace fs = std::filesystem;
    std::error_code ec;
    if (fs::is_regular_file(base_model, ec))
        return base_model;
    if (fs::is_directory(base_model, ec)) {
        for (const auto& e : fs::directory_iterator(base_model, ec)) {
            if (e.path().extension() == ".gguf")
                return e.path().string();
        }
    }
    if (err)
        *err = "base_model is not a GGUF file or a directory containing one: " + base_model;
    return {};
}

void write_result_json(const std::string& out_dir, const TrainingResult& r) {
    std::ofstream out(out_dir + "/result.json");
    if (!out)
        return;
    out << "{\n  \"success\": " << (r.success ? "true" : "false") << ",\n  \"stages\": [";
    for (size_t i = 0; i < r.stages_completed.size(); ++i) {
        out << (i ? ", " : "") << "\"" << training_stage_name(r.stages_completed[i]) << "\"";
    }
    out << "],\n  \"merged_gguf\": \"" << json_escape(r.merged_gguf_path) << "\",\n";
    out << "  \"last_loss\": " << r.last_loss << ",\n";
    out << "  \"wall_seconds\": " << r.wall_seconds << ",\n";
    out << "  \"peak_ws_mb\": " << r.peak_ws_mb << ",\n";
    out << "  \"error\": \"" << json_escape(r.error_msg) << "\"\n}\n";
}

#endif // XLLAMA_DEVICE_TRAIN

} // namespace

bool device_train_tensor_matches(const std::string& tensor_name,
                                 const std::vector<std::string>& patterns) {
    for (const auto& p : patterns) {
        if (!p.empty() && tensor_name.find(p) != std::string::npos)
            return true;
    }
    return false;
}

std::string device_train_unsupported_reason(const std::string& tensor_name, int last_block) {
    if (tensor_name == "token_embd.weight" || tensor_name == "rope_freqs.weight")
        return "tensor is excluded by llama_set_param and would not be optimized";
    if (tensor_name.rfind("blk.", 0) != 0) {
        if (tensor_name == "output.weight" || tensor_name == "output_norm.weight")
            return {};
        return "only output/output_norm are supported outside the last transformer block";
    }
    const size_t dot = tensor_name.find('.', 4);
    if (dot == std::string::npos)
        return "unrecognized block tensor name";
    int block = -1;
    try {
        block = std::stoi(tensor_name.substr(4, dot - 4));
    } catch (...) {
        return "unrecognized block tensor name";
    }
    if (block != last_block) {
        return "tensor is in block " + std::to_string(block) + " but only the last block (" +
               std::to_string(last_block) +
               ") is trainable on this llama.cpp pin (KV-cache set_rows has no backward; "
               "gradients cannot cross a downstream cache write)";
    }
    if (tensor_name.find("attn_k.") != std::string::npos ||
        tensor_name.find("attn_v.") != std::string::npos ||
        tensor_name.find("attn_kv") != std::string::npos) {
        return "attn_k/attn_v feed the block's own KV-cache write (set_rows, no backward); "
               "train attn_q / attn_output / ffn_* / norms instead";
    }
    return {};
}

std::string device_train_build_corpus(const std::string& dataset_path, const std::string& model_id,
                                      std::string* err) {
    std::ifstream in(dataset_path);
    if (!in) {
        if (err)
            *err = "cannot open dataset: " + dataset_path;
        return {};
    }
    const ChatFormat fmt = chat_format_for(model_id);
    std::string corpus;
    std::string line;
    bool mode_set = false;
    bool jsonl_mode = false;
    while (std::getline(in, line)) {
        const size_t first = line.find_first_not_of(" \t\r");
        if (first == std::string::npos)
            continue;
        if (!mode_set) {
            jsonl_mode = line[first] == '{';
            mode_set = true;
        }
        ChatSample sample;
        if (jsonl_mode) {
            if (line[first] != '{' || !parse_chat_sample_line(line, sample)) {
                if (err)
                    *err = "malformed or mixed-format JSONL dataset: " + dataset_path;
                return {};
            }
            if (sample.label == "dislike")
                continue; // negative preference: excluded from the LM objective
            std::string doc = render_sample(fmt, sample);
            if (!doc.empty()) {
                corpus += doc;
                corpus.push_back('\n');
            }
        } else {
            // Plain-text dataset: passthrough.
            corpus += line;
            corpus.push_back('\n');
        }
    }
    if (corpus.empty() && err)
        *err = "dataset produced an empty training corpus: " + dataset_path;
    return corpus;
}

#ifndef XLLAMA_DEVICE_TRAIN

TrainingResult run_device_train_job(const TrainingJob& job, const DeviceTrainCallbacks& cb) {
    (void)job;
    (void)cb;
    TrainingResult result;
    result.error_msg = "built without XLLAMA_DEVICE_TRAIN (llama.cpp backend required)";
    return result;
}

#else

TrainingResult run_device_train_job(const TrainingJob& job, const DeviceTrainCallbacks& cb) {
    TrainingResult result;
    const auto t_start = std::chrono::steady_clock::now();
    const auto finish = [&](bool ok, const std::string& err_msg) {
        result.success = ok;
        result.error_msg = err_msg;
        result.wall_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
        result.peak_ws_mb = peak_working_set_mb();
        write_result_json(job.out_dir, result);
        return result;
    };

    std::string err;
    if (!validate_training_job(job, &err))
        return finish(false, err);
    if (job.method != TrainMethod::PartialFt)
        return finish(false, "device engine supports method partial_ft only");

    std::error_code ec;
    std::filesystem::create_directories(job.out_dir, ec);
    if (ec)
        return finish(false, "cannot create out_dir: " + job.out_dir + ": " + ec.message());
    std::filesystem::remove(job.out_dir + "/result.json", ec);
    if (ec)
        return finish(false, "cannot clear stale result.json: " + ec.message());

    // ---- prepare ----
    const std::string base_gguf = resolve_base_gguf(job.base_model, &err);
    if (base_gguf.empty())
        return finish(false, err);
    const std::string mixed_gguf = job.out_dir + "/train-base-f32.gguf";
    if (!prepare_mixed_gguf(base_gguf, job.param_filter, mixed_gguf, cb, &err))
        return finish(false, err);

    const std::string corpus = device_train_build_corpus(job.dataset_path, job.base_model, &err);
    if (corpus.empty())
        return finish(false, err);
    result.stages_completed.push_back(TrainStage::Prepare);

    // ---- train ----
    llama_backend_init();
    llama_model_params mparams = llama_model_default_params();
    // Trainable weights must be writable, and UWP has no mmap. b10105 replaced
    // the use_mmap/use_mlock/use_direct_io booleans with this enum; NONE is the
    // former use_mmap=false.
    mparams.load_mode = LLAMA_LOAD_MODE_NONE;
    mparams.n_gpu_layers = 0; // CPU only: Xbox has no ggml GPU backend
    LlamaModelPtr model{llama_model_load_from_file(mixed_gguf.c_str(), mparams)};
    if (!model)
        return finish(false, "train: cannot load " + mixed_gguf);

    const uint32_t n_ctx = static_cast<uint32_t>(job.n_ctx_train);
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = n_ctx;
    cparams.n_batch = n_ctx;        // llama_opt requires n_ctx_train % n_batch == 0
    cparams.n_ubatch = n_ctx;       // opt_period 1 (no gradient accumulation)
    cparams.type_k = GGML_TYPE_F32; // OUT_PROD has no f16 support (see llama-finetune)
    cparams.type_v = GGML_TYPE_F32;
    // FLASH_ATTN_EXT has no backward implementation in the pinned ggml.
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cparams.n_threads = detect_threads_llama();
    cparams.n_threads_batch = cparams.n_threads;
    LlamaContextPtr lctx{llama_init_from_model(model.get(), cparams)};
    if (!lctx)
        return finish(false, "train: cannot create llama context");

    // Tokenize per line-document so each starts with BOS, then concatenate.
    const llama_vocab* vocab = llama_model_get_vocab(model.get());
    std::vector<llama_token> tokens;
    {
        std::istringstream docs(corpus);
        std::string doc;
        while (std::getline(docs, doc)) {
            if (doc.empty())
                continue;
            std::vector<llama_token> doc_tokens(doc.size() + 8);
            int n = llama_tokenize(vocab, doc.c_str(), static_cast<int32_t>(doc.size()),
                                   doc_tokens.data(), static_cast<int32_t>(doc_tokens.size()),
                                   /*add_special=*/true, /*parse_special=*/true);
            if (n < 0) {
                doc_tokens.resize(static_cast<size_t>(-n));
                n = llama_tokenize(vocab, doc.c_str(), static_cast<int32_t>(doc.size()),
                                   doc_tokens.data(), static_cast<int32_t>(doc_tokens.size()),
                                   /*add_special=*/true, /*parse_special=*/true);
            }
            if (n <= 0) {
                return finish(false, "train: failed to tokenize a dataset document");
            }
            tokens.insert(tokens.end(), doc_tokens.begin(), doc_tokens.begin() + n);
        }
    }
    // llama_opt_epoch needs ndata >= 1: T >= 1.5*n_ctx + 2 with stride n_ctx/2.
    // Tiny corpora (toy/preference datasets) are tiled to reach that floor.
    const size_t min_tokens = static_cast<size_t>(n_ctx) * 2 + 2;
    if (tokens.empty())
        return finish(false, "train: dataset tokenized to zero tokens");
    const std::vector<llama_token> token_tile = tokens;
    while (tokens.size() < min_tokens) {
        const size_t count = std::min(token_tile.size(), min_tokens - tokens.size());
        tokens.insert(tokens.end(), token_tile.begin(), token_tile.begin() + count);
    }

    // Dataset windows: same layout as common_opt_dataset_init (no common dep).
    const int64_t ne_datapoint = static_cast<int64_t>(n_ctx);
    const int64_t stride = ne_datapoint / 2;
    const int64_t ndata = (static_cast<int64_t>(tokens.size()) - ne_datapoint - 1) / stride;
    ggml_opt_dataset_t dataset = ggml_opt_dataset_init(GGML_TYPE_I32, GGML_TYPE_I32, ne_datapoint,
                                                       ne_datapoint, ndata, /*ndata_shard=*/1);
    if (!dataset)
        return finish(false, "train: cannot allocate optimizer dataset");
    {
        llama_token* data = static_cast<llama_token*>(ggml_opt_dataset_data(dataset)->data);
        llama_token* labels = static_cast<llama_token*>(ggml_opt_dataset_labels(dataset)->data);
        for (int64_t i = 0; i < ndata; ++i) {
            std::memcpy(data + i * ne_datapoint, tokens.data() + i * stride,
                        ne_datapoint * sizeof(llama_token));
            std::memcpy(labels + i * ne_datapoint, tokens.data() + i * stride + 1,
                        ne_datapoint * sizeof(llama_token));
        }
    }
    {
        std::ostringstream os;
        os << "train: " << tokens.size() << " tokens, " << ndata << " windows of " << n_ctx
           << ", epochs=" << job.epochs;
        status(cb, os.str());
    }

    OptPars opt_pars{job.learning_rate};
    std::vector<std::string> patterns = job.param_filter;
    llama_opt_params lopt{};
    lopt.n_ctx_train = n_ctx;
    lopt.param_filter = param_filter_cb;
    lopt.param_filter_ud = &patterns;
    lopt.get_opt_pars = opt_pars_cb;
    lopt.get_opt_pars_ud = &opt_pars;
    lopt.optimizer_type = GGML_OPT_OPTIMIZER_TYPE_ADAMW;
    llama_opt_init(lctx.get(), model.get(), lopt);

    g_progress = ProgressState{&cb, 0, job.epochs};
    ggml_opt_result_t result_train = ggml_opt_result_init();
    bool aborted = false;
    std::string train_error;
    for (int epoch = 1; epoch <= job.epochs; ++epoch) {
        if (cb.abort_flag && cb.abort_flag->load()) {
            aborted = true;
            break;
        }
        g_progress.epoch = epoch;
        // idata_split = ndata: all data trains; evaluation A/B happens below.
        llama_opt_epoch(lctx.get(), dataset, result_train, /*result_eval=*/nullptr,
                        /*idata_split=*/ndata, epoch_progress_cb, /*callback_eval=*/nullptr);
        double loss = 0.0, unc = 0.0;
        ggml_opt_result_loss(result_train, &loss, &unc);
        result.last_loss = loss;
        ggml_opt_result_reset(result_train);
        if (!std::isfinite(loss)) {
            train_error = "train: optimizer produced a non-finite loss";
            break;
        }
        {
            std::ostringstream os;
            os << "train: epoch " << epoch << "/" << job.epochs << " loss=" << loss;
            status(cb, os.str());
        }
        if (job.checkpoint_every > 0 && epoch % job.checkpoint_every == 0 && epoch != job.epochs) {
            const std::string ckpt =
                job.out_dir + "/checkpoint-epoch" + std::to_string(epoch) + ".gguf";
            std::error_code save_ec;
            std::filesystem::remove(ckpt, save_ec);
            save_ec.clear();
            llama_model_save_to_file(model.get(), ckpt.c_str());
            if (!std::filesystem::is_regular_file(ckpt, save_ec) ||
                std::filesystem::file_size(ckpt, save_ec) == 0) {
                train_error = "train: checkpoint write failed: " + ckpt;
                break;
            }
            status(cb, "train: checkpoint " + ckpt);
        }
    }
    ggml_opt_result_free(result_train);
    ggml_opt_dataset_free(dataset);
    g_progress = ProgressState{};
    if (aborted)
        return finish(false, "train: aborted by caller");
    if (!train_error.empty())
        return finish(false, train_error);
    result.stages_completed.push_back(TrainStage::Train);

    // ---- export (merged GGUF: trained f32 subset + frozen original tensors) ----
    const std::string merged = job.out_dir + "/merged.gguf";
    std::error_code save_ec;
    std::filesystem::remove(merged, save_ec);
    save_ec.clear();
    llama_model_save_to_file(model.get(), merged.c_str());
    if (!std::filesystem::is_regular_file(merged, save_ec) ||
        std::filesystem::file_size(merged, save_ec) == 0)
        return finish(false, "export: merged GGUF write failed: " + merged);
    lctx.reset(); // release optimizer/compute buffers before evaluation
    model.reset();
    result.merged_gguf_path = merged;
    result.stages_completed.push_back(TrainStage::ExportAdapter);
    result.stages_completed.push_back(TrainStage::Merge);
    status(cb, "export: " + merged);

    // ---- evaluate (optional marker A/B, same in-process Session as chat) ----
    if (!job.eval_prompt.empty()) {
        SessionParams sp;
        sp.model_path = merged;
        sp.backend = Backend::LlamaCpp;
        sp.n_ctx = 2048;
        std::string serr;
        auto session = Session::create(sp, &serr);
        if (!session)
            return finish(false, "evaluate: cannot load merged model: " + serr);
        const ChatFormat fmt = chat_format_for(job.base_model);
        GenerateParams gp;
        gp.prompt = fmt.render_prompt("You are a helpful AI assistant.", {}, job.eval_prompt);
        gp.stop_sequences = fmt.stop_sequences;
        gp.n_predict = 64;
        gp.top_k = 1; // greedy
        gp.temperature = 1.0f;
        auto res = session->generate(gp);
        const std::string output = fmt.postprocess_output(res.output_text);
        status(cb, "evaluate: output: " + output);
        if (!job.eval_expect_contains.empty() &&
            output.find(job.eval_expect_contains) == std::string::npos) {
            return finish(false, "evaluate: output does not contain marker \"" +
                                     job.eval_expect_contains + "\"");
        }
        result.stages_completed.push_back(TrainStage::Evaluate);
    }

    return finish(true, {});
}

#endif // XLLAMA_DEVICE_TRAIN

} // namespace xllama
