// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include "xllama/personalize.h"
#include "xllama/training.h"

#include <cctype>
#include <cstdio>
#include <fstream>
#include <sstream>

namespace xllama {
namespace {

void set_err(std::string* err, const std::string& msg) {
    if (err)
        *err = msg;
}

std::string to_lower(std::string s) {
    for (char& c : s)
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

// Minimal JSON string escape for values we control (paths, display).
std::string json_escape(const std::string& s) {
    std::string o;
    o.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
        case '"':
            o += "\\\"";
            break;
        case '\\':
            o += "\\\\";
            break;
        case '\n':
            o += "\\n";
            break;
        case '\r':
            o += "\\r";
            break;
        case '\t':
            o += "\\t";
            break;
        default:
            o += c;
            break;
        }
    }
    return o;
}

// Prefer the basename when the path looks like models/<id>/...
std::string model_id_key(const std::string& model_id_or_path) {
    std::string s = model_id_or_path;
    // Normalize separators.
    for (char& c : s) {
        if (c == '\\')
            c = '/';
    }
    // Strip trailing slash.
    while (!s.empty() && s.back() == '/')
        s.pop_back();
    auto slash = s.find_last_of('/');
    if (slash != std::string::npos)
        s = s.substr(slash + 1);
    // Strip extension for bare files.
    auto dot = s.find_last_of('.');
    if (dot != std::string::npos && s.substr(dot) == ".gguf")
        s = s.substr(0, dot);
    return to_lower(s);
}

} // namespace

std::vector<std::string> last_block_param_filter(int last_block) {
    if (last_block < 0)
        return {};
    const std::string b = "blk." + std::to_string(last_block);
    return {
        b + ".attn_q.weight", b + ".attn_output.weight", b + ".ffn_gate.weight",
        b + ".ffn_up.weight", b + ".ffn_down.weight",    "output_norm.weight",
    };
}

int guess_last_block_from_model_id(const std::string& model_id_or_path) {
    const std::string k = model_id_key(model_id_or_path);
    // base-f16 / training defaults → SmolLM2-360M recipe used by device-train.
    if (k == "base-f16" || k.find("base-f16") != std::string::npos)
        return 31;
    if (k.find("smollm2") != std::string::npos) {
        if (k.find("1.7") != std::string::npos || k.find("1_7") != std::string::npos)
            return 23; // SmolLM2-1.7B: 24 layers
        // 360M and unspecified smollm2 → 32 layers
        return 31;
    }
    return -1;
}

bool build_personalize_job(const PersonalizeSpec& spec, TrainingJob& out, std::string* err) {
    if (spec.base_model.empty()) {
        set_err(err, "base_model is required");
        return false;
    }
    if (spec.dataset_path.empty()) {
        set_err(err, "dataset_path is required");
        return false;
    }
    if (spec.out_dir.empty()) {
        set_err(err, "out_dir is required");
        return false;
    }
    if (spec.epochs < 1) {
        set_err(err, "epochs must be >= 1");
        return false;
    }
    if (spec.n_ctx_train < 32) {
        set_err(err, "n_ctx_train must be >= 32");
        return false;
    }

    int last = spec.last_block;
    if (last < 0)
        last = guess_last_block_from_model_id(spec.base_model);
    if (last < 0) {
        set_err(err, "cannot guess last_block from base_model; set PersonalizeSpec::last_block");
        return false;
    }

    TrainingJob job;
    job.schema_version = 1;
    job.name = spec.name.empty() ? "personalized" : spec.name;
    job.method = TrainMethod::PartialFt;
    job.device = TrainDevice::Device;
    job.base_model = spec.base_model;
    job.dataset_path = spec.dataset_path;
    job.out_dir = spec.out_dir;
    job.param_filter = last_block_param_filter(last);
    job.n_ctx_train = spec.n_ctx_train;
    job.epochs = spec.epochs;
    job.learning_rate = spec.learning_rate;
    job.eval_prompt = spec.eval_prompt;
    job.eval_expect_contains = spec.eval_expect_contains;
    job.checkpoint_every = 0;
    job.do_merge = true;

    if (!validate_training_job(job, err))
        return false;

    out = std::move(job);
    return true;
}

std::string format_personalize_job_json(const TrainingJob& job) {
    std::ostringstream os;
    os << "{\n";
    os << "  \"schema_version\": " << job.schema_version << ",\n";
    os << "  \"name\": \"" << json_escape(job.name) << "\",\n";
    os << "  \"method\": \"partial_ft\",\n";
    os << "  \"device\": \"device\",\n";
    os << "  \"base_model\": \"" << json_escape(job.base_model) << "\",\n";
    os << "  \"dataset\": \"" << json_escape(job.dataset_path) << "\",\n";
    os << "  \"out_dir\": \"" << json_escape(job.out_dir) << "\",\n";
    os << "  \"param_filter\": [\n";
    for (size_t i = 0; i < job.param_filter.size(); ++i) {
        os << "    \"" << json_escape(job.param_filter[i]) << "\"";
        if (i + 1 < job.param_filter.size())
            os << ",";
        os << "\n";
    }
    os << "  ],\n";
    os << "  \"n_ctx_train\": " << job.n_ctx_train << ",\n";
    os << "  \"epochs\": " << job.epochs << ",\n";
    os << "  \"learning_rate\": " << job.learning_rate;
    if (!job.eval_prompt.empty()) {
        os << ",\n  \"eval\": { \"prompt\": \"" << json_escape(job.eval_prompt)
           << "\", \"expect_contains\": \"" << json_escape(job.eval_expect_contains) << "\" }";
    }
    os << "\n}\n";
    return os.str();
}

std::string personalized_manifest_override_json(const std::string& model_id,
                                                const std::string& display, uint64_t approx_bytes) {
    const std::string id = model_id.empty() ? kPersonalizedModelId : model_id;
    const std::string disp = display.empty() ? kPersonalizedDisplay : display;
    std::ostringstream os;
    os << "{\n"
       << "  \"models\": [\n"
       << "    {\n"
       << "      \"name\": \"" << json_escape(id) << "\",\n"
       << "      \"display\": \"" << json_escape(disp) << "\",\n"
       << "      \"kind\": \"gguf\",\n"
       << "      \"files\": [\n"
       << "        {\n"
       << "          \"filename\": \"model.gguf\",\n"
       << "          \"approx_bytes\": " << approx_bytes << "\n"
       << "        }\n"
       << "      ]\n"
       << "    }\n"
       << "  ]\n"
       << "}\n";
    return os.str();
}

int count_usable_preference_samples(const std::string& samples_path) {
    std::ifstream in(samples_path);
    if (!in)
        return 0;
    int n = 0;
    std::string line;
    while (std::getline(in, line)) {
        // Skip empty / whitespace.
        size_t i = 0;
        while (i < line.size() && std::isspace(static_cast<unsigned char>(line[i])))
            ++i;
        if (i >= line.size())
            continue;
        // Engine skips dislike; do not count them for preflight.
        if (line.find("\"label\":\"dislike\"") != std::string::npos ||
            line.find("\"label\": \"dislike\"") != std::string::npos)
            continue;
        // Require a label field so garbage lines are ignored.
        if (line.find("\"label\"") == std::string::npos)
            continue;
        ++n;
    }
    return n;
}

std::string format_train_progress_json(const std::string& stage, int epoch, int epochs,
                                       int64_t ibatch, int64_t ibatch_max, double loss) {
    char buf[320];
    std::snprintf(buf, sizeof(buf),
                  "{\"stage\":\"%s\",\"epoch\":%d,\"epochs\":%d,\"ibatch\":%lld,"
                  "\"ibatch_max\":%lld,\"loss\":%.6f}",
                  json_escape(stage).c_str(), epoch, epochs, static_cast<long long>(ibatch),
                  static_cast<long long>(ibatch_max), loss);
    return std::string(buf);
}

std::string parse_train_result_done(const std::string& content) {
    size_t i = 0;
    while (i < content.size() && std::isspace(static_cast<unsigned char>(content[i])))
        ++i;
    if (content.compare(i, 2, "ok") == 0)
        return "ok";
    if (content.compare(i, 4, "fail") == 0)
        return "fail";
    return {};
}

} // namespace xllama
