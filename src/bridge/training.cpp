// Copyright (c) 2024 Venere Labs
// SPDX-License-Identifier: MIT

#include "xllama/training.h"

#include <cctype>
#include <cmath>
#include <fstream>
#include <sstream>

namespace xllama {
namespace {

void set_err(std::string* err, const std::string& msg) {
    if (err)
        *err = msg;
}

// --- tiny JSON helpers (object of string/number/bool/null only; nested one level) ---

void skip_ws(const std::string& s, size_t& i) {
    while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i])))
        ++i;
}

bool match_char(const std::string& s, size_t& i, char c) {
    skip_ws(s, i);
    if (i < s.size() && s[i] == c) {
        ++i;
        return true;
    }
    return false;
}

bool parse_string(const std::string& s, size_t& i, std::string& out) {
    skip_ws(s, i);
    if (i >= s.size() || s[i] != '"')
        return false;
    ++i;
    out.clear();
    while (i < s.size()) {
        char c = s[i++];
        if (c == '"')
            return true;
        if (c == '\\' && i < s.size()) {
            char e = s[i++];
            switch (e) {
            case '"':
            case '\\':
            case '/':
                out.push_back(e);
                break;
            case 'n':
                out.push_back('\n');
                break;
            case 't':
                out.push_back('\t');
                break;
            case 'r':
                out.push_back('\r');
                break;
            default:
                out.push_back(e);
                break;
            }
        } else {
            out.push_back(c);
        }
    }
    return false;
}

bool parse_number(const std::string& s, size_t& i, double& out) {
    skip_ws(s, i);
    size_t start = i;
    if (i < s.size() && (s[i] == '-' || s[i] == '+'))
        ++i;
    if (i >= s.size() || !std::isdigit(static_cast<unsigned char>(s[i])))
        return false;
    while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i])))
        ++i;
    if (i < s.size() && s[i] == '.') {
        ++i;
        while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i])))
            ++i;
    }
    if (i < s.size() && (s[i] == 'e' || s[i] == 'E')) {
        ++i;
        if (i < s.size() && (s[i] == '-' || s[i] == '+'))
            ++i;
        while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i])))
            ++i;
    }
    try {
        out = std::stod(s.substr(start, i - start));
    } catch (...) {
        return false;
    }
    return true;
}

bool parse_bool_or_null(const std::string& s, size_t& i, bool& is_null, bool& bool_val) {
    skip_ws(s, i);
    if (s.compare(i, 4, "null") == 0) {
        i += 4;
        is_null = true;
        return true;
    }
    if (s.compare(i, 4, "true") == 0) {
        i += 4;
        is_null = false;
        bool_val = true;
        return true;
    }
    if (s.compare(i, 5, "false") == 0) {
        i += 5;
        is_null = false;
        bool_val = false;
        return true;
    }
    return false;
}

// Skip a nested object or array without interpreting it (best-effort).
bool skip_value(const std::string& s, size_t& i);

bool skip_container(const std::string& s, size_t& i, char open_c, char close_c) {
    if (!match_char(s, i, open_c))
        return false;
    int depth = 1;
    while (i < s.size() && depth > 0) {
        char c = s[i];
        if (c == '"') {
            std::string tmp;
            if (!parse_string(s, i, tmp))
                return false;
            continue;
        }
        if (c == open_c)
            ++depth;
        else if (c == close_c)
            --depth;
        ++i;
    }
    return depth == 0;
}

bool skip_value(const std::string& s, size_t& i) {
    skip_ws(s, i);
    if (i >= s.size())
        return false;
    if (s[i] == '"') {
        std::string tmp;
        return parse_string(s, i, tmp);
    }
    if (s[i] == '{')
        return skip_container(s, i, '{', '}');
    if (s[i] == '[')
        return skip_container(s, i, '[', ']');
    bool is_null = false, b = false;
    if (parse_bool_or_null(s, i, is_null, b))
        return true;
    double d = 0;
    return parse_number(s, i, d);
}

bool parse_string_array(const std::string& s, size_t& i, std::vector<std::string>& out) {
    if (!match_char(s, i, '['))
        return false;
    out.clear();
    skip_ws(s, i);
    if (match_char(s, i, ']'))
        return true;
    for (;;) {
        std::string item;
        if (!parse_string(s, i, item))
            return false;
        out.push_back(item);
        if (match_char(s, i, ']'))
            return true;
        if (!match_char(s, i, ','))
            return false;
    }
}

bool apply_string_field(TrainingJob& job, const std::string& key, const std::string& val) {
    if (key == "name")
        job.name = val;
    else if (key == "base_model")
        job.base_model = val;
    else if (key == "dataset")
        job.dataset_path = val;
    else if (key == "out_dir")
        job.out_dir = val;
    else if (key == "method") {
        if (val == "lora_peft")
            job.method = TrainMethod::LoraPeft;
        else if (val == "partial_ft")
            job.method = TrainMethod::PartialFt;
        else if (val == "full_ft" || val == "full_ft_reserved")
            job.method = TrainMethod::FullFtReserved;
        else
            return false;
    } else if (key == "device") {
        if (val == "host")
            job.device = TrainDevice::Host;
        else if (val == "device")
            job.device = TrainDevice::Device;
        else
            return false;
    } else if (key == "prompt" || key == "eval_prompt")
        job.eval_prompt = val;
    else if (key == "expect_contains" || key == "eval_expect_contains")
        job.eval_expect_contains = val;
    // ignore unknown string keys
    return true;
}

bool apply_number_field(TrainingJob& job, const std::string& key, double val) {
    if (key == "schema_version")
        job.schema_version = static_cast<int>(val);
    else if (key == "rank" || key == "lora_rank")
        job.lora_rank = static_cast<int>(val);
    else if (key == "alpha" || key == "lora_alpha")
        job.lora_alpha = static_cast<int>(val);
    else if (key == "steps")
        job.steps = static_cast<int>(val);
    else if (key == "seed")
        job.seed = static_cast<int>(val);
    else if (key == "learning_rate" || key == "lr")
        job.learning_rate = static_cast<float>(val);
    else if (key == "n_ctx_train")
        job.n_ctx_train = static_cast<int>(val);
    else if (key == "epochs")
        job.epochs = static_cast<int>(val);
    else if (key == "checkpoint_every")
        job.checkpoint_every = static_cast<int>(val);
    return true;
}

bool apply_bool_field(TrainingJob& job, const std::string& key, bool val, bool is_null) {
    if (key == "merge") {
        job.do_merge = is_null ? true : val;
        return true;
    }
    if (key == "quantize") {
        job.do_quantize = is_null ? false : val;
        return true;
    }
    return true;
}

// Parse nested object keys into the flat TrainingJob (lora.* / eval.*).
bool parse_object_into_job(const std::string& s, size_t& i, TrainingJob& job, std::string* err) {
    if (!match_char(s, i, '{')) {
        set_err(err, "expected '{'");
        return false;
    }
    skip_ws(s, i);
    if (match_char(s, i, '}'))
        return true;

    for (;;) {
        std::string key;
        if (!parse_string(s, i, key)) {
            set_err(err, "expected object key string");
            return false;
        }
        if (!match_char(s, i, ':')) {
            set_err(err, "expected ':' after key");
            return false;
        }
        skip_ws(s, i);
        if (i >= s.size()) {
            set_err(err, "unexpected end of JSON");
            return false;
        }

        if (s[i] == '{') {
            // Nested object: flatten known nests (lora, eval)
            if (!parse_object_into_job(s, i, job, err))
                return false;
        } else if (s[i] == '[') {
            if (key == "param_filter") {
                if (!parse_string_array(s, i, job.param_filter)) {
                    set_err(err, "param_filter must be an array of strings");
                    return false;
                }
            } else if (!skip_value(s, i)) {
                set_err(err, "failed to skip array");
                return false;
            }
        } else if (s[i] == '"') {
            std::string val;
            if (!parse_string(s, i, val)) {
                set_err(err, "bad string value for " + key);
                return false;
            }
            if (!apply_string_field(job, key, val)) {
                set_err(err, "unknown enum value for " + key + ": " + val);
                return false;
            }
        } else {
            bool is_null = false, b = false;
            if (parse_bool_or_null(s, i, is_null, b)) {
                apply_bool_field(job, key, b, is_null);
            } else {
                double num = 0;
                if (!parse_number(s, i, num)) {
                    set_err(err, "bad value for " + key);
                    return false;
                }
                apply_number_field(job, key, num);
            }
        }

        skip_ws(s, i);
        if (match_char(s, i, '}'))
            return true;
        if (!match_char(s, i, ',')) {
            set_err(err, "expected ',' or '}' in object");
            return false;
        }
    }
}

} // namespace

const char* training_stage_name(TrainStage stage) {
    switch (stage) {
    case TrainStage::Prepare:
        return "prepare";
    case TrainStage::Train:
        return "train";
    case TrainStage::ExportAdapter:
        return "export_adapter";
    case TrainStage::Merge:
        return "merge";
    case TrainStage::Evaluate:
        return "evaluate";
    case TrainStage::Publish:
        return "publish";
    }
    return "unknown";
}

const char* training_method_name(TrainMethod method) {
    switch (method) {
    case TrainMethod::LoraPeft:
        return "lora_peft";
    case TrainMethod::PartialFt:
        return "partial_ft";
    case TrainMethod::FullFtReserved:
        return "full_ft_reserved";
    }
    return "unknown";
}

const char* training_device_name(TrainDevice device) {
    switch (device) {
    case TrainDevice::Host:
        return "host";
    case TrainDevice::Device:
        return "device";
    }
    return "unknown";
}

bool training_device_supported(TrainDevice device) {
#ifdef XLLAMA_DEVICE_TRAIN
    // Lane B engine (ggml-opt partial FT) is compiled in: both lanes exist.
    (void)device;
    return true;
#else
    return device == TrainDevice::Host;
#endif
}

namespace {

// SSOT table — keep in sync with docs/training-architecture.md capability matrix.
const TrainingCapabilityInfo kCapabilities[] = {
    {TrainingCapability::HostPeftLora, true, "available", "HostPeftLora",
     "training/host PEFT LoRA; host-validated marker job PASS"},
    {TrainingCapability::HostMergeGguf, true, "available", "HostMergeGguf",
     "llama-export-lora + convert_lora_to_gguf on host"},
    {TrainingCapability::HostEvaluateMarker, true, "available", "HostEvaluateMarker",
     "xllama-cli --chat --greedy A/B vs eval.expect_contains"},
    {TrainingCapability::RuntimeLoraLoadLlama, true, "available", "RuntimeLoraLoadLlama",
     "SessionParams.lora_path / CLI --lora → llama_set_adapters_lora (GGUF only)"},
    {TrainingCapability::RuntimeAdapterLoadOrtGenAI, false, "designed",
     "RuntimeAdapterLoadOrtGenAI",
     "OgaLoadAdapter in GenAI DLL; DML blocked (\"No adapter is available for DML\")"},
    {TrainingCapability::DeviceOrtOnDeviceTraining, false, "research", "DeviceOrtOnDeviceTraining",
     "needs ORT Training package + offline artifacts; not in MSIX NuGet pins"},
    {TrainingCapability::DeviceLlamaFinetune, false, "rejected", "DeviceLlamaFinetune",
     "llama-finetune full FT cites ~24 GB class; exceeds Series S practical budget"},
    {TrainingCapability::DeviceGgmlPartialFt,
#ifdef XLLAMA_DEVICE_TRAIN
     true, "experimental",
#else
     false, "designed",
#endif
     "DeviceGgmlPartialFt",
     "in-process ggml-opt partial FT (llama_opt param filter); host and console gates pending"},
    {TrainingCapability::DevicePreferenceCapture, true, "available", "DevicePreferenceCapture",
     "autopilot op rate → LocalState/training/samples.jsonl (host retrain input)"},
};

constexpr size_t kCapabilityCount = sizeof(kCapabilities) / sizeof(kCapabilities[0]);

} // namespace

size_t training_capabilities(const TrainingCapabilityInfo** out) {
    if (out)
        *out = kCapabilities;
    return kCapabilityCount;
}

const TrainingCapabilityInfo* training_capability_info(TrainingCapability c) {
    for (size_t i = 0; i < kCapabilityCount; ++i) {
        if (kCapabilities[i].id == c)
            return &kCapabilities[i];
    }
    return nullptr;
}

bool training_capability_available(TrainingCapability c) {
    const TrainingCapabilityInfo* info = training_capability_info(c);
    return info && info->available;
}

bool validate_training_job(const TrainingJob& job, std::string* err) {
    if (job.schema_version < 1) {
        set_err(err, "schema_version must be >= 1");
        return false;
    }
    if (job.name.empty()) {
        set_err(err, "name is required");
        return false;
    }
    if (job.base_model.empty()) {
        set_err(err, "base_model is required");
        return false;
    }
    if (job.dataset_path.empty()) {
        set_err(err, "dataset is required");
        return false;
    }
    if (job.out_dir.empty()) {
        set_err(err, "out_dir is required");
        return false;
    }
    if (job.method == TrainMethod::FullFtReserved) {
        set_err(err, "method full_ft_reserved is not implemented (exploration: use lora_peft)");
        return false;
    }
    if (job.method != TrainMethod::LoraPeft && job.method != TrainMethod::PartialFt) {
        set_err(err, "unsupported train method");
        return false;
    }
    if (!training_device_supported(job.device)) {
        set_err(err, "device partial_ft is unavailable in this build "
                     "(XLLAMA_DEVICE_TRAIN is required; use a llamacpp/unified build)");
        return false;
    }
    if (job.device == TrainDevice::Device && job.method != TrainMethod::PartialFt) {
        set_err(err, "device lane supports method partial_ft only "
                     "(lora_peft runs on the host PEFT pipeline)");
        return false;
    }
    if (job.method == TrainMethod::PartialFt) {
        if (job.param_filter.empty()) {
            set_err(err, "partial_ft requires a non-empty param_filter "
                         "(tensor-name substring patterns)");
            return false;
        }
        bool has_pattern = false;
        for (const auto& pattern : job.param_filter)
            has_pattern = has_pattern || !pattern.empty();
        if (!has_pattern) {
            set_err(err, "partial_ft param_filter must contain a non-empty pattern");
            return false;
        }
        if (job.n_ctx_train < 64 || job.n_ctx_train > 4096) {
            set_err(err, "n_ctx_train out of range [64, 4096]");
            return false;
        }
        if (job.epochs < 1) {
            set_err(err, "epochs must be >= 1");
            return false;
        }
        if (job.checkpoint_every < 0) {
            set_err(err, "checkpoint_every must be >= 0");
            return false;
        }
    }
    if (job.lora_rank < 1 || job.lora_rank > 256) {
        set_err(err, "lora rank out of range [1, 256]");
        return false;
    }
    if (job.lora_alpha < 1) {
        set_err(err, "lora alpha must be >= 1");
        return false;
    }
    if (job.steps < 1) {
        set_err(err, "steps must be >= 1");
        return false;
    }
    if (!std::isfinite(job.learning_rate) || job.learning_rate <= 0.0f) {
        set_err(err, "learning_rate must be finite and > 0");
        return false;
    }
    return true;
}

bool parse_training_job_json(const std::string& json, TrainingJob& out, std::string* err) {
    TrainingJob job; // defaults
    size_t i = 0;
    if (!parse_object_into_job(json, i, job, err))
        return false;
    skip_ws(json, i);
    // trailing whitespace ok
    out = std::move(job);
    return true;
}

bool load_training_job_file(const std::string& path, TrainingJob& out, std::string* err) {
    std::ifstream in(path);
    if (!in) {
        set_err(err, "cannot open training job file: " + path);
        return false;
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    if (!parse_training_job_json(ss.str(), out, err))
        return false;
    return validate_training_job(out, err);
}

std::string format_training_job_summary(const TrainingJob& job) {
    std::ostringstream os;
    os << "train-job name=" << job.name << " method=" << training_method_name(job.method)
       << " device=" << training_device_name(job.device) << " base=" << job.base_model;
    if (job.method == TrainMethod::PartialFt) {
        os << " epochs=" << job.epochs << " n_ctx_train=" << job.n_ctx_train
           << " param_filter=" << job.param_filter.size();
    } else {
        os << " steps=" << job.steps << " rank=" << job.lora_rank;
    }
    os << " out=" << job.out_dir;
    return os.str();
}

} // namespace xllama
