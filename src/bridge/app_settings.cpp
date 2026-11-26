// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
#include "xllama/app_settings.h"

#include "xllama/debug_dashboard.h" // kDebugDashboardSettingsKey

#include <nlohmann/json.hpp>

#include <charconv>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <optional>

namespace xllama {
namespace {

using json = nlohmann::json;

class Reader {
  public:
    Reader(const json& obj, std::string prefix, std::vector<std::string>& rejected)
        : obj_(obj), prefix_(std::move(prefix)), rejected_(rejected) {}

    void text(const char* key, std::string& out) {
        const json* v = find(key);
        if (!v)
            return;
        if (v->is_string())
            out = v->get<std::string>();
        else
            reject(key);
    }

    // true/false, plus the 1/0 an early writer used.
    void flag(const char* key, bool& out) {
        const json* v = find(key);
        if (!v)
            return;
        if (v->is_boolean())
            out = v->get<bool>();
        else if (const std::optional<long long> n = whole(*v); n && (*n == 0 || *n == 1))
            out = *n == 1;
        else
            reject(key);
    }

    template <typename Int> void integer(const char* key, Int& out, long long lo, long long hi) {
        const json* v = find(key);
        if (!v)
            return;
        const std::optional<long long> n = whole(*v);
        if (n && *n >= lo && *n <= hi)
            out = static_cast<Int>(*n);
        else
            reject(key);
    }

    void real(const char* key, float& out, float lo, float hi) {
        const json* v = find(key);
        if (!v)
            return;
        if (v->is_number()) {
            const double d = v->get<double>();
            if (std::isfinite(d) && d >= lo && d <= hi) {
                out = static_cast<float>(d);
                return;
            }
        }
        reject(key);
    }

    const json* object(const char* key) {
        const json* v = find(key);
        if (v && !v->is_object()) {
            reject(key);
            return nullptr;
        }
        return v;
    }

  private:
    const json* find(const char* key) const {
        const auto it = obj_.find(key);
        return it == obj_.end() ? nullptr : &*it;
    }

    // A JSON number with no fractional part (1 and 1.0 both count), in range
    // of long long.
    static std::optional<long long> whole(const json& v) {
        if (v.is_number_integer())
            return v.get<long long>();
        if (v.is_number_float()) {
            const double d = v.get<double>();
            if (std::isfinite(d) && std::trunc(d) == d && std::fabs(d) < 9.0e15)
                return static_cast<long long>(d);
        }
        return std::nullopt;
    }

    void reject(const char* key) {
        rejected_.push_back(prefix_ + key);
    }

    const json& obj_;
    std::string prefix_;
    std::vector<std::string>& rejected_;
};

// The shortest decimal that reads back as |f|, as a double for the writer:
// 0.925f is written as 0.925, not 0.925000011920929.
double shortest(float f) {
    char buf[32];
    const auto res = std::to_chars(buf, buf + sizeof(buf), f);
    *res.ptr = '\0';
    return std::strtod(buf, nullptr);
}

} // namespace

bool AppSettings::operator==(const AppSettings& o) const {
    return system_prompt == o.system_prompt && model == o.model && kv_reuse == o.kv_reuse &&
           debug_dashboard == o.debug_dashboard && routing == o.routing &&
           gpu_model == o.gpu_model && diffuse_taesd_vae == o.diffuse_taesd_vae &&
           diffuse_seed == o.diffuse_seed && temperature == o.temperature && top_p == o.top_p &&
           top_k == o.top_k && repetition_penalty == o.repetition_penalty &&
           n_predict == o.n_predict;
}

AppSettingsLoad parse_app_settings(std::string_view text) {
    namespace lim = settings_limits;
    AppSettingsLoad out;
    const json root = json::parse(text.begin(), text.end(), nullptr, /*allow_exceptions=*/false);
    if (root.is_discarded() || !root.is_object()) {
        out.error = root.is_discarded() ? "settings.json is not valid JSON"
                                        : "settings.json is not a JSON object";
        return out;
    }
    AppSettings& s = out.settings;
    Reader top(root, "", out.rejected);
    top.text("system_prompt", s.system_prompt);
    top.text("model", s.model);
    top.flag("kv_reuse", s.kv_reuse);
    top.flag(kDebugDashboardSettingsKey, s.debug_dashboard);
    top.integer("routing", s.routing, lim::kRoutingMin, lim::kRoutingMax);
    top.text("gpu_model", s.gpu_model);
    top.flag("diffuse_taesd_vae", s.diffuse_taesd_vae);
    top.integer("diffuse_seed", s.diffuse_seed, 0, std::numeric_limits<std::uint32_t>::max());
    if (const json* sampling = top.object("sampling")) {
        Reader r(*sampling, "sampling.", out.rejected);
        r.real("temperature", s.temperature, lim::kTemperatureMin, lim::kTemperatureMax);
        r.real("top_p", s.top_p, lim::kTopPMin, lim::kTopPMax);
        r.integer("top_k", s.top_k, lim::kTopKMin, lim::kTopKMax);
        r.real("repetition_penalty", s.repetition_penalty, lim::kRepetitionPenaltyMin,
               lim::kRepetitionPenaltyMax);
        r.integer("n_predict", s.n_predict, lim::kNPredictMin, lim::kNPredictMax);
    }
    // An empty gpu_model would select no GPU asset at all; the previous reader
    // also ignored it.
    if (s.gpu_model.empty())
        s.gpu_model = kDefaultGpuModel;
    return out;
}

std::string format_app_settings(const AppSettings& s) {
    nlohmann::ordered_json root;
    root["system_prompt"] = s.system_prompt;
    root["model"] = s.model;
    root["kv_reuse"] = s.kv_reuse;
    root[kDebugDashboardSettingsKey] = s.debug_dashboard;
    root["routing"] = s.routing;
    root["gpu_model"] = s.gpu_model;
    root["diffuse_taesd_vae"] = s.diffuse_taesd_vae;
    root["diffuse_seed"] = s.diffuse_seed;
    nlohmann::ordered_json sampling;
    sampling["temperature"] = shortest(s.temperature);
    sampling["top_p"] = shortest(s.top_p);
    sampling["top_k"] = s.top_k;
    sampling["repetition_penalty"] = shortest(s.repetition_penalty);
    sampling["n_predict"] = s.n_predict;
    root["sampling"] = std::move(sampling);
    return root.dump(2) + "\n";
}

} // namespace xllama
