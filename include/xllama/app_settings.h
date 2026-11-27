// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// The app's persisted settings (LocalState\settings.json): one model, one
// parser and one writer, WinRT-free and host-tested. MainPage only binds it to
// the UI.
//
// The previous hand-written reader skipped unknown keys up to the next
// whitespace (an unknown string value with spaces, or a nested object,
// desynchronised it), parsed numbers with stoi/stof unguarded (a corrupt file
// threw out of page init) and applied no range; the writer rounded floats to
// two decimals. Here every field is validated on its own: a missing, mistyped
// or out-of-range value takes that field's default, unknown keys are ignored
// at any depth, and malformed JSON yields the defaults plus an error to log.
//
// Sampling holds only the values the user chose: an unset field inherits the
// model's value, then the global default (model_sampling.h, the "user" layer).
// Files before schema 2 wrote all four values whether chosen or not; a value
// equal to the old global default reads back unset, so it inherits.
#pragma once

#include "xllama/chat_prompt.h" // kDefaultSystemPrompt
#include "xllama/model_sampling.h"
#include "xllama/sampling.h" // sampling_defaults

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace xllama {

// Ranges the Settings dialog offers and settings.json must respect. The
// dialog builds its sliders from these, so the two cannot drift.
namespace settings_limits {
inline constexpr float kTemperatureMin = 0.0f;
inline constexpr float kTemperatureMax = 2.0f;
inline constexpr float kTopPMin = 0.0f;
inline constexpr float kTopPMax = 1.0f;
inline constexpr float kRepetitionPenaltyMin = 1.0f;
inline constexpr float kRepetitionPenaltyMax = 2.0f;
inline constexpr int kTopKMin = 1;
inline constexpr int kTopKMax = 200;
inline constexpr int kNPredictMin = 16;
inline constexpr int kNPredictMax = 2048;
inline constexpr int kRoutingMin = 0; // RoutingMode::CpuOnly
inline constexpr int kRoutingMax = 2; // RoutingMode::Auto
} // namespace settings_limits

inline constexpr int kDefaultNPredict = 512;
inline constexpr const char kDefaultGpuModel[] = "smollm2-360m-dml-fp16-v2";
inline constexpr const char kSettingsFileName[] = "settings.json";
// Written as "schema"; 2 = sampling holds only the user's own values.
inline constexpr int kAppSettingsSchema = 2;

struct AppSettings {
    std::string system_prompt = kDefaultSystemPrompt;
    std::string model; // UTF-8 catalogue id; empty = not chosen (caller falls back)
    bool kv_reuse = true;
    bool debug_dashboard = false;
    int routing = 0; // RoutingMode as int
    std::string gpu_model = kDefaultGpuModel;
    bool diffuse_taesd_vae = false;
    std::uint32_t diffuse_seed = 0; // 0 = choose a random seed per image
    SamplingOverrides sampling;     // the user's own values; unset = the model's
    int n_predict = kDefaultNPredict;

    bool operator==(const AppSettings& o) const;
};

struct AppSettingsLoad {
    AppSettings settings;
    // Non-empty when the text is not a JSON object: every field is default.
    std::string error;
    // Keys present but rejected (wrong type or out of range), so they took the
    // default, e.g. "sampling.top_k". For the log, not for the user.
    std::vector<std::string> rejected;
};

AppSettingsLoad parse_app_settings(std::string_view json);

// Pretty-printed JSON object with a trailing newline. Floats are written with
// the shortest text that reads back as the same float.
std::string format_app_settings(const AppSettings& s);

} // namespace xllama
