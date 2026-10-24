// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// One-way decision scoring: a single forward pass, then softmax over the
// candidate label-token logits. No generation on the decider path.
// Reference: ggml-org/llama.cpp#30361 and typesafe.ai System One / Jev.
//
// 204/231 is vendor-claimed (H2O-Lightning-4B stock-Q4_K_M GGUF card). This
// header never treats that figure as a measured xllama result.

#pragma once

#include "xllama/inference_params.h"
#include "xllama/json_utils.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace xllama {

// Vendor-claimed JevBench public-231 figure for H2O-Lightning-4B Q4_K_M.
// Source: https://huggingface.co/h2oai/h2o-lightning-4b-gguf (Correct / 231
// table). Not an xllama measurement.
inline constexpr int kVendorClaimedJevbenchCorrect = 204;
inline constexpr int kVendorClaimedJevbenchTotal = 231;
inline constexpr const char* kVendorClaimedLabel = "vendor-claimed";

inline constexpr const char* kDefaultDecisionSystem =
    "You are a decision engine. You read a record and answer one question "
    "about it by choosing exactly one option. Reply with a single letter.";
inline constexpr const char* kDefaultDecisionPrefill = "Answer:";

// H2O serve_config.json temperatures (one-way readout). Used when a job leaves
// temperature at 0 and mode is "score".
inline constexpr float kH2oTempChoice = 0.75f;
inline constexpr float kH2oTempNoul = 0.8f;
inline constexpr float kH2oTempScore = 0.65f;
inline constexpr float kH2oNoulFloor = 0.801f;

// Default filenames only. Resolved by the existing path helpers / the job
// JSON the operator supplies. Never a host, console path, or engine index.
inline constexpr const char* kDefaultLightningGgufName = "h2o-lightning-4b-stock-Q4_K_M.gguf";
inline constexpr const char* kDefaultPhi4MiniName = "phi4-mini";
inline constexpr const char* kDefaultJevbenchDirName = "jevbench";
inline constexpr const char* kDefaultDecideJobName = "decide_job.json";
inline constexpr const char* kDefaultDecideResultName = "decide-result.json";

// ggml-org/llama.cpp#30361 tools/server/server-decision.cpp:24
// DECISION_H2O_MAX_STATE_TOKENS. A state longer than this keeps its head and
// tail around a "..." line (h2o_cut_state, same file ~725-745).
inline constexpr size_t kH2oMaxStateTokens = 32000;
// serve_config.json / #30361 n_options_max for H2O-Lightning.
inline constexpr size_t kH2oMaxOptions = 255;

// A-Z then a-z. H2O's full 255-label table is h2o_lightning_letter_labels().
inline constexpr std::string_view kAsciiLetterLabels[] = {
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R",
    "S", "T", "U", "V", "W", "X", "Y", "Z", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j",
    "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
};

struct GgufDecisionMeta {
    bool ok = false;
    std::string error;
    std::string architecture;
    std::string decision_type;       // empty when {arch}.decision.type is absent
    std::vector<std::string> labels; // {arch}.decision.labels, as stored
    bool has_decision_labels = false;
    // "gguf" when the array is present; "serve_config-fallback" when it is not.
    std::string label_source;
};

struct LabelScore {
    std::string label;
    int32_t first_token = -1;
    int token_count = 0;
    float logit = 0.0f;
    float probability = 0.0f;
};

struct LabelScoreResult {
    bool ok = false;
    std::string error;
    std::vector<LabelScore> scores;
    int winner = -1;
    bool tie = false;
    bool first_token_collision = false;
};

struct DecisionQuestion {
    std::string type; // choice | noul | score
    std::string instructions;
    std::vector<std::string> option_names;
    std::vector<std::string> option_descs;
};

struct DecisionPromptSpec {
    std::string system = kDefaultDecisionSystem;
    std::string prefill = kDefaultDecisionPrefill;
    bool chat_template = true;
};

struct JevBenchItem {
    std::string id;
    std::string family;
    std::string split;
    std::string expected;
    std::string state;
    DecisionQuestion question;
};

struct DecisionJob {
    std::string model_path;
    std::string questions_dir;
    std::string out_path;
    std::string mode = "score"; // score | generate-json
    std::string backend = "cpu";
    int n_gpu_layers = 0;
    int n_threads = 0;
    int n_ctx = 8192;
    float temperature = 0.0f; // 0 = per-type H2O defaults in score mode
    bool apply_noul_floor = false;
    float noul_floor = kH2oNoulFloor;
    std::string host_label;
    std::string build_id;
    std::string git_commit;
    std::string device;
};

struct DecisionItemResult {
    std::string id;
    std::string family;
    std::string expected;
    std::string predicted;
    bool correct = false;
    bool ok = false;
    std::string error;
    double latency_ms = 0.0;
    std::vector<LabelScore> scores;
};

struct DecisionRunReport {
    bool ok = false;
    std::string error;
    std::string mode;
    std::string model_path;
    std::string backend;
    int n_gpu_layers = 0;
    std::string host_label;
    std::string build_id = "pending";
    std::string git_commit = "pending";
    std::string device = "pending";
    int n_items = 0;
    int n_correct = 0;
    int n_errors = 0;
    double accuracy = 0.0;
    double latency_p50_ms = 0.0;
    double latency_p95_ms = 0.0;
    double t_load_ms = 0.0;
    size_t peak_ws_mb = 0;
    std::string vendor_claimed_label = kVendorClaimedLabel;
    int vendor_claimed_correct = kVendorClaimedJevbenchCorrect;
    int vendor_claimed_total = kVendorClaimedJevbenchTotal;
    std::string gguf_architecture;
    std::string decision_type;
    std::string label_source;
    int n_gguf_labels = 0;
    std::vector<DecisionItemResult> items;
};

struct DecisionScoreRequest {
    std::string prompt;
    std::vector<std::string> labels;
    float temperature = 1.0f;
};

// Softmax over the candidate first-token logits (not the full vocabulary).
// temperature <= 0 is greedy: probability 1 on the unique argmax.
// A shared first token among two or more labels is a collision: no unique
// winner. Equal winning probabilities (1e-6 relative, 1e-9 absolute) are a
// tie; the lowest index stays the reported winner.
inline LabelScoreResult score_label_logits(const std::vector<std::string>& labels,
                                           const std::vector<int32_t>& first_tokens,
                                           const std::vector<int>& token_counts,
                                           const float* logits, int n_vocab, float temperature) {
    LabelScoreResult out;
    if (labels.empty()) {
        out.error = "no labels";
        return out;
    }
    if (first_tokens.size() != labels.size() || token_counts.size() != labels.size()) {
        out.error = "label / token-id size mismatch";
        return out;
    }
    if (!logits || n_vocab <= 0) {
        out.error = "missing logits";
        return out;
    }

    out.scores.resize(labels.size());
    for (size_t i = 0; i < labels.size(); ++i) {
        const int32_t tok = first_tokens[i];
        if (tok < 0 || tok >= n_vocab) {
            out.error = "label token out of range: " + labels[i];
            return out;
        }
        if (token_counts[i] <= 0) {
            out.error = "empty label tokenization: " + labels[i];
            return out;
        }
        out.scores[i].label = labels[i];
        out.scores[i].first_token = tok;
        out.scores[i].token_count = token_counts[i];
        out.scores[i].logit = logits[tok];
        if (!std::isfinite(out.scores[i].logit)) {
            out.error = "non-finite logit for label: " + labels[i];
            return out;
        }
    }

    for (size_t i = 0; i < out.scores.size(); ++i) {
        for (size_t j = i + 1; j < out.scores.size(); ++j) {
            if (out.scores[i].first_token == out.scores[j].first_token) {
                out.first_token_collision = true;
            }
        }
    }

    if (temperature <= 0.0f) {
        int best = 0;
        bool tie = false;
        for (size_t i = 1; i < out.scores.size(); ++i) {
            if (out.scores[i].logit > out.scores[static_cast<size_t>(best)].logit) {
                best = static_cast<int>(i);
                tie = false;
            } else if (out.scores[i].logit == out.scores[static_cast<size_t>(best)].logit) {
                tie = true;
            }
        }
        for (auto& s : out.scores)
            s.probability = 0.0f;
        out.scores[static_cast<size_t>(best)].probability = 1.0f;
        out.winner = best;
        out.tie = tie;
        out.ok = !out.first_token_collision;
        if (out.first_token_collision)
            out.error = "first-token collision; one forward pass cannot split these labels";
        return out;
    }

    float max_logit = out.scores[0].logit / temperature;
    for (size_t i = 1; i < out.scores.size(); ++i)
        max_logit = std::max(max_logit, out.scores[i].logit / temperature);

    double sum = 0.0;
    std::vector<double> exps(out.scores.size(), 0.0);
    for (size_t i = 0; i < out.scores.size(); ++i) {
        exps[i] = std::exp(static_cast<double>(out.scores[i].logit / temperature - max_logit));
        sum += exps[i];
    }
    if (!(sum > 0.0) || !std::isfinite(sum)) {
        out.error = "softmax overflow";
        return out;
    }
    for (size_t i = 0; i < out.scores.size(); ++i)
        out.scores[i].probability = static_cast<float>(exps[i] / sum);

    int best = 0;
    bool tie = false;
    constexpr float kAbs = 1e-9f;
    constexpr float kRel = 1e-6f;
    for (size_t i = 1; i < out.scores.size(); ++i) {
        const float a = out.scores[static_cast<size_t>(best)].probability;
        const float b = out.scores[i].probability;
        const float tol = kAbs + kRel * std::max(std::fabs(a), std::fabs(b));
        if (b > a + tol) {
            best = static_cast<int>(i);
            tie = false;
        } else if (std::fabs(b - a) <= tol) {
            tie = true;
        }
    }
    out.winner = best;
    out.tie = tie;
    out.ok = !out.first_token_collision;
    if (out.first_token_collision)
        out.error = "first-token collision; one forward pass cannot split these labels";
    return out;
}

inline std::string gbnf_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (unsigned char c : s) {
        if (c == '"' || c == '\\')
            out.push_back('\\');
        out.push_back(static_cast<char>(c));
    }
    return out;
}

// Constrained-generation baseline: force {"answer":"<one label>"}.
inline std::string json_answer_gbnf(const std::vector<std::string>& labels) {
    std::string gbnf =
        "root ::= \"{\" space \"\\\"answer\\\"\" space \":\" space answer space \"}\"\n";
    gbnf += "space ::= [ \\t\\n]*\nanswer ::= ";
    for (size_t i = 0; i < labels.size(); ++i) {
        if (i)
            gbnf += " | ";
        gbnf += "\"\\\"";
        gbnf += gbnf_escape(labels[i]);
        gbnf += "\\\"\"";
    }
    gbnf += "\n";
    return gbnf;
}

inline bool extract_json_answer(const std::string& text, std::string* answer) {
    if (!answer)
        return false;
    const auto key = text.find("\"answer\"");
    if (key == std::string::npos)
        return false;
    size_t pos = key + 8;
    while (pos < text.size() && (text[pos] == ' ' || text[pos] == '\t' || text[pos] == ':'))
        ++pos;
    if (pos >= text.size() || text[pos] != '"')
        return false;
    ++pos;
    return json_read_string(text, pos, *answer);
}

inline bool noul_answers_match(std::string expected, std::string predicted) {
    auto norm = [](std::string s) {
        for (char& c : s)
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (s == "true")
            return std::string("yes");
        if (s == "false")
            return std::string("no");
        return s;
    };
    return norm(std::move(expected)) == norm(std::move(predicted));
}

inline bool decision_answers_match(const DecisionQuestion& q, const std::string& expected,
                                   const std::string& predicted) {
    if (q.type == "noul")
        return noul_answers_match(expected, predicted);
    if (expected == predicted)
        return true;
    for (size_t i = 0; i < q.option_names.size(); ++i) {
        if (q.option_names[i] == predicted && q.option_descs[i] == expected)
            return true;
        if (q.option_descs[i] == predicted && q.option_names[i] == expected)
            return true;
    }
    return false;
}

inline float h2o_temperature_for(const std::string& type) {
    if (type == "noul")
        return kH2oTempNoul;
    if (type == "score")
        return kH2oTempScore;
    return kH2oTempChoice;
}

inline float apply_noul_floor(float p, float floor) {
    if (floor <= 0.5f)
        return p;
    const float lo = 1.0f - floor;
    if (std::max(p, 1.0f - p) < floor)
        return p >= 0.5f ? floor : lo;
    return p;
}

// #30361 converter: add_decision_labels([" " + label for label in cfg["labels"]]).
// The answer follows "Answer:", so the scored token has a leading space.
inline std::string h2o_leading_space_label(std::string_view label) {
    if (label.empty())
        return " ";
    if (label.front() == ' ')
        return std::string(label);
    return " " + std::string(label);
}

// systemone template uses `o.label | trim` in the user turn.
inline std::string h2o_trim_label(std::string_view label) {
    size_t i = 0;
    while (i < label.size() &&
           (label[i] == ' ' || label[i] == '\t' || label[i] == '\n' || label[i] == '\r'))
        ++i;
    return std::string(label.substr(i));
}

// #30361 h2o_cut_state (server-decision.cpp ~725-745). Byte-level BPE never
// has more tokens than bytes, so text.size() <= 32000 is kept verbatim. When
// token pieces are supplied and exceed the cap, keep head/tail halves around
// a "..." line. Empty pieces after the byte check leave the text unchanged
// (caller could not tokenize; documented limitation).
inline std::string cut_h2o_state(const std::string& text, const std::vector<std::string>& pieces) {
    if (text.size() <= kH2oMaxStateTokens)
        return text;
    if (pieces.size() <= kH2oMaxStateTokens)
        return text;
    if (pieces.empty())
        return text;
    const size_t half = kH2oMaxStateTokens / 2;
    std::string head;
    std::string tail;
    for (size_t i = 0; i < half && i < pieces.size(); ++i)
        head += pieces[i];
    const size_t tail_begin = pieces.size() - half;
    for (size_t i = tail_begin; i < pieces.size(); ++i)
        tail += pieces[i];
    return head + "\n...\n" + tail;
}

// One option object from #30361 render_h2o (server-decision.cpp ~811-815):
// {"label": <GGUF/serve text>, "key": <option name>, "description": <text>}.
inline std::string render_h2o_option_json(const std::string& label, const std::string& key,
                                          const std::string& description) {
    return "{\"label\":\"" + json_escape(label) + "\",\"key\":\"" + json_escape(key) +
           "\",\"description\":\"" + json_escape(description) + "\"}";
}

inline std::string render_h2o_options_json(const DecisionQuestion& q,
                                           const std::vector<std::string>& labels) {
    std::string out = "[";
    const size_t n = q.option_names.size();
    for (size_t i = 0; i < n; ++i) {
        if (i)
            out += ",";
        const std::string label = i < labels.size() ? labels[i] : std::to_string(i);
        const std::string desc = i < q.option_descs.size() ? q.option_descs[i] : q.option_names[i];
        out += render_h2o_option_json(label, q.option_names[i], desc);
    }
    out += "]";
    return out;
}

// systemone user turn after the template consumes render_h2o JSON
// (PR #30361 converter _systemone_template: `{{ o.label | trim }}) {{ o.key }}: …`).
inline std::string render_decision_user(const std::string& state, const DecisionQuestion& q,
                                        const std::vector<std::string>& labels) {
    std::string body;
    const size_t n = q.option_names.size();
    for (size_t i = 0; i < n; ++i) {
        const std::string shown = h2o_trim_label(i < labels.size() ? labels[i] : std::to_string(i));
        const std::string desc = i < q.option_descs.size() ? q.option_descs[i] : q.option_names[i];
        body += shown;
        body += ") ";
        body += q.option_names[i];
        body += ": ";
        body += desc;
        if (i + 1 < n)
            body += "\n";
    }
    return "record: " + state + "\nquestion: " + q.instructions + "\noptions:\n" + body;
}

inline std::string render_decision_prompt(const std::string& state, const DecisionQuestion& q,
                                          const std::vector<std::string>& labels,
                                          const DecisionPromptSpec& spec = {}) {
    const std::string user = render_decision_user(state, q, labels);
    if (!spec.chat_template) {
        std::string out;
        if (!spec.system.empty())
            out += spec.system + "\n\n";
        out += user;
        out += spec.prefill;
        return out;
    }
    // #30361 converter _systemone_template (PR diff, H2OLightningModel):
    //   <|im_start|>system\n{system}<|im_end|>\n
    //   <|im_start|>user\n{user | trim}<|im_end|>\n
    //   <|im_start|>assistant\n + " {{ '\\n\\n' }} {{ '\\n\\n' }}" + prefill
    // which renders as a space, two newlines, two spaces, two newlines, then
    // "Answer:". The stock+shim ChatML path writes <think>\\n\\n</think> here
    // instead; that mismatch is a known limitation of the no-bump pin.
    std::string out;
    if (!spec.system.empty()) {
        out += "<|im_start|>system\n";
        out += spec.system;
        out += "<|im_end|>\n";
    }
    out += "<|im_start|>user\n";
    out += user;
    out += "<|im_end|>\n<|im_start|>assistant\n \n\n  \n\n";
    out += spec.prefill;
    return out;
}

inline std::string render_json_decision_prompt(const std::string& state,
                                               const DecisionQuestion& q) {
    std::string out = "Answer with JSON only: {\"answer\":\"<option>\"}.\nOptions:\n";
    for (size_t i = 0; i < q.option_names.size(); ++i) {
        out += "- ";
        out += q.option_names[i];
        if (i < q.option_descs.size() && q.option_descs[i] != q.option_names[i]) {
            out += ": ";
            out += q.option_descs[i];
        }
        out += "\n";
    }
    out += "Record: ";
    out += state;
    out += "\nQuestion: ";
    out += q.instructions;
    out += "\n";
    return out;
}

// Nearest-rank percentile. Empty → 0.
inline double percentile_ms(std::vector<double> samples, double p) {
    if (samples.empty())
        return 0.0;
    std::sort(samples.begin(), samples.end());
    const double clamped = std::min(100.0, std::max(0.0, p));
    const size_t idx = static_cast<size_t>(
        std::min<double>(static_cast<double>(samples.size() - 1),
                         std::ceil(clamped / 100.0 * static_cast<double>(samples.size())) - 1.0));
    return samples[idx];
}

inline std::string pending_or(const std::string& value) {
    return value.empty() ? "pending" : value;
}

const std::vector<std::string>& h2o_lightning_letter_labels();

// Read general.architecture, {arch}.decision.type and {arch}.decision.labels
// through the gguf API (gguf_init_from_file, no_alloc). Does not call
// llama_model_decision_label — that symbol is not on pin d812350.
bool read_gguf_decision_meta(const std::string& path, GgufDecisionMeta& out,
                             std::string* err = nullptr);

// GGUF labels when present; otherwise serve_config letters with a leading space.
std::vector<std::string> h2o_score_labels_for(size_t n, const GgufDecisionMeta& meta);

bool parse_jevbench_line(const std::string& line, JevBenchItem& out, std::string* err = nullptr);
bool load_jevbench_dir(const std::string& dir, std::vector<JevBenchItem>& out,
                       std::string* err = nullptr);
bool load_decision_job_file(const std::string& path, DecisionJob& out, std::string* err = nullptr);
std::string format_decision_report_json(const DecisionRunReport& report);
bool write_decision_report(const std::string& path, const DecisionRunReport& report,
                           std::string* err = nullptr);

// One forward pass + label softmax. Implemented for the llama.cpp session.
class Session;
LabelScoreResult score_labels_on_session(Session& session, const DecisionScoreRequest& req);

// Load one session and score or generate-json every public JevBench item.
DecisionRunReport run_decision_job(const DecisionJob& job);

} // namespace xllama
