// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include "xllama/decision_score.h"

#include "xllama/chat_prompt.h"
#include "xllama/grammar_spec.h"
#include "xllama/json_utils.h"
#include "xllama/path_utils.h"
#include "xllama/platform.h"
#include "xllama/session.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <gguf.h>
#include <nlohmann/json.hpp>
#include <sstream>

namespace xllama {
namespace {

using ojson = nlohmann::ordered_json;

const std::vector<std::string> kH2oLetters = {
    "A",   "B",   "C",   "D",   "E",   "F",   "G",   "H",   "I",   "J",   "K",   "L",   "M",
    "N",   "O",   "P",   "Q",   "R",   "S",   "T",   "U",   "V",   "W",   "X",   "Y",   "Z",
    "a",   "b",   "c",   "d",   "e",   "f",   "g",   "h",   "i",   "j",   "k",   "l",   "m",
    "n",   "o",   "p",   "q",   "r",   "s",   "t",   "u",   "v",   "w",   "x",   "y",   "z",
    "α",   "β",   "γ",   "δ",   "ε",   "ζ",   "η",   "θ",   "ι",   "κ",   "λ",   "μ",   "ν",
    "ξ",   "ο",   "π",   "σ",   "τ",   "υ",   "φ",   "χ",   "ψ",   "ω",   "in",  "er",  "on",
    "re",  "at",  "st",  "en",  "or",  "th",  "le",  "it",  "an",  "ar",  "al",  "the", "ou",
    "is",  "ing", "es",  "ion", "ed",  "ic",  "et",  "ro",  "as",  "el",  "ct",  "nd",  "ent",
    "id",  "am",  "to",  "of",  "om",  "im",  "il",  "and", "ur",  "se",  "ex",  "ad",  "ch",
    "ut",  "if",  "em",  "ol",  "ig",  "iv",  "ce",  "od",  "ate", "ag",  "ay",  "ot",  "us",
    "un",  "ul",  "ue",  "ow",  "ew",  "for", "ab",  "ort", "um",  "ame", "pe",  "tr",  "ck",
    "ist", "he",  "lo",  "be",  "ers", "con", "ap",  "ub",  "ass", "int", "ly",  "urn", "av",
    "ir",  "nt",  "end", "de",  "ith", "out", "our", "lic", "res", "pt",  "wh",  "ver", "age",
    "ht",  "ext", "ess", "os",  "ect", "ke",  "rom", "qu",  "iz",  "op",  "up",  "get", "ile",
    "ata", "ore", "ri",  "pro", "ter", "ain", "com", "art", "ack", "est", "ine", "ill", "ind",
    "ere", "elf", "orm", "ult", "str", "you", "pl",  "new", "ld",  "oc",  "ant", "ase", "ave",
    "ang", "are", "ial", "act", "ive", "ode", "ost", "not", "og",  "ord", "all", "ff",  "ont",
    "ime", "pr",  "ies", "ure", "by",  "ire", "sh",  "ice", "ast", "ok",  "cl",  "ard", "was",
    "ip",  "The", "ide", "ace", "ib",  "ac",  "rou", "we",  "ak",  "ve",  "ath", "oid", "ust",
    "que", "ans", "pp",  "one", "ail", "ach", "div", "ell",
};

std::string json_string_field(const ojson& obj, const char* key) {
    if (!obj.contains(key) || !obj[key].is_string())
        return {};
    return obj[key].get<std::string>();
}

bool fill_question(const ojson& qj, DecisionQuestion& q, std::string* err) {
    q.type = json_string_field(qj, "type");
    if (q.type.empty())
        q.type = "choice";
    if (qj.contains("instructions")) {
        if (qj["instructions"].is_string())
            q.instructions = qj["instructions"].get<std::string>();
        else
            q.instructions = qj["instructions"].dump();
    }
    const ojson criteria = qj.contains("criteria") ? qj["criteria"] : ojson();
    if (q.type == "noul") {
        q.option_names = {"true", "false"};
        std::string true_desc = "the statement holds";
        std::string false_desc = "it does not";
        if (criteria.is_object()) {
            if (criteria.contains("true") && criteria["true"].is_string())
                true_desc = criteria["true"].get<std::string>();
            else if (criteria.contains("yes") && criteria["yes"].is_string())
                true_desc = criteria["yes"].get<std::string>();
            if (criteria.contains("false") && criteria["false"].is_string())
                false_desc = criteria["false"].get<std::string>();
            else if (criteria.contains("no") && criteria["no"].is_string())
                false_desc = criteria["no"].get<std::string>();
        }
        q.option_descs = {true_desc, false_desc};
        return true;
    }
    if (q.type == "score") {
        if (!criteria.is_array() || criteria.empty()) {
            if (err)
                *err = "score criteria must be a non-empty array";
            return false;
        }
        for (size_t i = 0; i < criteria.size(); ++i) {
            q.option_names.push_back(std::to_string(i));
            if (criteria[i].is_string())
                q.option_descs.push_back(criteria[i].get<std::string>());
            else
                q.option_descs.push_back(criteria[i].dump());
        }
        return true;
    }
    if (!criteria.is_object() || criteria.empty()) {
        if (err)
            *err = "choice criteria must be a non-empty object";
        return false;
    }
    for (auto it = criteria.begin(); it != criteria.end(); ++it) {
        q.option_names.push_back(it.key());
        if (it.value().is_string())
            q.option_descs.push_back(it.value().get<std::string>());
        else if (it.value().is_null())
            q.option_descs.push_back(it.key());
        else
            q.option_descs.push_back(it.value().dump());
    }
    return true;
}

std::string read_file(const std::string& path, std::string* err) {
    std::ifstream in(path);
    if (!in) {
        if (err)
            *err = "cannot read " + path;
        return {};
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

} // namespace

const std::vector<std::string>& h2o_lightning_letter_labels() {
    return kH2oLetters;
}

bool read_gguf_decision_meta(const std::string& path, GgufDecisionMeta& out, std::string* err) {
    out = {};
    const std::string resolved = first_gguf_in_dir(resolve_model_path(path), {});
    const std::string file = resolved.empty() ? path : resolved;
    gguf_init_params params{};
    params.no_alloc = true;
    params.ctx = nullptr;
    gguf_context* ctx = gguf_init_from_file(file.c_str(), params);
    if (!ctx) {
        out.error = "gguf_init_from_file failed: " + file;
        if (err)
            *err = out.error;
        return false;
    }

    const int64_t arch_id = gguf_find_key(ctx, "general.architecture");
    if (arch_id >= 0 && gguf_get_kv_type(ctx, arch_id) == GGUF_TYPE_STRING) {
        const char* arch = gguf_get_val_str(ctx, arch_id);
        if (arch)
            out.architecture = arch;
    }
    if (out.architecture.empty()) {
        gguf_free(ctx);
        out.error = "GGUF missing general.architecture";
        if (err)
            *err = out.error;
        return false;
    }

    const std::string type_key = out.architecture + ".decision.type";
    const int64_t type_id = gguf_find_key(ctx, type_key.c_str());
    if (type_id >= 0 && gguf_get_kv_type(ctx, type_id) == GGUF_TYPE_STRING) {
        const char* type = gguf_get_val_str(ctx, type_id);
        if (type)
            out.decision_type = type;
    }

    const std::string labels_key = out.architecture + ".decision.labels";
    const int64_t labels_id = gguf_find_key(ctx, labels_key.c_str());
    if (labels_id >= 0 && gguf_get_kv_type(ctx, labels_id) == GGUF_TYPE_ARRAY &&
        gguf_get_arr_type(ctx, labels_id) == GGUF_TYPE_STRING) {
        const size_t n = gguf_get_arr_n(ctx, labels_id);
        out.labels.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            const char* lab = gguf_get_arr_str(ctx, labels_id, i);
            out.labels.emplace_back(lab ? lab : "");
        }
        out.has_decision_labels = true;
        out.label_source = "gguf";
    } else {
        out.label_source = "serve_config-fallback";
    }

    gguf_free(ctx);
    out.ok = true;
    return true;
}

std::vector<std::string> h2o_score_labels_for(size_t n, const GgufDecisionMeta& meta) {
    std::vector<std::string> out;
    if (n == 0 || n > kH2oMaxOptions)
        return out;
    if (meta.has_decision_labels && meta.labels.size() >= n) {
        out.assign(meta.labels.begin(), meta.labels.begin() + static_cast<std::ptrdiff_t>(n));
        return out;
    }
    const auto& letters = h2o_lightning_letter_labels();
    if (n > letters.size())
        return out;
    out.reserve(n);
    for (size_t i = 0; i < n; ++i)
        out.push_back(h2o_leading_space_label(letters[i]));
    return out;
}

bool parse_jevbench_line(const std::string& line, JevBenchItem& out, std::string* err) {
    out = {};
    try {
        const ojson j = ojson::parse(line);
        out.id = json_string_field(j, "id");
        out.family = json_string_field(j, "family");
        out.split = json_string_field(j, "split");
        if (j.contains("expected")) {
            if (j["expected"].is_string())
                out.expected = j["expected"].get<std::string>();
            else
                out.expected = j["expected"].dump();
        }
        if (j.contains("state")) {
            if (j["state"].is_string())
                out.state = j["state"].get<std::string>();
            else
                out.state = j["state"].dump();
        }
        if (!j.contains("question") || !j["question"].is_object()) {
            if (err)
                *err = "missing question object";
            return false;
        }
        if (!fill_question(j["question"], out.question, err))
            return false;
        if (out.question.type == "choice" && j.contains("labels") && j["labels"].is_array() &&
            !j["labels"].empty()) {
            std::vector<std::string> names;
            std::vector<std::string> descs;
            std::vector<std::string> from_file;
            for (const auto& lab : j["labels"]) {
                if (lab.is_string())
                    from_file.push_back(lab.get<std::string>());
            }
            for (const auto& name : from_file) {
                auto it = std::find(out.question.option_names.begin(),
                                    out.question.option_names.end(), name);
                names.push_back(name);
                if (it != out.question.option_names.end()) {
                    const size_t idx = static_cast<size_t>(it - out.question.option_names.begin());
                    descs.push_back(out.question.option_descs[idx]);
                } else {
                    descs.push_back(name);
                }
            }
            if (!names.empty()) {
                out.question.option_names = std::move(names);
                out.question.option_descs = std::move(descs);
            }
        }
        return true;
    } catch (const std::exception& e) {
        if (err)
            *err = e.what();
        return false;
    }
}

bool load_jevbench_dir(const std::string& dir, std::vector<JevBenchItem>& out, std::string* err) {
    out.clear();
    const char* files[] = {"easy.jsonl", "original.jsonl", "hard.jsonl"};
    for (const char* name : files) {
        const std::filesystem::path p = std::filesystem::path(dir) / name;
        if (!std::filesystem::exists(p)) {
            if (err)
                *err = "missing " + p.string();
            return false;
        }
        std::ifstream in(p);
        if (!in) {
            if (err)
                *err = "cannot read " + p.string();
            return false;
        }
        std::string line;
        size_t lineno = 0;
        while (std::getline(in, line)) {
            ++lineno;
            if (line.empty() || line[0] == '#')
                continue;
            JevBenchItem item;
            std::string perr;
            if (!parse_jevbench_line(line, item, &perr)) {
                if (err)
                    *err = p.string() + ":" + std::to_string(lineno) + ": " + perr;
                return false;
            }
            out.push_back(std::move(item));
        }
    }
    return true;
}

bool load_decision_job_file(const std::string& path, DecisionJob& out, std::string* err) {
    out = {};
    std::string body = read_file(path, err);
    if (body.empty())
        return false;
    try {
        const nlohmann::json j = nlohmann::json::parse(body);
        if (j.contains("model_path"))
            out.model_path = j["model_path"].get<std::string>();
        if (j.contains("questions_dir"))
            out.questions_dir = j["questions_dir"].get<std::string>();
        if (j.contains("out_path"))
            out.out_path = j["out_path"].get<std::string>();
        if (j.contains("mode"))
            out.mode = j["mode"].get<std::string>();
        if (j.contains("backend"))
            out.backend = j["backend"].get<std::string>();
        if (j.contains("n_gpu_layers"))
            out.n_gpu_layers = j["n_gpu_layers"].get<int>();
        if (j.contains("n_threads"))
            out.n_threads = j["n_threads"].get<int>();
        if (j.contains("n_ctx"))
            out.n_ctx = j["n_ctx"].get<int>();
        if (j.contains("temperature"))
            out.temperature = j["temperature"].get<float>();
        if (j.contains("apply_noul_floor"))
            out.apply_noul_floor = j["apply_noul_floor"].get<bool>();
        if (j.contains("noul_floor"))
            out.noul_floor = j["noul_floor"].get<float>();
        if (j.contains("host_label"))
            out.host_label = j["host_label"].get<std::string>();
        if (j.contains("build_id"))
            out.build_id = j["build_id"].get<std::string>();
        if (j.contains("git_commit"))
            out.git_commit = j["git_commit"].get<std::string>();
        if (j.contains("device"))
            out.device = j["device"].get<std::string>();
        if (out.model_path.empty()) {
            if (err)
                *err = "job.model_path is required";
            return false;
        }
        if (out.questions_dir.empty()) {
            if (err)
                *err = "job.questions_dir is required";
            return false;
        }
        if (out.mode != "score" && out.mode != "generate-json") {
            if (err)
                *err = "job.mode must be score or generate-json";
            return false;
        }
        return true;
    } catch (const std::exception& e) {
        if (err)
            *err = e.what();
        return false;
    }
}

std::string format_decision_report_json(const DecisionRunReport& report) {
    nlohmann::json j;
    j["ok"] = report.ok;
    j["error"] = report.error;
    j["mode"] = report.mode;
    j["model_path"] = report.model_path;
    j["backend"] = report.backend;
    j["n_gpu_layers"] = report.n_gpu_layers;
    j["host_label"] = pending_or(report.host_label);
    j["build_id"] = pending_or(report.build_id);
    j["git_commit"] = pending_or(report.git_commit);
    j["device"] = pending_or(report.device);
    j["n_items"] = report.n_items;
    j["n_correct"] = report.n_correct;
    j["n_errors"] = report.n_errors;
    if (report.n_items > 0) {
        j["accuracy"] = report.accuracy;
        j["latency_p50_ms"] = report.latency_p50_ms;
        j["latency_p95_ms"] = report.latency_p95_ms;
    } else {
        j["accuracy"] = nullptr;
        j["latency_p50_ms"] = nullptr;
        j["latency_p95_ms"] = nullptr;
    }
    j["t_load_ms"] = report.t_load_ms;
    j["peak_ws_mb"] = report.peak_ws_mb;
    j["vendor_claimed"] = {
        {"label", report.vendor_claimed_label},
        {"correct", report.vendor_claimed_correct},
        {"total", report.vendor_claimed_total},
        {"note", "H2O-Lightning-4B Q4_K_M GGUF card. Not an xllama measurement."}};
    j["gguf_architecture"] = report.gguf_architecture;
    j["decision_type"] = report.decision_type;
    j["label_source"] = report.label_source;
    j["n_gguf_labels"] = report.n_gguf_labels;
    nlohmann::json items = nlohmann::json::array();
    for (const auto& it : report.items) {
        nlohmann::json ji;
        ji["id"] = it.id;
        ji["family"] = it.family;
        ji["expected"] = it.expected;
        ji["predicted"] = it.predicted;
        ji["correct"] = it.correct;
        ji["ok"] = it.ok;
        ji["error"] = it.error;
        ji["latency_ms"] = it.latency_ms;
        nlohmann::json sc = nlohmann::json::array();
        for (const auto& s : it.scores) {
            sc.push_back({{"label", s.label},
                          {"first_token", s.first_token},
                          {"token_count", s.token_count},
                          {"logit", s.logit},
                          {"probability", s.probability}});
        }
        ji["scores"] = std::move(sc);
        items.push_back(std::move(ji));
    }
    j["items"] = std::move(items);
    return j.dump(2) + "\n";
}

bool write_decision_report(const std::string& path, const DecisionRunReport& report,
                           std::string* err) {
    std::ofstream out(path);
    if (!out) {
        if (err)
            *err = "cannot write " + path;
        return false;
    }
    out << format_decision_report_json(report);
    return static_cast<bool>(out);
}

LabelScoreResult score_labels_on_session(Session& session, const DecisionScoreRequest& req) {
    return session.score_labels(req);
}

DecisionRunReport run_decision_job(const DecisionJob& job) {
    DecisionRunReport report;
    report.mode = job.mode;
    report.model_path = job.model_path;
    report.backend = job.backend;
    report.n_gpu_layers = job.n_gpu_layers;
    report.host_label = pending_or(job.host_label);
    report.build_id = pending_or(job.build_id);
    report.git_commit = pending_or(job.git_commit);
    report.device = pending_or(job.device);

    std::vector<JevBenchItem> items;
    std::string err;
    if (!load_jevbench_dir(job.questions_dir, items, &err)) {
        report.error = err;
        return report;
    }
    report.n_items = static_cast<int>(items.size());

    GgufDecisionMeta meta;
    std::string meta_err;
    if (read_gguf_decision_meta(job.model_path, meta, &meta_err) && meta.ok) {
        report.gguf_architecture = meta.architecture;
        report.decision_type = meta.decision_type;
        report.label_source = meta.label_source;
        report.n_gguf_labels = static_cast<int>(meta.labels.size());
    } else {
        report.label_source = "serve_config-fallback";
        if (report.error.empty() && !meta_err.empty())
            report.gguf_architecture = "";
    }

    SessionParams sp;
    sp.model_path = job.model_path;
    sp.n_ctx = job.n_ctx;
    sp.n_threads = job.n_threads;
    sp.n_gpu_layers = job.n_gpu_layers;
    sp.backend = Backend::LlamaCpp;
    sp.dml_warmup = false;

    const auto t0 = std::chrono::steady_clock::now();
    std::unique_ptr<Session> session = Session::create(sp, &err);
    report.t_load_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (!session) {
        report.error = err.empty() ? "session create failed" : err;
        return report;
    }

    const ChatFormat fmt = chat_format_for(job.model_path);
    std::vector<double> latencies;
    latencies.reserve(items.size());

    for (const auto& item : items) {
        DecisionItemResult row;
        row.id = item.id;
        row.family = item.family;
        row.expected = item.expected;
        const auto t_item = std::chrono::steady_clock::now();
        if (job.mode == "generate-json") {
            GenerateParams gp;
            gp.prompt = fmt.render_prompt(std::string(kDefaultSystemPrompt), {},
                                          render_json_decision_prompt(item.state, item.question));
            gp.stop_sequences = fmt.stop_sequences;
            gp.n_predict = 64;
            gp.temperature = 0.0f;
            gp.grammar.gbnf = json_answer_gbnf(item.question.option_names);
            const InferenceResult gen = session->generate(gp);
            row.ok = gen.success;
            row.error = gen.error_msg;
            if (gen.success) {
                std::string ans;
                if (extract_json_answer(gen.output_text, &ans))
                    row.predicted = ans;
                else
                    row.predicted = gen.output_text;
            }
        } else {
            const size_t n = item.question.option_names.size();
            if (n == 0 || n > kH2oMaxOptions) {
                row.error = "unsupported option count";
            } else {
                const std::vector<std::string> labels = h2o_score_labels_for(n, meta);
                if (labels.size() != n) {
                    row.error = "label table shorter than option count";
                } else {
                    std::string state = item.state;
                    if (state.size() > kH2oMaxStateTokens) {
                        const std::vector<std::string> pieces = session->token_pieces(state);
                        state = cut_h2o_state(state, pieces);
                    }
                    DecisionScoreRequest req;
                    req.prompt = render_decision_prompt(state, item.question, labels);
                    req.labels = labels;
                    req.temperature = job.temperature > 0.0f
                                          ? job.temperature
                                          : h2o_temperature_for(item.question.type);
                    const LabelScoreResult scored = session->score_labels(req);
                    row.ok = scored.ok;
                    row.error = scored.error;
                    row.scores = scored.scores;
                    if (scored.ok && scored.winner >= 0 &&
                        static_cast<size_t>(scored.winner) < item.question.option_names.size()) {
                        row.predicted =
                            item.question.option_names[static_cast<size_t>(scored.winner)];
                        if (job.apply_noul_floor && item.question.type == "noul" &&
                            scored.scores.size() >= 2) {
                            const float p_true =
                                apply_noul_floor(scored.scores[0].probability, job.noul_floor);
                            row.scores[0].probability = p_true;
                            row.scores[1].probability = 1.0f - p_true;
                        }
                    }
                }
            }
        }
        row.latency_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_item)
                .count();
        if (row.ok)
            latencies.push_back(row.latency_ms);
        else
            ++report.n_errors;
        row.correct = row.ok && decision_answers_match(item.question, item.expected, row.predicted);
        if (row.correct)
            ++report.n_correct;
        report.items.push_back(std::move(row));
    }

    report.accuracy =
        report.n_items > 0 ? static_cast<double>(report.n_correct) / report.n_items : 0.0;
    report.latency_p50_ms = percentile_ms(latencies, 50);
    report.latency_p95_ms = percentile_ms(latencies, 95);
    report.peak_ws_mb = peak_working_set_mb();
    report.ok = report.n_errors == 0 && !items.empty();
    if (!report.ok && report.error.empty() && report.n_errors > 0)
        report.error = std::to_string(report.n_errors) + " item errors";
    return report;
}

} // namespace xllama
