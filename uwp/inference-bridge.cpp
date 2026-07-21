// Copyright (c) 2024 Venere Labs
// SPDX-License-Identifier: MIT

#include "inference-bridge.h"

#include "xllama/chat_prompt.h"
#include "xllama/device_train.h"
#include "xllama/inference.h"
#include "xllama/membw.h"
#include "xllama/path_utils.h"
#include "xllama/platform.h"
#include "xllama/session.h"
#include "xllama/training.h"
#include "xllama/utf8_utils.h"

#include <cstdio>
#include <ctime>
#include <string>

namespace xllama::bridge {

#ifdef XLLAMA_UWP
namespace {

std::string read_local_file(const char* name) {
    std::string out;
    std::string p = resolve_local_path(name);
    FILE* f = _wfopen(utf8_to_wstring(p).c_str(), L"r");
    if (!f)
        return out;
    char buf[8192];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0)
        out.append(buf, n);
    fclose(f);
    while (!out.empty() && (out.back() == '\n' || out.back() == '\r' || out.back() == ' '))
        out.pop_back();
    return out;
}

// Small integer knob from a LocalState file; |fallback| when absent or unparsable.
int read_local_int(const char* name, int fallback) {
    const std::string s = read_local_file(name);
    return s.empty() ? fallback : std::atoi(s.c_str());
}

// Multi-turn TTFT bench: measures turn-2 prefill with KV reuse (append only the
// new turn) against the cold baseline (full re-prefill of the 2-turn context),
// on the same persistent Session. The ratio is the KV-reuse win. Writes
// bench-kv-result.csv (+ .done) and logs the numbers.
void run_kv_bench(const std::string& model_name, const std::string& sys, const std::string& u1,
                  const std::string& u2, int n_threads, int n_ctx, const char* host) {
    // KV-reuse now works on both backends: ORT-GenAI (persistent generator) and
    // GGUF/llama.cpp (persistent llama_context in LlamaSession). No early skip.
    std::string err;
    ::xllama::SessionParams sp;
    sp.model_path = model_name;
    sp.n_ctx = n_ctx > 0 ? n_ctx : 2048;
    sp.n_threads = n_threads;
    auto sess = ::xllama::Session::create(sp, &err);
    if (!sess) {
        log_output("[xllama] kv-bench: session create failed: " + err + "\n");
        return;
    }

    const ::xllama::ChatFormat fmt = ::xllama::chat_format_for(model_name);
    auto mkgp = [&](const std::string& p, bool reuse, bool reset) {
        ::xllama::GenerateParams gp;
        gp.prompt = p;
        gp.n_predict = 96;
        gp.reuse_kv = reuse;
        gp.reset_kv = reset;
        gp.stop_sequences = fmt.stop_sequences;
        return gp;
    };

    // Turn 1: seed the persistent generator.
    auto r1 = sess->generate(mkgp(fmt.render_prompt(sys, {}, u1), /*reuse=*/true, /*reset=*/true));
    // Turn 2 (KV reuse): append only the new turn's delta.
    std::string delta = fmt.render_delta(u2, r1.ended_with_stop);
    auto r2 = sess->generate(mkgp(delta, /*reuse=*/true, /*reset=*/false));
    // Turn 2 (cold): full re-prefill of the whole 2-turn context — the pre-Stage-2
    // behaviour. Uses turn-1's actual output so the token count matches.
    std::string full2 = fmt.render_prompt(sys, {::xllama::ChatTurn{u1, r1.output_text}}, u2);
    auto r2c = sess->generate(mkgp(full2, /*reuse=*/true, /*reset=*/true));

    double speedup = (r2.t_p_eval_ms > 0) ? r2c.t_p_eval_ms / r2.t_p_eval_ms : 0.0;
    char lb[320];
    snprintf(lb, sizeof(lb),
             "[xllama] kv-bench: turn2 prefill reuse=%.1fms (%d tok) cold=%.1fms (%d tok) "
             "speedup=%.2fx\n",
             r2.t_p_eval_ms, r2.n_p_eval, r2c.t_p_eval_ms, r2c.n_p_eval, speedup);
    log_output(lb);

    const std::string csv = resolve_local_path("bench-kv-result.csv");
    FILE* fp = _wfopen(utf8_to_wstring(csv).c_str(), L"w");
    if (fp) {
        time_t now = time(nullptr);
        char date_buf[32];
        strftime(date_buf, sizeof(date_buf), "%Y-%m-%dT%H:%M:%SZ", gmtime(&now));
        fputs("model,prefill1_ms,n_p1,prefill2_reuse_ms,n_p2_reuse,prefill2_cold_ms,n_p2_cold,"
              "speedup,decode_tok_s,n_ctx,host,date\n",
              fp);
        double dtok = (r2.n_eval > 0 && r2.t_eval_ms > 0) ? r2.n_eval / (r2.t_eval_ms / 1000.0) : 0;
        fprintf(fp, "%s,%.1f,%d,%.1f,%d,%.1f,%d,%.2f,%.2f,%d,%s,%s\n", model_name.c_str(),
                r1.t_p_eval_ms, r1.n_p_eval, r2.t_p_eval_ms, r2.n_p_eval, r2c.t_p_eval_ms,
                r2c.n_p_eval, speedup, dtok, sp.n_ctx, host ? host : "unknown", date_buf);
        fclose(fp);
        FILE* done =
            _wfopen(utf8_to_wstring(resolve_local_path("bench-kv-result.csv.done")).c_str(), L"w");
        if (done) {
            fputs("done\n", done);
            fclose(done);
        }
        log_output("[xllama] bench-kv-result.csv written\n");
    }
}

} // namespace
#endif // XLLAMA_UWP

// ---------------------------------------------------------------------------
// main_loop (called from UWP bench mode background thread)
// ---------------------------------------------------------------------------

void main_loop() {
#ifdef XLLAMA_UWP
    // Pin CWD to LocalState so relative paths from genai_config.json (e.g. the
    // ORT enable_profiling prefix) land in a writable, WDP-fetchable location.
    set_cwd_to_local_folder();

    // Read prompt from LocalFolder/prompt.txt, fallback to default.
    // SmolLM2-360M-Instruct uses ChatML format; bare text triggers EOS immediately.
    // read_local_file reads to EOF; the hand-rolled reader this replaced used a
    // fixed char buf[8192] and truncated silently at ~2k tokens — exactly the
    // range a prompt-length sweep needs. The bench would then report a shorter
    // prompt's throughput under the long prompt's label with nothing in the log.
    std::string user_prompt = "Hello from Xbox Series S. Tell me about your architecture.";
    {
        std::string from_file = read_local_file("prompt.txt");
        if (!from_file.empty()) {
            user_prompt = std::move(from_file);
            log_output("[xllama] bench: prompt.txt " + std::to_string(user_prompt.size()) +
                       " bytes\n");
        }
    }
    // Read model directory/filename from LocalFolder/model.txt, fallback to default.
    std::string model_name = "smollm2-360m-cpu-int4";
    {
        std::string model_cfg = resolve_local_path("model.txt");
        FILE* mf = _wfopen(utf8_to_wstring(model_cfg).c_str(), L"r");
        if (mf) {
            char buf[512] = {};
            size_t n = fread(buf, 1, sizeof(buf) - 1, mf);
            fclose(mf);
            if (n > 0) {
                model_name = buf;
                while (!model_name.empty() &&
                       (model_name.back() == '\n' || model_name.back() == '\r' ||
                        model_name.back() == ' '))
                    model_name.pop_back();
            }
        }
    }

    // Optional numeric knobs from LocalState, written by the bench scripts.
    // bench_threads.txt also labels the host column; bench_ctx.txt and
    // bench_npredict.txt exist because #130 needs to vary n_ctx and n_predict:
    // the DirectML prefill band's edges sit near n_ctx/2 and n_ctx - n_predict,
    // and that hypothesis is only falsifiable if both are controllable here.
    const int bench_threads = read_local_int("bench_threads.txt", 0);
    const int bench_ctx = read_local_int("bench_ctx.txt", 0);
    const int bench_npredict = read_local_int("bench_npredict.txt", 0);
    // #130: max_length is the variable that governs DirectML prefill, and it is
    // normally derived from n_predict. This decouples them. 0 = derive,
    // -1 = saturate to n_ctx (what the shipping app does).
    const int bench_maxlen = read_local_int("bench_maxlen.txt", 0);

    // Multi-turn TTFT bench (Stage 2b): if bench_turns.txt is present it holds the
    // turn-2 user prompt; prompt.txt supplies turn 1. Measures the KV-reuse win
    // and skips the normal single-turn bench.
    {
        std::string turn2 = read_local_file("bench_turns.txt");
        if (!turn2.empty()) {
            char host_buf2[64];
            if (bench_threads > 0)
                snprintf(host_buf2, sizeof(host_buf2), "xbox-series-s-t%d", bench_threads);
            else
                snprintf(host_buf2, sizeof(host_buf2), "xbox-series-s");
            log_output("[xllama] kv-bench model: " + model_name + "\n");
            run_kv_bench(model_name, "You are a helpful AI assistant.", user_prompt, turn2,
                         bench_threads, bench_ctx, host_buf2);
            return;
        }
    }

    // Apply the per-model chat template (ChatML default, required for
    // SmolLM2-Instruct; Gemma/Qwen get their own format via the model name).
    const ::xllama::ChatFormat fmt = ::xllama::chat_format_for(model_name);
    std::string prompt = fmt.render_prompt("You are a helpful AI assistant.", {}, user_prompt);

    log_output("[xllama] bench model: " + model_name + "\n");
    log_output("[xllama] bench prompt: " + prompt.substr(0, 80) + "...\n");

    InferenceParams params;
    params.model_path = model_name;
    params.prompt = prompt;
    // 0 = keep the default (n_predict 512, n_ctx from InferenceParams).
    params.n_predict = bench_npredict > 0 ? bench_npredict : 512;
    if (bench_ctx > 0)
        params.n_ctx = bench_ctx;
    params.max_length_override = bench_maxlen;
    params.n_threads = bench_threads;           // 0 = auto; set by bench-xbox-ort.sh
    params.stop_sequences = fmt.stop_sequences; // clean stop for Gemma's <end_of_turn>

    char host_buf[64];
    if (bench_threads > 0)
        snprintf(host_buf, sizeof(host_buf), "xbox-series-s-t%d", bench_threads);
    else
        snprintf(host_buf, sizeof(host_buf), "xbox-series-s");

    InferenceResult res = ::xllama::run_inference(params);
    xllama::write_bench_csv(params, res, host_buf);
#endif
}

// ---------------------------------------------------------------------------
// run_logits (called from UWP logits.flag mode background thread)
// ---------------------------------------------------------------------------

void run_logits() {
#ifdef XLLAMA_UWP
    set_cwd_to_local_folder();

    // Raw prompt (NOT chat-templated): parity feeds the identical string to both
    // backends so scripts/compare-logits.py can diff the resulting distributions.
    std::string prompt = read_local_file("prompt.txt");
    if (prompt.empty())
        prompt = "Hello from Xbox Series S. Tell me about your architecture.";
    std::string model_name = read_local_file("model.txt");
    if (model_name.empty())
        model_name = "smollm2-360m-cpu-int4";

    log_output("[xllama] logits model: " + model_name + "\n");
    log_output("[xllama] logits prompt: " + prompt.substr(0, 80) + "\n");

    InferenceParams params;
    params.model_path = model_name;
    params.prompt = prompt;
    params.n_predict = 1; // one deterministic forward pass; we only need prefill logits
    params.greedy = true;
    params.dump_logits_path = resolve_local_path("logits.bin");

    InferenceResult res = ::xllama::run_inference(params);
    log_output(res.success ? "[xllama] logits dump OK\n"
                           : "[xllama] logits dump FAILED: " + res.error_msg + "\n");

    // Completion marker for scripts/validate-logit-parity.sh to poll (WDP).
    FILE* done = _wfopen(utf8_to_wstring(resolve_local_path("logits.done")).c_str(), L"w");
    if (done) {
        fputs(res.success ? "ok" : "fail", done);
        fclose(done);
    }
#endif
}

// ---------------------------------------------------------------------------
// run_membw (called from UWP membw.flag mode background thread)
// ---------------------------------------------------------------------------

void run_membw() {
#ifdef XLLAMA_UWP
    log_output("[xllama] membw: measuring CPU memory bandwidth\n");
    // Two passes: single-thread and full-width, so the scaling ratio is visible.
    // 256 MB working set overflows the LLC → measures DRAM, not cache.
    const std::size_t buf = static_cast<std::size_t>(256) << 20;
    const ::xllama::MembwResult st = ::xllama::measure_membw(buf, 5, 1);
    const ::xllama::MembwResult mt = ::xllama::measure_membw(buf, 5, 0);

    char lb[256];
    snprintf(lb, sizeof(lb),
             "[xllama] membw: 1t read=%.1f copy=%.1f triad=%.1f | %dt read=%.1f copy=%.1f "
             "triad=%.1f GB/s\n",
             st.read_gbs, st.copy_gbs, st.triad_gbs, mt.threads, mt.read_gbs, mt.copy_gbs,
             mt.triad_gbs);
    log_output(lb);

    const std::string csv = resolve_local_path("membw-result.csv");
    FILE* fp = _wfopen(utf8_to_wstring(csv).c_str(), L"w");
    if (fp) {
        fputs(::xllama::membw_csv_header(), fp);
        fputs(::xllama::format_membw_row(st, "xbox-series-s-t1").c_str(), fp);
        fputs(::xllama::format_membw_row(mt, "xbox-series-s").c_str(), fp);
        fclose(fp);
        FILE* done =
            _wfopen(utf8_to_wstring(resolve_local_path("membw-result.csv.done")).c_str(), L"w");
        if (done) {
            fputs("done\n", done);
            fclose(done);
        }
        log_output("[xllama] membw-result.csv written\n");
    }
#endif
}

// ---------------------------------------------------------------------------
// run_train (called from UWP train.flag mode background thread)
// ---------------------------------------------------------------------------

void run_train() {
#ifdef XLLAMA_UWP
    bool ok = false;
    std::string err;
    #ifdef XLLAMA_DEVICE_TRAIN
    const std::string job_path = resolve_local_path("training/job.json");
    ::xllama::TrainingJob job;
    ok = ::xllama::load_training_job_file(job_path, job, &err);
    if (ok) {
        // Job paths are LocalState-relative on console (the process cwd is the
        // read-only install dir). Absolute paths pass through untouched.
        auto localize = [](std::string& p) {
            if (!p.empty() && p.find(':') == std::string::npos && p[0] != '\\' && p[0] != '/')
                p = resolve_local_path(p);
        };
        localize(job.base_model);
        localize(job.dataset_path);
        localize(job.out_dir);

        log_output(
            ("[xllama] train: " + ::xllama::format_training_job_summary(job) + "\n").c_str());
        ::xllama::DeviceTrainCallbacks cb;
        cb.on_status = [](const std::string& line) {
            log_output(("[xllama] train: " + line + "\n").c_str());
        };
        cb.on_progress = [](const ::xllama::DeviceTrainProgress& p) {
            char lb[160];
            snprintf(lb, sizeof(lb), "[xllama] train: epoch %d/%d batch %lld/%lld loss=%.4f\n",
                     p.epoch, p.epochs, static_cast<long long>(p.ibatch),
                     static_cast<long long>(p.ibatch_max), p.loss);
            log_output(lb);
        };
        const ::xllama::TrainingResult r = ::xllama::run_device_train_job(job, cb);
        ok = r.success;
        if (!ok)
            err = r.error_msg;
    }
    if (!ok)
        log_output(("[xllama] train FAIL: " + err + "\n").c_str());

    #else
    err = "built without XLLAMA_DEVICE_TRAIN";
    log_output(("[xllama] train FAIL: " + err + "\n").c_str());
    #endif
    FILE* done = _wfopen(utf8_to_wstring(resolve_local_path("training/result.done")).c_str(), L"w");
    if (done) {
        fputs(ok ? "ok\n" : "fail\n", done);
        fclose(done);
    }
#endif
}

} // namespace xllama::bridge
