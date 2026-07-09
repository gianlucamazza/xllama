// Copyright (c) 2024 Venere Labs
// SPDX-License-Identifier: MIT

#include "inference-bridge.h"

#include "xllama/inference.h"
#include "xllama/path_utils.h"
#include "xllama/platform.h"
#include "xllama/session.h"
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

std::string chatml(const std::string& sys, const std::string& user) {
    return "<|im_start|>system\n" + sys + "<|im_end|>\n<|im_start|>user\n" + user +
           "<|im_end|>\n<|im_start|>assistant\n";
}

// Multi-turn TTFT bench: measures turn-2 prefill with KV reuse (append only the
// new turn) against the cold baseline (full re-prefill of the 2-turn context),
// on the same persistent Session. The ratio is the KV-reuse win. Writes
// bench-kv-result.csv (+ .done) and logs the numbers.
void run_kv_bench(const std::string& model_name, const std::string& sys, const std::string& u1,
                  const std::string& u2, int n_threads, const char* host) {
    // KV-reuse bench is ORT GenAI specific (persistent generator + reuse_kv).
    // GGUF / llama.cpp is stateless; guard early and produce a clear artifact.
    if (::xllama::model_uses_llama_backend(model_name)) {
        log_output("[xllama] kv-bench: skipping (GGUF model is stateless via llama.cpp)\n");
        // Write a minimal .done so orchestrators do not hang.
        FILE* done =
            _wfopen(utf8_to_wstring(resolve_local_path("bench-kv-result.csv.done")).c_str(), L"w");
        if (done) {
            fputs("skipped-gguf\n", done);
            fclose(done);
        }
        return;
    }

    std::string err;
    ::xllama::SessionParams sp;
    sp.model_path = model_name;
    sp.n_ctx = 2048;
    sp.n_threads = n_threads;
    auto sess = ::xllama::Session::create(sp, &err);
    if (!sess) {
        log_output("[xllama] kv-bench: session create failed: " + err + "\n");
        return;
    }

    auto mkgp = [&](const std::string& p, bool reuse, bool reset) {
        ::xllama::GenerateParams gp;
        gp.prompt = p;
        gp.n_predict = 96;
        gp.reuse_kv = reuse;
        gp.reset_kv = reset;
        gp.stop_sequences.push_back("<|im_end|>");
        return gp;
    };

    // Turn 1: seed the persistent generator.
    auto r1 = sess->generate(mkgp(chatml(sys, u1), /*reuse=*/true, /*reset=*/true));
    // Turn 2 (KV reuse): append only the new turn's delta.
    std::string delta = (r1.ended_with_stop ? std::string("\n") : std::string("<|im_end|>\n")) +
                        "<|im_start|>user\n" + u2 + "<|im_end|>\n<|im_start|>assistant\n";
    auto r2 = sess->generate(mkgp(delta, /*reuse=*/true, /*reset=*/false));
    // Turn 2 (cold): full re-prefill of the whole 2-turn context — the pre-Stage-2
    // behaviour. Uses turn-1's actual output so the token count matches.
    std::string full2 = "<|im_start|>system\n" + sys + "<|im_end|>\n<|im_start|>user\n" + u1 +
                        "<|im_end|>\n<|im_start|>assistant\n" + r1.output_text +
                        "<|im_end|>\n<|im_start|>user\n" + u2 +
                        "<|im_end|>\n<|im_start|>assistant\n";
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
    std::string user_prompt = "Hello from Xbox Series S. Tell me about your architecture.";
    {
        std::string prompt_path = resolve_local_path("prompt.txt");
        FILE* pf = _wfopen(utf8_to_wstring(prompt_path).c_str(), L"r");
        if (pf) {
            char buf[8192] = {};
            size_t n = fread(buf, 1, sizeof(buf) - 1, pf);
            fclose(pf);
            if (n > 0) {
                user_prompt = buf;
                // Strip trailing whitespace/newlines
                while (!user_prompt.empty() &&
                       (user_prompt.back() == '\n' || user_prompt.back() == '\r' ||
                        user_prompt.back() == ' '))
                    user_prompt.pop_back();
            }
        }
    }
    // Wrap with ChatML template (required for SmolLM2-Instruct).
    std::string prompt = "<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n"
                         "<|im_start|>user\n" +
                         user_prompt +
                         "<|im_end|>\n"
                         "<|im_start|>assistant\n";

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

    // Read optional bench_threads.txt — written by bench-xbox-ort.sh per variant.
    // Used both to set params.n_threads (CSV tracking) and to label the host column.
    int bench_threads = 0;
    {
        std::string tpath = resolve_local_path("bench_threads.txt");
        FILE* tf = _wfopen(utf8_to_wstring(tpath).c_str(), L"r");
        if (tf) {
            char buf[16] = {};
            if (fread(buf, 1, sizeof(buf) - 1, tf) > 0)
                bench_threads = std::atoi(buf);
            fclose(tf);
        }
    }

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
                         bench_threads, host_buf2);
            return;
        }
    }

    log_output("[xllama] bench model: " + model_name + "\n");
    log_output("[xllama] bench prompt: " + prompt.substr(0, 80) + "...\n");

    InferenceParams params;
    params.model_path = model_name;
    params.prompt = prompt;
    params.n_predict = 512;
    params.n_threads = bench_threads; // 0 = auto; set by bench-xbox-ort.sh

    char host_buf[64];
    if (bench_threads > 0)
        snprintf(host_buf, sizeof(host_buf), "xbox-series-s-t%d", bench_threads);
    else
        snprintf(host_buf, sizeof(host_buf), "xbox-series-s");

    InferenceResult res = ::xllama::run_inference(params);
    xllama::write_bench_csv(params, res, host_buf);
#endif
}

} // namespace xllama::bridge
