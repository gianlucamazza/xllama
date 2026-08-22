// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include "inference-bridge.h"

#include "xllama/chat_prompt.h"
#include "xllama/device_train.h"
#include "xllama/diskbw.h"
#include "xllama/gpubw.h"
#include "xllama/gpugemv.h"
#include "xllama/inference.h"
#include "xllama/json_utils.h"
#include "xllama/membw.h"
#include "xllama/path_utils.h"
#include "xllama/personalize.h"
#include "xllama/platform.h"
#include "xllama/ramceil.h"
#include "xllama/session.h"
#include "xllama/training.h"
#include "xllama/utf8_utils.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <mutex>
#include <string>
#include <vector>

#ifdef XLLAMA_UWP
    // WS-F microphone probe (run_mic_probe). Universal contract, so no
    // SDKReference is needed — but per uwp-constraints.md §10b that says
    // nothing about activation, which is what the probe measures.
    //
    // unknwn.h first, and not by taste: WIN32_LEAN_AND_MEAN omits objbase.h,
    // which is what would otherwise define IUnknown, and winrt/base.h only
    // forward-declares it. The probe derives IMemoryBufferByteAccessXll from
    // IUnknown, so an incomplete declaration is a hard error here rather than
    // the usual static_assert. pch.h carries the same include for the same
    // reason; this TU does not include pch.h.
    #include <unknwn.h>

    #include <winrt/Windows.Foundation.Metadata.h>
    #include <winrt/Windows.Media.Audio.h>
    #include <winrt/Windows.Media.Capture.h>
    #include <winrt/Windows.Media.MediaProperties.h>
    #include <winrt/Windows.Media.Render.h>
    #include <winrt/Windows.Media.h>
#endif

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
                  const std::string& u2, int n_threads, int n_ctx, const char* host,
                  int run_index) {
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
        // run_index appended last (W1.1), same as the single-turn CSV: repeats are
        // recoverable rather than pre-averaged. 0 = single-run / legacy.
        fputs("model,prefill1_ms,n_p1,prefill2_reuse_ms,n_p2_reuse,prefill2_cold_ms,n_p2_cold,"
              "speedup,decode_tok_s,n_ctx,host,date,run_index\n",
              fp);
        double dtok = (r2.n_eval > 0 && r2.t_eval_ms > 0) ? r2.n_eval / (r2.t_eval_ms / 1000.0) : 0;
        fprintf(fp, "%s,%.1f,%d,%.1f,%d,%.1f,%d,%.2f,%.2f,%d,%s,%s,%d\n", model_name.c_str(),
                r1.t_p_eval_ms, r1.n_p_eval, r2.t_p_eval_ms, r2.n_p_eval, r2c.t_p_eval_ms,
                r2c.n_p_eval, speedup, dtok, sp.n_ctx, host ? host : "unknown", date_buf,
                run_index);
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

    // model.txt and prompt.txt are REQUIRED, and their absence aborts the run.
    //
    // Both used to fall back to a hardcoded default — smollm2-360m-cpu-int4 and a
    // 58-character prompt. That turns a lost upload into a silent measurement of
    // the wrong thing, and WDP POSTs are documented to fail silently in this
    // project (a POST without X-CSRF-Token returns success and writes nothing).
    // The result would be a real, plausible CSV row describing a run nobody asked
    // for, appended to the results file of the run that was asked for. Same defect
    // class as the invented quant label: a benchmark that guesses its own inputs
    // produces evidence indistinguishable from the genuine kind.
    //
    // read_local_file reads to EOF; the hand-rolled reader this replaced used a
    // fixed char buf[8192] and truncated silently at ~2k tokens — exactly the
    // range a prompt-length sweep needs.
    std::string user_prompt = read_local_file("prompt.txt");
    if (user_prompt.empty()) {
        log_output("[xllama] bench: prompt.txt missing or empty — refusing to bench a prompt "
                   "nobody asked for. Upload it and retry.\n");
        return;
    }
    log_output("[xllama] bench: prompt.txt " + std::to_string(user_prompt.size()) + " bytes\n");

    std::string model_name;
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
    if (model_name.empty()) {
        log_output("[xllama] bench: model.txt missing or empty — refusing to pick a model. "
                   "Upload it and retry.\n");
        return;
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
    // #172: llama.cpp physical prefill chunk (n_ubatch). 0 = llama default
    // (512). Labeled in the host column (-uN) because the CSV has no ubatch
    // column — a row that does not carry the variable under study is not
    // interpretable (the Phase 12 lesson, twice).
    const int bench_ubatch = read_local_int("bench_ubatch.txt", 0);
    // #171: q8_0 KV cache + flash attention A/B. Host-column tag -kvq8, same
    // rationale as -uN (the CSV schema carries no cache-type column).
    const int bench_kvq8 = read_local_int("bench_kvq8.txt", 0);
    // Phase 15 W2 (#210): draft-free prompt-lookup speculative decoding.
    // 0 = off (default); 1 = on. Host-column tag -plookup (CSV has no dedicated
    // column — same pattern as -kvq8 / -uN).
    const int bench_prompt_lookup = read_local_int("bench_prompt_lookup.txt", 0);
    // W1.1: which repetition this run is, written by the bench driver before each
    // iteration. Echoed into the CSV run_index column so the driver can append
    // every repeat and the summary generator can report a spread. 0 = single run.
    const int bench_run_index = read_local_int("bench_run_index.txt", 0);

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
                         bench_threads, bench_ctx, host_buf2, bench_run_index);
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
    params.n_threads = bench_threads;                // 0 = auto; set by bench-xbox-ort.sh
    params.n_ubatch = bench_ubatch;                  // #172: 0 = llama default (512)
    params.kv_q8 = bench_kvq8 != 0;                  // #171: q8_0 KV + flash attention
    params.prompt_lookup = bench_prompt_lookup != 0; // #210 W2
    params.stop_sequences = fmt.stop_sequences;      // clean stop for Gemma's <end_of_turn>
    params.run_index = bench_run_index;              // W1.1: echo into CSV (0 = single-run)

    char host_buf[80];
    int host_len = snprintf(host_buf, sizeof(host_buf), "xbox-series-s");
    if (bench_threads > 0)
        host_len +=
            snprintf(host_buf + host_len, sizeof(host_buf) - host_len, "-t%d", bench_threads);
    if (bench_ubatch > 0)
        host_len +=
            snprintf(host_buf + host_len, sizeof(host_buf) - host_len, "-u%d", bench_ubatch);
    if (bench_kvq8 != 0)
        host_len += snprintf(host_buf + host_len, sizeof(host_buf) - host_len, "-kvq8");
    if (bench_prompt_lookup != 0)
        snprintf(host_buf + host_len, sizeof(host_buf) - host_len, "-plookup");

    InferenceResult res = ::xllama::run_inference(params);
    if (params.prompt_lookup) {
        char spec_buf[192];
        snprintf(spec_buf, sizeof(spec_buf),
                 "[xllama] SPEC_STATS success=%d n_eval=%d t_eval_ms=%.1f "
                 "n_drafted=%d n_spec_accepted=%d\n",
                 res.success ? 1 : 0, res.n_eval, res.t_eval_ms, res.n_drafted,
                 res.n_spec_accepted);
        log_output(spec_buf);
    }
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
// run_diskbw (called from UWP diskbw.flag mode background thread)
// ---------------------------------------------------------------------------

void run_diskbw() {
#ifdef XLLAMA_UWP
    log_output("[xllama] diskbw: measuring sandboxed NVMe read bandwidth\n");
    // 4 GiB incompressible file in LocalState — above the RAM budget, so a
    // buffered pass cannot be served entirely from cache. Deleted afterwards.
    const std::string path = resolve_local_path("diskbw-test.bin");
    std::string err;
    if (!::xllama::ensure_diskbw_file(path, ::xllama::kDiskbwDefaultFileBytes, &err)) {
        log_output(("[xllama] diskbw FAIL: " + err + "\n").c_str());
        // A failed creation (disk full) can leave a multi-GiB partial file in
        // LocalState that nothing else would ever clean up.
        _wremove(utf8_to_wstring(path).c_str());
        return;
    }
    const ::xllama::DiskbwResult runs[] = {
        ::xllama::measure_diskbw(path, ::xllama::kDiskbwDefaultFileBytes,
                                 ::xllama::kDiskbwSeqBlockBytes, /*random=*/false, 1, 3, true),
        ::xllama::measure_diskbw(path, ::xllama::kDiskbwDefaultFileBytes,
                                 ::xllama::kDiskbwSeqBlockBytes, /*random=*/false, 4, 3, true),
        ::xllama::measure_diskbw(path, ::xllama::kDiskbwDefaultFileBytes,
                                 ::xllama::kDiskbwRndBlockBytes, /*random=*/true, 1, 3, true),
        ::xllama::measure_diskbw(path, ::xllama::kDiskbwDefaultFileBytes,
                                 ::xllama::kDiskbwRndBlockBytes, /*random=*/true, 4, 3, true),
    };

    const std::string csv = resolve_local_path("diskbw-result.csv");
    FILE* fp = _wfopen(utf8_to_wstring(csv).c_str(), L"w");
    if (fp) {
        fputs(::xllama::diskbw_csv_header(), fp);
        for (const auto& r : runs) {
            char lb[320];
            if (!r.error_msg.empty()) {
                snprintf(lb, sizeof(lb), "[xllama] diskbw FAIL: %s\n", r.error_msg.c_str());
                log_output(lb);
                continue;
            }
            snprintf(lb, sizeof(lb), "[xllama] diskbw: %s %dt unbuf=%d first=%.2f best=%.2f GB/s\n",
                     r.random ? "rnd" : "seq", r.threads, r.unbuffered ? 1 : 0, r.read_gbs_first,
                     r.read_gbs_best);
            log_output(lb);
            fputs(::xllama::format_diskbw_row(r, "xbox-series-s").c_str(), fp);
        }
        fclose(fp);
        FILE* done =
            _wfopen(utf8_to_wstring(resolve_local_path("diskbw-result.csv.done")).c_str(), L"w");
        if (done) {
            fputs("done\n", done);
            fclose(done);
        }
        log_output("[xllama] diskbw-result.csv written\n");
    } else {
        log_output("[xllama] diskbw: cannot open diskbw-result.csv\n");
    }
    _wremove(utf8_to_wstring(path).c_str());
#endif
}

// ---------------------------------------------------------------------------
// run_gpubw (called from UWP gpubw.flag mode background thread)
// ---------------------------------------------------------------------------

void run_gpubw() {
#ifdef XLLAMA_UWP
    log_output("[xllama] gpubw: measuring GPU STREAM bandwidth (own CS, no Agility)\n");
    // ~1 GiB default (issue #211). If alloc fails, measure_gpubw reports error.
    const ::xllama::GpubwResult r = ::xllama::measure_gpubw(::xllama::kGpubwDefaultBufferBytes, 3);

    char lb[320];
    snprintf(lb, sizeof(lb),
             "[xllama] gpubw: read=%.2f GB/s checksum_ok=%d d3d12_ran=%d kill_gate=%d "
             "buf=%zu MB err=%s\n",
             r.read_gbs, r.checksum_ok ? 1 : 0, r.d3d12_ran ? 1 : 0,
             ::xllama::gpubw_passes_kill_gate(r) ? 1 : 0, r.buffer_bytes / (1024 * 1024),
             r.error_msg.empty() ? "-" : r.error_msg.c_str());
    log_output(lb);

    const std::string csv = resolve_local_path("gpubw-result.csv");
    FILE* fp = _wfopen(utf8_to_wstring(csv).c_str(), L"w");
    if (fp) {
        fputs(::xllama::gpubw_csv_header(), fp);
        fputs(::xllama::format_gpubw_row(r, "xbox-series-s").c_str(), fp);
        fclose(fp);
        FILE* done =
            _wfopen(utf8_to_wstring(resolve_local_path("gpubw-result.csv.done")).c_str(), L"w");
        if (done) {
            fputs("done\n", done);
            fclose(done);
        }
        log_output("[xllama] gpubw-result.csv written\n");
    }
#endif
}

// ---------------------------------------------------------------------------
// run_gpugemv (called from UWP gpugemv.flag mode background thread)
// ---------------------------------------------------------------------------

void run_gpugemv() {
#ifdef XLLAMA_UWP
    log_output("[xllama] gpugemv: measuring Q4_K GEMV density (wave32 + naive A/B, no Agility)\n");

    const ::xllama::GpugemvKernel kernels[] = {::xllama::GpugemvKernel::Naive,
                                               ::xllama::GpugemvKernel::Wave32};
    ::xllama::GpugemvKernelSummary denser[1] = {};
    bool have_denser = false;

    const std::string csv = resolve_local_path("gpugemv-result.csv");
    FILE* fp = _wfopen(utf8_to_wstring(csv).c_str(), L"w");
    if (fp)
        fputs(::xllama::gpugemv_csv_header(), fp);

    for (auto kernel : kernels) {
        std::vector<::xllama::GpugemvResult> rows;
        ::xllama::measure_gpugemv_each(::xllama::kGpugemvDefaultN, ::xllama::kGpugemvDefaultK,
                                       /*recorded=*/3, kernel, &rows);
        bool g1_all3 = rows.size() >= 3;
        std::vector<double> gbs;
        if (!rows.empty()) {
            char cap[200];
            snprintf(
                cap, sizeof(cap),
                "[xllama] gpugemv: kernel=%s WaveOps=%d WaveLaneCountMin=%u WaveLaneCountMax=%u "
                "wave_ops=%d\n",
                ::xllama::gpugemv_kernel_name(kernel), rows[0].wave_ops_cap ? 1 : 0,
                rows[0].wave_lane_min, rows[0].wave_lane_max, rows[0].wave_ops ? 1 : 0);
            log_output(cap);
        }
        for (const auto& r : rows) {
            if (fp)
                fputs(::xllama::format_gpugemv_row(r, "xbox-series-s").c_str(), fp);
            g1_all3 =
                g1_all3 && r.run_index >= 1 && r.run_index <= 3 && ::xllama::gpugemv_passes_g1(r);
            if (r.run_index >= 1 && r.run_index <= 3)
                gbs.push_back(r.packed_gbs);
            char lb[400];
            snprintf(lb, sizeof(lb),
                     "[xllama] gpugemv: kernel=%s run=%d packed_gbs=%.2f packed_gbs_cpu=%.2f "
                     "gpu_timestamp=%d max_abs_err=%.6g checksum_ok=%d d3d12_ran=%d g1=%d g2=%d "
                     "err=%s\n",
                     ::xllama::gpugemv_kernel_name(r.kernel), r.run_index, r.packed_gbs,
                     r.packed_gbs_cpu, r.gpu_timestamp ? 1 : 0, static_cast<double>(r.max_abs_err),
                     r.checksum_ok ? 1 : 0, r.d3d12_ran ? 1 : 0,
                     ::xllama::gpugemv_passes_g1(r) ? 1 : 0, ::xllama::gpugemv_passes_g2(r) ? 1 : 0,
                     r.error_msg.empty() ? "-" : r.error_msg.c_str());
            log_output(lb);
        }
        if (rows.empty()) {
            char lb[200];
            snprintf(lb, sizeof(lb), "[xllama] gpugemv: kernel=%s produced no rows (PSO/setup)\n",
                     ::xllama::gpugemv_kernel_name(kernel));
            log_output(lb);
            g1_all3 = false;
        }
        double median = 0.0;
        if (!gbs.empty()) {
            std::sort(gbs.begin(), gbs.end());
            median = gbs[gbs.size() / 2];
        }
        const ::xllama::GpugemvLadder ladder = ::xllama::gpugemv_ladder(median, g1_all3);
        char lb[240];
        snprintf(lb, sizeof(lb), "[xllama] gpugemv: kernel=%s median=%.2f g1_all3=%d ladder=%s\n",
                 ::xllama::gpugemv_kernel_name(kernel), median, g1_all3 ? 1 : 0,
                 ::xllama::gpugemv_ladder_name(ladder));
        log_output(lb);
        if (kernel == ::xllama::GpugemvKernel::Wave32) {
            denser[0].kernel = kernel;
            denser[0].median_packed_gbs = median;
            denser[0].g1_all3 = g1_all3;
            denser[0].ladder = ladder;
            have_denser = true;
        }
    }

    const ::xllama::GpugemvLadder campaign = have_denser
                                                 ? ::xllama::gpugemv_campaign_verdict(denser, 1)
                                                 : ::xllama::GpugemvLadder::NotAVerdict;
    char clb[160];
    snprintf(clb, sizeof(clb), "[xllama] gpugemv: campaign_verdict=%s\n",
             ::xllama::gpugemv_ladder_name(campaign));
    log_output(clb);

    // Always write .done so bench-gpugemv.sh's 300 s wait cannot hang on partial PSO fail.
    if (fp) {
        fflush(fp);
        fclose(fp);
    }
    FILE* done =
        _wfopen(utf8_to_wstring(resolve_local_path("gpugemv-result.csv.done")).c_str(), L"w");
    if (done) {
        fputs("done\n", done);
        fclose(done);
    }
    log_output("[xllama] gpugemv-result.csv written\n");
#endif
}

// ---------------------------------------------------------------------------
// run_ramceil (called from UWP ramceil.flag mode background thread)
//
// Writes each row as it is produced and flushes: the whole point of the probe
// is to approach the point where the OS stops cooperating, so a buffered write
// would lose the last — and most informative — rows to a PLM kill.
// ---------------------------------------------------------------------------

void run_ramceil() {
#ifdef XLLAMA_UWP
    log_output("[xllama] ramceil: probing the committable heap ceiling\n");

    const std::string csv = resolve_local_path("ramceil-result.csv");
    FILE* fp = _wfopen(utf8_to_wstring(csv).c_str(), L"w");
    if (!fp) {
        log_output("[xllama] ramceil: cannot open ramceil-result.csv\n");
        return;
    }
    fputs(::xllama::ramceil_csv_header(), fp);
    fflush(fp);

    // 128 MB steps: fine enough to place the ceiling within a model quant's
    // margin, coarse enough that the probe stays short. The 8 GB limit is above
    // the console's 10 GB unified pool minus the OS reservation, so the stop
    // reason is the platform's answer, not ours. The 256 MB floor keeps a
    // margin for the OS rather than racing it to the kill.
    const ::xllama::RamCeilResult r = ::xllama::probe_ram_ceiling(
        /*step_mb=*/128, /*limit_mb=*/8192, /*floor_avail_mb=*/256,
        [fp](const ::xllama::RamCeilStep& s) {
            fputs(::xllama::format_ramceil_row(s, "xbox-series-s").c_str(), fp);
            fflush(fp);
        });

    char lb[256];
    snprintf(lb, sizeof(lb),
             "[xllama] ramceil: max committed %zu MB (start avail %zu MB, stop: %s)\n",
             r.max_committed_mb, r.avail_phys_start_mb, r.stop_reason.c_str());
    log_output(lb);

    fclose(fp);
    FILE* done =
        _wfopen(utf8_to_wstring(resolve_local_path("ramceil-result.csv.done")).c_str(), L"w");
    if (done) {
        fputs(r.stop_reason.c_str(), done);
        fputs("\n", done);
        fclose(done);
    }
    log_output("[xllama] ramceil-result.csv written\n");
#endif
}

// ---------------------------------------------------------------------------
// run_mic_probe (called from UWP mic.flag mode background thread)
//
// Phase 16 WS-F / card H16.6. The question is not "does the API exist" — the
// [mic] line in App.cpp already answers that, and per uwp-constraints.md §10b a
// present type says nothing about whether it can be activated. The question is
// whether an AppContainer app on GameOS can actually capture audio.
//
// The whole value of this probe is that it does NOT collapse to a boolean.
// The WinRT status enums already separate the cases that matter, and the JSON
// carries them by name rather than as a derived verdict:
//
//   AudioDeviceNodeCreationStatus::AccessDenied      -> the sandbox refuses.
//                                                       WS-F FAIL, and a
//                                                       permanent constraint.
//   AudioDeviceNodeCreationStatus::DeviceNotAvailable-> NO MIC IS PLUGGED IN.
//                                                       This is NOT a verdict
//                                                       on the sandbox; rerun
//                                                       with a headset before
//                                                       concluding anything.
//   Success but RMS ~ 0                              -> opened and silenced.
//   Success and RMS > 1e-3                           -> real capture.
//
// Everything is wrapped: this runs on HeadlessView's std::thread, where an
// escaping exception is a process kill with no log line and no .done marker —
// indistinguishable from a hang on the host side (uwp-constraints.md §10c).
// A probe that dies mute teaches the operator to read a timeout as a FAIL,
// which is the one reading this probe exists to prevent. So every exit path
// writes mic-result.json, including the ones that threw.
// ---------------------------------------------------------------------------

#ifdef XLLAMA_UWP
namespace {

// C++/WinRT does not project the byte-access interop, and AudioFrame samples
// are only reachable through it.
struct __declspec(uuid("5b0d3235-4dba-4d44-865e-8f1d0e4fd04d")) __declspec(novtable)
IMemoryBufferByteAccessXll : ::IUnknown {
    virtual HRESULT __stdcall GetBuffer(uint8_t** value, uint32_t* capacity) = 0;
};

const char* graph_status_name(winrt::Windows::Media::Audio::AudioGraphCreationStatus s) {
    using S = winrt::Windows::Media::Audio::AudioGraphCreationStatus;
    switch (s) {
    case S::Success:
        return "Success";
    case S::DeviceNotAvailable:
        return "DeviceNotAvailable";
    case S::FormatNotSupported:
        return "FormatNotSupported";
    case S::UnknownFailure:
        return "UnknownFailure";
    }
    return "Unrecognised";
}

const char* input_status_name(winrt::Windows::Media::Audio::AudioDeviceNodeCreationStatus s) {
    using S = winrt::Windows::Media::Audio::AudioDeviceNodeCreationStatus;
    switch (s) {
    case S::Success:
        return "Success";
    case S::DeviceNotAvailable:
        return "DeviceNotAvailable";
    case S::FormatNotSupported:
        return "FormatNotSupported";
    case S::UnknownFailure:
        return "UnknownFailure";
    case S::AccessDenied:
        return "AccessDenied";
    }
    return "Unrecognised";
}

// json_escape is in xllama::json_utils.h — included above.

} // namespace
#endif // XLLAMA_UWP

void run_mic_probe() {
#ifdef XLLAMA_UWP
    using namespace winrt::Windows::Media::Audio;
    using winrt::Windows::Foundation::Metadata::ApiInformation;
    using winrt::Windows::Media::AudioBufferAccessMode;
    using winrt::Windows::Media::Capture::MediaCategory;
    using winrt::Windows::Media::Render::AudioRenderCategory;

    log_output("[xllama] mic: probing AppContainer audio capture (WS-F / H16.6)\n");

    // Capture duration. Three seconds is the card's number; long enough that a
    // hum or a breath clears the silence floor, short enough to keep the whole
    // probe inside one app launch.
    const int capture_ms = 3000;

    int ag_present = 0, mc_present = 0;
    std::string graph_status = "not-attempted";
    std::string input_status = "not-attempted";
    std::string error;
    uint32_t sample_rate = 0, channels = 0;
    uint64_t samples = 0;
    double rms = -1.0, peak = -1.0;

    try {
        ag_present = ApiInformation::IsTypePresent(L"Windows.Media.Audio.AudioGraph") ? 1 : 0;
        mc_present = ApiInformation::IsTypePresent(L"Windows.Media.Capture.MediaCapture") ? 1 : 0;

        if (ag_present) {
            AudioGraphSettings settings(AudioRenderCategory::Speech);
            auto graph_result = AudioGraph::CreateAsync(settings).get();
            graph_status = graph_status_name(graph_result.Status());

            if (graph_result.Status() == AudioGraphCreationStatus::Success) {
                auto graph = graph_result.Graph();
                sample_rate = graph.EncodingProperties().SampleRate();
                channels = graph.EncodingProperties().ChannelCount();

                auto in_result = graph.CreateDeviceInputNodeAsync(MediaCategory::Speech).get();
                input_status = input_status_name(in_result.Status());

                if (in_result.Status() == AudioDeviceNodeCreationStatus::Success) {
                    auto out = graph.CreateFrameOutputNode();
                    in_result.DeviceInputNode().AddOutgoingConnection(out);

                    // QuantumStarted fires on the audio engine thread; the graph
                    // is stopped and the handler revoked before these are read.
                    std::mutex acc_mu;
                    double sum_sq = 0.0, pk = 0.0;
                    uint64_t n = 0;

                    auto token = graph.QuantumStarted(
                        [&](AudioGraph const&, winrt::Windows::Foundation::IInspectable const&) {
                            try {
                                auto frame = out.GetFrame();
                                auto buffer = frame.LockBuffer(AudioBufferAccessMode::Read);
                                auto ref = buffer.CreateReference();
                                uint8_t* data = nullptr;
                                uint32_t cap = 0;
                                if (FAILED(ref.as<IMemoryBufferByteAccessXll>()->GetBuffer(&data,
                                                                                           &cap)))
                                    return;
                                const float* f = reinterpret_cast<const float*>(data);
                                const size_t count = cap / sizeof(float);
                                double s2 = 0.0, p = 0.0;
                                for (size_t i = 0; i < count; ++i) {
                                    const double v = static_cast<double>(f[i]);
                                    s2 += v * v;
                                    const double a = v < 0 ? -v : v;
                                    if (a > p)
                                        p = a;
                                }
                                std::lock_guard<std::mutex> lk(acc_mu);
                                sum_sq += s2;
                                n += count;
                                if (p > pk)
                                    pk = p;
                            } catch (...) {
                                // A throw here would cross the audio thread and take
                                // the process; the probe would rather lose a quantum.
                            }
                        });

                    graph.Start();
                    ::Sleep(static_cast<DWORD>(capture_ms));
                    graph.Stop();
                    graph.QuantumStarted(token); // revoke before reading

                    std::lock_guard<std::mutex> lk(acc_mu);
                    samples = n;
                    peak = pk;
                    rms = n ? std::sqrt(sum_sq / static_cast<double>(n)) : 0.0;
                }
            }
        }
    } catch (winrt::hresult_error const& e) {
        char b[256];
        snprintf(b, sizeof(b), "hresult 0x%08X", static_cast<unsigned>(e.code().value));
        error = b;
        error += ": " + winrt::to_string(e.message());
    } catch (std::exception const& e) {
        error = e.what();
    } catch (...) {
        error = "unknown exception";
    }

    char json[1024];
    snprintf(json, sizeof(json),
             "{\n"
             "  \"audiograph_type_present\": %d,\n"
             "  \"mediacapture_type_present\": %d,\n"
             "  \"graph_status\": \"%s\",\n"
             "  \"input_node_status\": \"%s\",\n"
             "  \"sample_rate\": %u,\n"
             "  \"channels\": %u,\n"
             "  \"capture_ms\": %d,\n"
             "  \"samples\": %llu,\n"
             "  \"rms\": %.8f,\n"
             "  \"peak\": %.8f,\n"
             "  \"error\": \"%s\"\n"
             "}\n",
             ag_present, mc_present, xllama::json_escape(graph_status).c_str(),
             xllama::json_escape(input_status).c_str(), sample_rate, channels, capture_ms,
             static_cast<unsigned long long>(samples), rms, peak,
             xllama::json_escape(error).c_str());

    log_output(std::string("[mic] ") + graph_status + " / " + input_status +
               " rms=" + std::to_string(rms) + " samples=" + std::to_string(samples) +
               (error.empty() ? "" : (" error=" + error)) + "\n");

    const std::string path = resolve_local_path("mic-result.json");
    FILE* fp = _wfopen(utf8_to_wstring(path).c_str(), L"w");
    if (fp) {
        fputs(json, fp);
        fclose(fp);
    } else {
        log_output("[xllama] mic: cannot open mic-result.json\n");
    }

    // The .done marker is written even when the probe failed — the host waits
    // on it, and a missing marker must mean "the process died", not "the answer
    // was no". That distinction is the whole reason for the try/catch above.
    FILE* done = _wfopen(utf8_to_wstring(resolve_local_path("mic-result.json.done")).c_str(), L"w");
    if (done) {
        fputs(error.empty() ? "ok\n" : "error\n", done);
        fclose(done);
    }
    log_output("[xllama] mic-result.json written\n");
#endif
}

// ---------------------------------------------------------------------------
// run_train_job_localized — shared by headless train.flag and in-app personalize
// ---------------------------------------------------------------------------

xllama::TrainingResult run_train_job_localized(const xllama::TrainingJob& job_in,
                                               const xllama::DeviceTrainCallbacks& cb_in) {
    xllama::TrainingResult fail;
    fail.success = false;
#ifdef XLLAMA_UWP
    #ifdef XLLAMA_DEVICE_TRAIN
    xllama::TrainingJob job = job_in;
    // Job paths are LocalState-relative on console (the process cwd is the
    // read-only install dir). Absolute paths pass through untouched.
    auto localize = [](std::string& p) {
        if (!p.empty() && p.find(':') == std::string::npos && p[0] != '\\' && p[0] != '/')
            p = resolve_local_path(p);
    };
    localize(job.base_model);
    localize(job.dataset_path);
    localize(job.out_dir);

    log_output(("[xllama] train: " + xllama::format_training_job_summary(job) + "\n").c_str());

    // Compose callbacks: always log + write training/progress.json so the UI
    // and GET /v1/training/status can poll without WinRT.
    xllama::DeviceTrainCallbacks cb = cb_in;
    auto prev_status = cb.on_status;
    auto prev_progress = cb.on_progress;
    cb.on_status = [prev_status](const std::string& line) {
        log_output(("[xllama] train: " + line + "\n").c_str());
        if (prev_status)
            prev_status(line);
    };
    cb.on_progress = [prev_progress](const xllama::DeviceTrainProgress& p) {
        const char* stage = xllama::training_stage_name(p.stage);
        char lb[160];
        snprintf(lb, sizeof(lb), "[xllama] train: epoch %d/%d batch %lld/%lld loss=%.4f\n", p.epoch,
                 p.epochs, static_cast<long long>(p.ibatch), static_cast<long long>(p.ibatch_max),
                 p.loss);
        log_output(lb);
        const std::string json = xllama::format_train_progress_json(
            stage ? stage : "train", p.epoch, p.epochs, p.ibatch, p.ibatch_max, p.loss);
        const std::string prog_path = resolve_local_path("training/progress.json");
        FILE* fp = _wfopen(utf8_to_wstring(prog_path).c_str(), L"w");
        if (fp) {
            fputs(json.c_str(), fp);
            fclose(fp);
        }
        if (prev_progress)
            prev_progress(p);
    };

    return xllama::run_device_train_job(job, cb);
    #else
    (void)job_in;
    (void)cb_in;
    fail.error_msg = "built without XLLAMA_DEVICE_TRAIN";
    return fail;
    #endif
#else
    (void)job_in;
    (void)cb_in;
    fail.error_msg = "run_train_job_localized is UWP-only";
    return fail;
#endif
}

// run_train (called from UWP train.flag mode background thread)
// ---------------------------------------------------------------------------

void run_train() {
#ifdef XLLAMA_UWP
    bool ok = false;
    std::string err;
    #ifdef XLLAMA_DEVICE_TRAIN
    const std::string job_path = resolve_local_path("training/job.json");
    xllama::TrainingJob job;
    ok = xllama::load_training_job_file(job_path, job, &err);
    if (ok) {
        const xllama::TrainingResult r = run_train_job_localized(job);
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
