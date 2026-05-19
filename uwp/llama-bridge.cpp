#include "llama-bridge.h"

#include "llama.h"

#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>
#include <thread>
#include <vector>

#ifdef XLLAMA_UWP
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
// PSAPI_VERSION=2 routes GetProcessMemoryInfo → K32GetProcessMemoryInfo
// (kernel32), which is in WINAPI_PARTITION_APP. No psapi.lib needed.
#define PSAPI_VERSION 2
#include <psapi.h>
#include <winrt/Windows.Storage.h>
// mmap_replacement: see llama-mmap-uwp.cpp (Stage 1E will wire in mmap).
#endif

namespace xllama::bridge {

// ---------------------------------------------------------------------------
// Platform helpers
// ---------------------------------------------------------------------------

static int detect_threads() {
    int n = (int)std::thread::hardware_concurrency();
    return n > 0 ? n : 4;
}

static void log_output(const char* msg) {
#ifdef XLLAMA_UWP
    OutputDebugStringA(msg);
#else
    std::fputs(msg, stderr);
#endif
}

#ifdef XLLAMA_UWP
// Returns the absolute path to LocalFolder\models\<filename>.
static std::string resolve_model_path(const std::string& filename) {
    using namespace winrt::Windows::Storage;
    auto folder = ApplicationData::Current().LocalFolder();
    std::wstring wpath(folder.Path().c_str());
    wpath += L"\\models\\";
    // Convert filename to wide
    int sz = MultiByteToWideChar(CP_UTF8, 0, filename.c_str(), -1, nullptr, 0);
    std::wstring wfilename(sz, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, filename.c_str(), -1, wfilename.data(), sz);
    wpath += wfilename;
    // Convert back to UTF-8 for llama.cpp (which uses ggml_fopen -> _wfopen internally)
    int nsz = WideCharToMultiByte(CP_UTF8, 0, wpath.c_str(), -1, nullptr, 0, nullptr, nullptr);
    std::string result(nsz, '\0');
    WideCharToMultiByte(CP_UTF8, 0, wpath.c_str(), -1, result.data(), nsz, nullptr, nullptr);
    if (!result.empty() && result.back() == '\0') result.pop_back();
    return result;
}

// Returns the absolute path to LocalFolder\<filename>.
static std::string resolve_local_path(const std::string& filename) {
    using namespace winrt::Windows::Storage;
    auto folder = ApplicationData::Current().LocalFolder();
    std::wstring wpath(folder.Path().c_str());
    wpath += L"\\";
    int sz = MultiByteToWideChar(CP_UTF8, 0, filename.c_str(), -1, nullptr, 0);
    std::wstring wfn(sz, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, filename.c_str(), -1, wfn.data(), sz);
    wpath += wfn;
    int nsz = WideCharToMultiByte(CP_UTF8, 0, wpath.c_str(), -1, nullptr, 0, nullptr, nullptr);
    std::string result(nsz, '\0');
    WideCharToMultiByte(CP_UTF8, 0, wpath.c_str(), -1, result.data(), nsz, nullptr, nullptr);
    if (!result.empty() && result.back() == '\0') result.pop_back();
    return result;
}

static size_t peak_working_set_mb() {
    PROCESS_MEMORY_COUNTERS pmc{};
    if (K32GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
        return pmc.PeakWorkingSetSize / (1024 * 1024);
    return 0;
}
#else
static size_t peak_working_set_mb() { return 0; }
#endif

// ---------------------------------------------------------------------------
// run_inference
// ---------------------------------------------------------------------------

InferenceResult run_inference(const InferenceParams& params) {
    InferenceResult res;

#ifdef XLLAMA_UWP
    // Resolve model path relative to LocalFolder/models/
    std::string abs_model_path = resolve_model_path(params.model_path);
#else
    const std::string& abs_model_path = params.model_path;
#endif

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0; // CPU only in Phase 1

#ifdef XLLAMA_UWP
    mparams.use_mmap = false; // POSIX mmap unavailable in UWP; Stage 1E will add CreateFileMappingFromApp
#endif

    llama_model* model = llama_model_load_from_file(abs_model_path.c_str(), mparams);
    if (!model) {
        res.error_msg = "failed to load model: " + abs_model_path;
        log_output(("[xllama] " + res.error_msg + "\n").c_str());
        return res;
    }
    log_output("[xllama] model loaded\n");

    const int n_threads = params.n_threads > 0 ? params.n_threads : detect_threads();

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx     = (uint32_t)params.n_ctx;
    cparams.n_threads = n_threads;

    llama_context* ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        res.error_msg = "failed to create context";
        log_output("[xllama] failed to create context\n");
        llama_model_free(model);
        return res;
    }

    const llama_vocab* vocab = llama_model_get_vocab(model);

    // Tokenize prompt
    std::vector<llama_token> tokens(params.prompt.size() + 16);
    int n_tokens = llama_tokenize(
        vocab,
        params.prompt.c_str(), (int32_t)params.prompt.size(),
        tokens.data(), (int32_t)tokens.size(),
        /*add_special=*/true, /*parse_special=*/false);

    if (n_tokens < 0) {
        res.error_msg = "tokenization failed";
        log_output("[xllama] tokenization failed\n");
        llama_free(ctx);
        llama_model_free(model);
        return res;
    }
    tokens.resize((size_t)n_tokens);

    // Decode prompt + generate
    llama_batch batch = llama_batch_get_one(tokens.data(), n_tokens);
    if (llama_decode(ctx, batch) != 0) {
        res.error_msg = "prompt decode failed";
        log_output("[xllama] prompt decode failed\n");
        llama_free(ctx);
        llama_model_free(model);
        return res;
    }

    const llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    llama_sampler* sampler = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(params.temperature));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    // Generation loop
#ifdef XLLAMA_UWP
    // On UWP, write tokens to LocalFolder/output.txt
    std::string out_path = resolve_local_path("output.txt");
    FILE* out_fp = _wfopen(std::wstring(out_path.begin(), out_path.end()).c_str(), L"w");
#endif

    int n_generated = 0;
    while (n_generated < params.n_predict) {
        llama_token token = llama_sampler_sample(sampler, ctx, -1);
        if (llama_vocab_is_eog(vocab, token)) break;

        char buf[256] = {};
        int len = llama_token_to_piece(vocab, token, buf, sizeof(buf) - 1, 0, false);
        if (len > 0) {
            buf[len] = '\0';
            res.output_text += buf;
#ifdef XLLAMA_UWP
            if (out_fp) { fputs(buf, out_fp); fflush(out_fp); }
            OutputDebugStringA(buf);
#else
            std::fputs(buf, stdout);
            std::fflush(stdout);
#endif
        }

        llama_batch next = llama_batch_get_one(&token, 1);
        if (llama_decode(ctx, next) != 0) break;
        ++n_generated;
    }

#ifdef XLLAMA_UWP
    if (out_fp) fclose(out_fp);
#else
    std::fputc('\n', stdout);
#endif

    // Collect perf metrics via llama.cpp API
    llama_perf_context_data perf = llama_perf_context(ctx);
    res.t_load_ms   = perf.t_load_ms;
    res.t_p_eval_ms = perf.t_p_eval_ms;
    res.t_eval_ms   = perf.t_eval_ms;
    res.n_p_eval    = perf.n_p_eval;
    res.n_eval      = n_generated;
    res.peak_ws_mb  = peak_working_set_mb();
    res.success     = true;

    char log_buf[256];
    snprintf(log_buf, sizeof(log_buf),
             "[xllama] done: load=%.0fms prompt=%.1f tok/s decode=%.1f tok/s peak=%zuMB\n",
             res.t_load_ms,
             res.n_p_eval > 0 && res.t_p_eval_ms > 0
                 ? (double)res.n_p_eval / (res.t_p_eval_ms / 1000.0) : 0.0,
             res.n_eval > 0 && res.t_eval_ms > 0
                 ? (double)res.n_eval / (res.t_eval_ms / 1000.0) : 0.0,
             res.peak_ws_mb);
    log_output(log_buf);

    llama_sampler_free(sampler);
    llama_free(ctx);
    llama_model_free(model);
    return res;
}

// ---------------------------------------------------------------------------
// Write bench CSV row to LocalFolder/bench-result.csv
// (called from main_loop after a successful run)
// ---------------------------------------------------------------------------

#ifdef XLLAMA_UWP
static void write_bench_csv(const InferenceParams& params, const InferenceResult& res) {
    if (!res.success) return;

    std::string csv_path = resolve_local_path("bench-result.csv");
    FILE* fp = _wfopen(std::wstring(csv_path.begin(), csv_path.end()).c_str(), L"w");
    if (!fp) return;

    const char* header = "model,quant,backend,n_ctx,n_threads,"
                         "prompt_tok_s,decode_tok_s,peak_ws_mb,load_ms,host,date\n";
    fputs(header, fp);

    double prompt_tok_s = (res.n_p_eval > 0 && res.t_p_eval_ms > 0)
        ? (double)res.n_p_eval / (res.t_p_eval_ms / 1000.0) : 0.0;
    double decode_tok_s = (res.n_eval > 0 && res.t_eval_ms > 0)
        ? (double)res.n_eval / (res.t_eval_ms / 1000.0) : 0.0;

    // ISO-8601 date
    time_t now = time(nullptr);
    char date_buf[32];
    struct tm* tm_utc = gmtime(&now);
    strftime(date_buf, sizeof(date_buf), "%Y-%m-%dT%H:%M:%SZ", tm_utc);

    // model name without extension
    std::string model_name = params.model_path;
    auto dot = model_name.rfind('.');
    if (dot != std::string::npos) model_name = model_name.substr(0, dot);

    fprintf(fp, "%s,Q4_K_M,cpu,%d,%d,%.2f,%.2f,%zu,%.0f,xbox-series-s,%s\n",
            model_name.c_str(), params.n_ctx,
            params.n_threads > 0 ? params.n_threads : detect_threads(),
            prompt_tok_s, decode_tok_s,
            res.peak_ws_mb, res.t_load_ms, date_buf);
    fclose(fp);

    // Write marker file so bench-xbox.sh knows the run is complete
    std::string done_path = resolve_local_path("bench-result.csv.done");
    FILE* done = _wfopen(std::wstring(done_path.begin(), done_path.end()).c_str(), L"w");
    if (done) { fputs("done\n", done); fclose(done); }

    log_output("[xllama] bench-result.csv written\n");
}
#endif

// ---------------------------------------------------------------------------
// main_loop (called from App::Run background thread)
// ---------------------------------------------------------------------------

void main_loop() {
#ifdef XLLAMA_UWP
    using namespace winrt::Windows::Storage;

    // Read prompt from LocalFolder/prompt.txt, fallback to default.
    std::string prompt = "Hello from Xbox Series S. Tell me about your architecture.";
    {
        std::string prompt_path = resolve_local_path("prompt.txt");
        FILE* pf = _wfopen(std::wstring(prompt_path.begin(), prompt_path.end()).c_str(), L"r");
        if (pf) {
            char buf[8192] = {};
            size_t n = fread(buf, 1, sizeof(buf) - 1, pf);
            fclose(pf);
            if (n > 0) prompt = buf;
        }
    }

    // Read model filename from LocalFolder/model.txt, fallback to default.
    std::string model_filename = "qwen3-1.7b-Q4_K_M.gguf";
    {
        std::string model_cfg = resolve_local_path("model.txt");
        FILE* mf = _wfopen(std::wstring(model_cfg.begin(), model_cfg.end()).c_str(), L"r");
        if (mf) {
            char buf[512] = {};
            size_t n = fread(buf, 1, sizeof(buf) - 1, mf);
            fclose(mf);
            if (n > 0) {
                model_filename = buf;
                // trim trailing whitespace
                while (!model_filename.empty() &&
                       (model_filename.back() == '\n' || model_filename.back() == '\r' ||
                        model_filename.back() == ' '))
                    model_filename.pop_back();
            }
        }
    }

    log_output(("[xllama] model: " + model_filename + "\n").c_str());
    log_output(("[xllama] prompt: " + prompt.substr(0, 80) + "...\n").c_str());

    InferenceParams params;
    params.model_path = model_filename;
    params.prompt     = prompt;
    params.n_predict  = 128;

    InferenceResult res = run_inference(params);
    write_bench_csv(params, res);
#endif
    // Linux: no-op (use xllama-cli directly)
}

} // namespace xllama::bridge
