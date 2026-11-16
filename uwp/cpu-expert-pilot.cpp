// Developer-only, explicit CPU pilot. No GUI/LAN setting or automatic selection.
#include "pch.h"
#include "inference-bridge.h"
#include "model-downloader.h"
#include "xllama/compute_admission.h"
#include "xllama/cpu_expert_session.h"
#include "xllama/cpu_pilot_binding.h"
#include "xllama/session_hub.h"
#include "xllama/platform.h"
#include <winrt/Windows.System.h>
#include <winrt/Windows.Storage.h>
#include <windows.h>
#include <chrono>
#include <optional>
#include <array>
#include <condition_variable>
#include <filesystem>
#include <mutex>
#include <thread>
#include <cstdio>
#include "xllama/first_answer_record.h"
#include "ua50/cache_trace.h"
#include "../src/platform/ua50/dev_run_io.h"
#include "dev-run-input.h"
#include "gpu-proof-input.h"
#include "gpu-identity-input.h"
#include "xllama/expert_gpu_d3d12.h"
#include "xllama/gpu_proof_controller.h"
#include "xllama/gpu_proof_build_pins.h"
#include "xllama/generation_runner.h"
#include "xllama/generation_profile.h"
#include "xllama/generation_trace.h"
#include "xllama/generation_result.h"
#include "xllama/generation_capabilities.h"
#include <winrt/Windows.ApplicationModel.h>

namespace xllama::bridge {
namespace {
using namespace cpu_pilot;
struct Pilot {
    uint64_t initial_limit=0,ticket_expiry=0;
    std::atomic<uint64_t> max_sampled_usage{0};
    std::shared_ptr<std::atomic<bool>> abort=std::make_shared<std::atomic<bool>>(false);
    std::mutex mutex;
    std::condition_variable wake;
    bool done=false;
    std::thread monitor;
    winrt::Windows::System::MemoryManager::AppMemoryUsageLimitChanging_revoker limit_changed;
    std::weak_ptr<ua50::uwp::WindowsInputControl> control;
    std::weak_ptr<gpu_proof::Controller> gpu_controller;
    std::weak_ptr<ua50::ExpertGpu> generation_gpu; // null for all old stages
    std::atomic<int64_t> gpu_progress_ns{0};
    bool gpu_aggregate_quota=false;std::atomic<uint64_t> gpu_quota_snapshots{0};
    FILE* held=nullptr; // borrowed only on the serialized owning hub thread
    std::wstring stop_path;
    ua50::uwp::Clock::time_point deadline;
    Quota snapshot(const char* phase) {
        using M=winrt::Windows::System::MemoryManager;
        const auto level=M::AppMemoryUsageLevel();
        Quota q{M::AppMemoryUsageLimit(),M::AppMemoryUsage(),
            level==winrt::Windows::System::AppMemoryUsageLevel::High || level==winrt::Windows::System::AppMemoryUsageLevel::OverLimit};
        auto previous=max_sampled_usage.load();
        while(previous<q.usage && !max_sampled_usage.compare_exchange_weak(previous,q.usage)){}
        // Bracket both reads: a changing quota is a refusal, not an invented stable sample.
        const auto final_limit=M::AppMemoryUsageLimit();
        FILETIME ft{};GetSystemTimeAsFileTime(&ft);ULARGE_INTEGER utc{};utc.LowPart=ft.dwLowDateTime;utc.HighPart=ft.dwHighDateTime;
        char line[384];std::snprintf(line,sizeof(line),"[cpu-pilot] quota phase=%s pid=%lu utc100ns=%llu limit=%llu usage=%llu level=%d pressure=%d\n",
            phase,GetCurrentProcessId(),static_cast<unsigned long long>(utc.QuadPart),static_cast<unsigned long long>(q.limit),static_cast<unsigned long long>(q.usage),static_cast<int>(level),q.pressure?1:0);
        if(gpu_aggregate_quota)gpu_quota_snapshots.fetch_add(1,std::memory_order_relaxed);else log_output(line);
        if(final_limit!=q.limit) {log_output("[cpu-pilot] quota changed during bracketed sample\n");throw std::runtime_error("quota changed during sample");}
        if(level!=winrt::Windows::System::AppMemoryUsageLevel::Low && level!=winrt::Windows::System::AppMemoryUsageLevel::Medium && !q.pressure)
            throw std::runtime_error("unknown MemoryManager level");
        return q;
    }
    void phase(const char* name,uint64_t pending=0) {
        if(abort->load() || ua50::uwp::Clock::now()>=deadline || (ticket_expiry&&dev_utc_ms()>=ticket_expiry))throw std::runtime_error("pilot stop/deadline");
        const auto progress=gpu_progress_ns.load();if(progress>0){const auto now=std::chrono::duration_cast<std::chrono::nanoseconds>(ua50::uwp::Clock::now().time_since_epoch()).count();if(now<progress||now-progress>180000000000ll)throw std::runtime_error("GPU proof watchdog/clock");}
        check(snapshot(name),initial_limit,pending);
    }
    void request_stop() noexcept {
        abort->store(true);
        std::shared_ptr<ua50::uwp::WindowsInputControl> c;
        {std::lock_guard<std::mutex> lock(mutex);c=control.lock();}
        if(c)c->request_stop(); // stop request is NOT proof of terminal I/O
        std::shared_ptr<gpu_proof::Controller> gpu;{std::lock_guard<std::mutex> lock(mutex);gpu=gpu_controller.lock();}if(gpu)gpu->request_stop();
        std::shared_ptr<ua50::ExpertGpu> gen;{std::lock_guard<std::mutex> lock(mutex);gen=generation_gpu.lock();}if(gen)gen->request_stop();
    }
    void start_monitor() {
        monitor=std::thread([this]{
            try {winrt::init_apartment();}catch(...) {request_stop();return;}
            std::unique_lock<std::mutex> lock(mutex);
            while(!done) {
                if(wake.wait_for(lock,std::chrono::milliseconds(250),[this]{return done;}))break;
                lock.unlock();
                try {
                    HANDLE stop_file=CreateFile2(stop_path.c_str(),FILE_READ_ATTRIBUTES,FILE_SHARE_READ,OPEN_EXISTING,nullptr);
                    if(stop_file!=INVALID_HANDLE_VALUE){CloseHandle(stop_file);throw std::runtime_error("pilot stop flag");}
                    if(GetLastError()!=ERROR_FILE_NOT_FOUND)throw std::runtime_error("pilot stop-path query failure");
                    phase("runtime");
                }catch(const std::exception& e) {if(!abort->load())log_output(std::string("[cpu-pilot] monitor stop: ")+e.what()+"\n");request_stop();}
                 catch(...) {if(!abort->load())log_output("[cpu-pilot] monitor stop: unknown telemetry failure\n");request_stop();}
                lock.lock();
            }
            lock.unlock();winrt::uninit_apartment();
        });
    }
    void finish_monitor() noexcept {
        {std::lock_guard<std::mutex> lock(mutex);done=true;}wake.notify_all();
        if(monitor.joinable())monitor.join();
    }
    ~Pilot(){finish_monitor();limit_changed.revoke();}
};
void require_matching_catalogue() {
    // An absent entry uses the built-in pin. An existing entry must agree exactly;
    // neither mutable overrides nor an unsigned manifest can replace this trust root.
    const auto entries=LoadModelManifest();
    const auto* entry=FindManifestEntry(entries,L"lfm25-8b-a1b-q4-k-m");
    if(!entry)return;
    if(entry->kind!=L"gguf" || entry->files.size()!=1 || entry->files[0].filename!=file ||
       entry->files[0].approx_bytes!=bytes || entry->files[0].sha256!=std::wstring(sha,sha+64) ||
       entry->gpu_layers!=0 || !entry->lora.empty())
        throw std::runtime_error("existing catalogue entry differs from built-in pilot pin");
}
} // namespace

// Shared admitted lifecycle; old smoke namespace and behavior remain unchanged.
static void run_cpu_expert_stage(bool answer_stage,bool finished=false,unsigned cache_profile=0,std::shared_ptr<DevRunInput> configured={}) {
    if(cache_profile>3 || (cache_profile && (!answer_stage||!finished)))throw std::invalid_argument("cache profile");
    auto compute=try_acquire_compute_job();
    if(!compute){log_output(configured?"[dev-run-v1] REFUSED: compute busy\n":"[cpu-pilot] NOT_ADMITTED compute busy\n");return;}
    auto& hub=session_hub();std::unique_lock<std::mutex> hub_lock(hub.mtx,std::try_to_lock);
    if(!hub_lock || hub.session || hub.failed_expert_load) {
        log_output(configured?"[dev-run-v1] REFUSED: existing hub owner\n":"[cpu-pilot] NOT_ADMITTED existing hub owner\n");return;
    }
    auto state=std::make_shared<Pilot>();
    bool success=false;
    auto record=answer_stage?std::make_shared<first_answer::Record>(configured?first_answer::Profile::Configured:cache_profile==1?first_answer::Profile::CacheA2:cache_profile==2?first_answer::Profile::CacheB512:finished?first_answer::Profile::Finished384:first_answer::Profile::First128):nullptr;
    if(configured)record->configure(configured->config.max_new,configured->config.prompt_cap);
    const std::wstring cache_prefix=cache_profile==1?L"ua50-cpu-lfm24-cache-a2-v1.":L"ua50-cpu-lfm24-cache-b512-v1.";
    std::shared_ptr<ua50::CacheTrace> trace;
    HANDLE trace_file=INVALID_HANDLE_VALUE;
    auto io=configured?std::make_shared<ua50::uwp::RunIo>():nullptr;
    std::shared_ptr<CpuExpertEffective> native_effective;
    uint64_t planned_stage=0,charged_instrumentation=0;
    if(io)io->set_phase(7);
    const std::string marker=configured?"[dev-run-v1]":cache_profile==1?"[cache-a2-v1]":cache_profile==2?"[cache-b512-v1]":finished?"[finished-answer-v2]":"[first-answer-v1]";
    const size_t max_new=record?record->max_new():2;
    // Batch1 evaluates prompt tokens and all sampled nonterminal tokens except the last.
    const uint64_t route_bound=ua50::mul(ua50::add(configured?configured->config.prompt_cap:128u,max_new)-1u,456u);
    HANDLE output=INVALID_HANDLE_VALUE;
    if(record)record->pid=GetCurrentProcessId();
    try {
        const std::wstring root=winrt::Windows::Storage::ApplicationData::Current().LocalFolder().Path().c_str();
        const auto base=std::filesystem::path(root);
        const auto path=base/L"models"/L"lfm25-8b-a1b-q4-k-m"/file;
        ULARGE_INTEGER free_caller{},total{},free_total{};
        if(!GetDiskFreeSpaceExW(base.c_str(),&free_caller,&total,&free_total))throw std::runtime_error("pilot LocalFolder disk space unknown");
        char disk[256];std::snprintf(disk,sizeof(disk),"[cpu-pilot] LocalFolder caller_free=%llu total=%llu total_free=%llu\n",static_cast<unsigned long long>(free_caller.QuadPart),static_cast<unsigned long long>(total.QuadPart),static_cast<unsigned long long>(free_total.QuadPart));log_output(disk);
        if(free_caller.QuadPart<64ull*1024*1024)throw std::runtime_error("pilot telemetry disk reserve refused");
        state->stop_path=configured?configured->path("stop").wstring():(base/(cache_profile?cache_prefix+L"stop":finished?L"ua50-cpu-lfm24-finished-answer-v2.stop":answer_stage?L"ua50-cpu-lfm24-first-answer-v1.stop":L"ua50-cpu-lfm24-pilot.stop")).wstring();
        state->deadline=ua50::uwp::Clock::now()+std::chrono::minutes(20);
        if(configured){state->ticket_expiry=configured->ticket.expires;const auto ticket_now=dev_utc_ms();dev_run::need(state->ticket_expiry>ticket_now,"ticket expired at admission");
         state->deadline=std::min(state->deadline,ua50::uwp::Clock::now()+std::chrono::milliseconds(state->ticket_expiry-ticket_now));}
        // Persistent O_EXCL equivalent: restarting the app/flag cannot silently retry.
        if(configured)configured->spend();else {
        HANDLE claim=CreateFile2((base/(cache_profile?cache_prefix+L"claim":finished?L"ua50-cpu-lfm24-finished-answer-v2.claim":answer_stage?L"ua50-cpu-lfm24-first-answer-v1.claim":L"ua50-cpu-lfm24-pilot.claim")).c_str(),GENERIC_WRITE,0,CREATE_NEW,nullptr);
        if(claim==INVALID_HANDLE_VALUE)throw std::runtime_error("pilot claim exists/unavailable");
        CloseHandle(claim);
        }
        if(answer_stage) {
            log_output((marker+" claimed new stage; old pilot claim untouched\n"));
            output=CreateFile2((configured?configured->path("output.json"):base/(cache_profile?cache_prefix+L"output.json":finished?L"ua50-cpu-lfm24-finished-answer-v2.output.json":L"ua50-cpu-lfm24-first-answer-v1.output.json")).c_str(),GENERIC_WRITE,FILE_SHARE_READ,CREATE_NEW,nullptr);
            if(output==INVALID_HANDLE_VALUE)throw std::runtime_error("first-answer output exists/unavailable");
            if(cache_profile){trace_file=CreateFile2((configured?configured->path("trace.bin"):base/(cache_prefix+L"trace.bin")).c_str(),GENERIC_WRITE,FILE_SHARE_READ,CREATE_NEW,nullptr);
             if(trace_file==INVALID_HANDLE_VALUE)throw std::runtime_error("cache trace exists/unavailable");}
        }
        require_matching_catalogue();
        auto q=state->snapshot("entry");state->initial_limit=q.limit;check(q,q.limit,probe_allowance);
        std::weak_ptr<Pilot> weak_state=state;
        state->limit_changed=winrt::Windows::System::MemoryManager::AppMemoryUsageLimitChanging(winrt::auto_revoke,
            [weak_state](const auto&,const winrt::Windows::System::AppMemoryUsageLimitChangingEventArgs& args) noexcept {
                if(auto s=weak_state.lock()) {
                    try {if(args.NewLimit()!=s->initial_limit){log_output("[cpu-pilot] quota limit-changing event -> stop\n");s->request_stop();}}
                    catch(...) {s->request_stop();}
                }
            });
        state->phase("after-quota-subscription",probe_allowance);
        if(cache_profile){
         if(configured)trace=std::make_shared<ua50::CacheTrace>(GetCurrentProcessId(),configured->config.slots,configured->config.cache_ceiling,configured->config.max_new,configured->config.prompt_cap,configured->config_sha,dev_run::engine_digest,configured->config.id,configured->ticket_sha);
         else trace=std::make_shared<ua50::CacheTrace>(cache_profile,GetCurrentProcessId());
         state->phase("trace-buffer-allocated",probe_allowance);}
        state->start_monitor();
        llama_backend_init();state->phase("backend-initialized",probe_allowance);
        auto opt=std::make_shared<CpuExpertOptIn>();opt->enabled=true;
        if(cache_profile){opt->slots=configured?configured->config.slots:cache_profile==1?2:512;opt->cache_ceiling=configured?configured->config.cache_ceiling:cache_profile==1?16777216ull:cache_b512_ceiling;
         opt->instrumentation_owned_bytes=trace->required_owned_bytes();
         if(configured){opt->instrumentation_owned_bytes=ua50::add(opt->instrumentation_owned_bytes,ua50::add(configured->owned_bytes(),ua50::add(io->owned_bytes(),sizeof(ua50::uwp::ObservedWindowsReader)+4096)));charged_instrumentation=opt->instrumentation_owned_bytes;
          native_effective=std::make_shared<CpuExpertEffective>();opt->effective_observer=[native_effective,configured](const CpuExpertEffective&e){if(e.context!=512||e.threads!=1||e.threads_batch!=1||e.slots!=configured->config.slots||e.ceiling!=configured->config.cache_ceiling||e.owned>e.ceiling)throw std::runtime_error("constructed native effective mismatch");*native_effective=e;};}
         opt->cache_events=[trace,io,record](const char* event,const ua50::Counters& counters){
          if(std::strcmp(event,"begin")==0){trace->begin(counters);if(io)io->begin_frame(record->samples.empty());}
          else if(std::strcmp(event,"end")==0){trace->end(counters);if(io)io->end_frame();}
          else if(std::strcmp(event,"terminal")==0){trace->terminal(counters);if(io)io->after_drain(counters.read_calls,counters.read_bytes);}
          else throw std::runtime_error("unknown cache event");};}
        opt->plan_profile=finished?CpuExpertOptIn::PlanProfile::FinishedAnswer512:answer_stage?CpuExpertOptIn::PlanProfile::FirstAnswer512:CpuExpertOptIn::PlanProfile::Smoke64;
        opt->context.n_ctx=answer_stage?512:64;opt->context.n_batch=1;opt->context.n_ubatch=1;opt->context.n_seq_max=1;
        opt->context.n_outputs_max=1;opt->context.n_outputs_max_per_seq=1;
        opt->context.n_threads=1;opt->context.n_threads_batch=1;
        opt->context.offload_kqv=false;opt->context.op_offload=false;opt->context.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;
        opt->context.type_k=GGML_TYPE_F16;opt->context.type_v=GGML_TYPE_F16;
        opt->model.n_gpu_layers=0;opt->model.main_gpu=-1;opt->model.split_mode=LLAMA_SPLIT_MODE_NONE;
        opt->model.load_mode=LLAMA_LOAD_MODE_NONE;opt->model.lazy_mode=LLAMA_LAZY_MODE_OFF;
        opt->model.use_extra_bufts=false;
        opt->model.progress_callback=[](float,void* user)->bool {return !static_cast<Pilot*>(user)->abort->load();};
        opt->model.progress_callback_user_data=state.get();
        ua50::uwp::ExpectedFile expected{bytes,sha};
        HashReadObserver sha_observer;if(io)sha_observer=[io](size_t want,size_t got,int64_t ns){io->sha(want,got,ns);};
        auto source=make_windows_expert_source(path.wstring(),expected,state->abort,state->deadline,opt->stage_ceiling,sha_observer);
        opt->admitted_source=[state,source,base,record,io]() {
            state->phase("before-held-SHA",probe_allowance);
            auto writer=ModelDownloader::WriterGate().try_acquire();
            if(!writer.owns_lock())throw std::runtime_error("model writer busy");
        for(const auto& folder:{base,base/L"models",base/L"models"/L"lfm25-8b-a1b-q4-k-m"}) {
            CREATEFILE2_EXTENDED_PARAMETERS open{};open.dwSize=sizeof(open);
            open.dwFileFlags=FILE_FLAG_BACKUP_SEMANTICS|FILE_FLAG_OPEN_REPARSE_POINT;
            HANDLE dir=CreateFile2(folder.c_str(),FILE_READ_ATTRIBUTES,FILE_SHARE_READ|FILE_SHARE_WRITE|FILE_SHARE_DELETE,OPEN_EXISTING,&open);
            if(dir==INVALID_HANDLE_VALUE)throw std::runtime_error("pilot model directory unavailable");
            FILE_ATTRIBUTE_TAG_INFO attrs{};BOOL ok=GetFileInformationByHandleEx(dir,FileAttributeTagInfo,&attrs,sizeof(attrs));CloseHandle(dir);
            if(!ok || !(attrs.FileAttributes&FILE_ATTRIBUTE_DIRECTORY) || (attrs.FileAttributes&FILE_ATTRIBUTE_REPARSE_POINT))
                throw std::runtime_error("pilot LocalFolder model directory missing/reparse");
        }

            if(record)record->pre_sha_begin_ns=record->tick();
            if(io)io->set_phase(0);
            auto c=bind_writer(std::move(writer),source);
            if(io){auto make=c.make_reader;c.make_reader=[make,io](size_t stage){
             auto r=std::make_shared<ua50::uwp::ObservedWindowsReader>(make(stage),io);io->observed=r;return r;};}
            if(record)record->pre_sha_end_ns=record->tick();
            {std::lock_guard<std::mutex> lock(state->mutex);state->control=c.control;}
            state->held=c.file;
            state->phase("held-SHA-admitted",probe_allowance);
            if(record)record->pre_sha=true;
            return c;
        };
        std::weak_ptr<CpuExpertOptIn> weak_opt=opt;
        opt->admit_budget=[state,weak_opt,trace,&planned_stage](const ua50::ExpertLoadPlan& input,const llama_context_params&) {
            auto o=weak_opt.lock();if(!o)throw std::runtime_error("pilot config expired");
            state->phase("before-native-sizing",probe_allowance);
            auto p=plan_cpu_expert_memory(state->held,*o,input);
            if(trace)trace->plan(input.cache_owned_bytes,input.stage_bytes);planned_stage=input.stage_bytes;
            char line[512];std::snprintf(line,sizeof(line),"[cpu-pilot] joint-plan resident=%llu kv_recurrent=%llu scheduler=%llu cpu_workspace=%llu output=%llu cache=%llu stage=%llu extra_reserve=%llu\n",
                static_cast<unsigned long long>(p.resident),static_cast<unsigned long long>(p.kv_recurrent),static_cast<unsigned long long>(p.compute),static_cast<unsigned long long>(p.cpu_workspace),static_cast<unsigned long long>(p.output),static_cast<unsigned long long>(p.cache),static_cast<unsigned long long>(p.stage),static_cast<unsigned long long>(reserve));
            log_output(line);state->phase("before-payload-allocation",p.total());
        };
        opt->phase_check=[state,io](const char* phase){if(io){if(std::strcmp(phase,"before-sample")==0)io->set_phase(4);else if(std::strcmp(phase,"before-metadata")==0)io->set_phase(1);else if(std::strcmp(phase,"before-model-load")==0)io->set_phase(2);}state->phase(phase);};
        opt->route_trace=[record,finished,route_bound,trace,count=size_t(0)](uint32_t role,uint64_t expert,uint64_t offset,uint64_t length) mutable {
            if(record){
                if(trace)trace->key(role,expert,length);
                if(++count>(finished?route_bound:256u*456u))throw std::runtime_error("first-answer route request bound exceeded");
                ++record->route_requests;record->route_logical_bytes=add(record->route_logical_bytes,length);return;
            }
            if(++count>4096)throw std::runtime_error("bounded pilot route trace exceeded");
            char line[256];std::snprintf(line,sizeof(line),"[cpu-pilot] route-request role=%u expert=%llu offset=%llu bytes=%llu\n",role,static_cast<unsigned long long>(expert),static_cast<unsigned long long>(offset),static_cast<unsigned long long>(length));log_output(line);
        };
        SessionParams sp;sp.model_path=id;sp.backend=Backend::LlamaCpp;sp.cpu_experts=opt;
        sp.n_ctx=answer_stage?512:64;sp.n_batch=1;sp.n_ubatch=1;sp.n_threads=1;
        if(record)record->opening_begin_ns=record->tick();
        std::string error;auto* session=hub.ensure_locked(id,sp,&error);
        if(record)record->opening_end_ns=record->tick();
        if(!session)throw std::runtime_error("pilot opening refused: "+error);
        GenerateParams gp;gp.prompt="Hi";gp.n_predict=2;gp.temperature=0;gp.abort_flag=state->abort.get();
        if(record) {
            const auto* model=static_cast<const llama_model*>(session->resident_llama_model());
            if(!model)throw std::runtime_error("first-answer native model unavailable");
            const auto* vocab=llama_model_get_vocab(model);
            record->bos=llama_vocab_bos(vocab);
            record->prompt=configured?first_answer::render_user(llama_model_chat_template(model,nullptr),llama_vocab_get_add_bos(vocab),record->bos,configured->config.prompt):first_answer::render(llama_model_chat_template(model,nullptr),llama_vocab_get_add_bos(vocab),record->bos);
            record->template_verified=true;
            gp.prompt=record->prompt;gp.n_predict=static_cast<int>(max_new);gp.expert_answer=record;
            if((cache_profile&&session->context_length()!=512) || session->context_length()>512 || session->count_tokens(gp.prompt)>(configured?configured->config.prompt_cap:128))throw std::runtime_error("first-answer context/prompt refused");
            log_output(marker+" generate begin CPU1 ctx512 maxnew"+std::to_string(max_new)+" template ea663864 tokenizer-owned BOS\n");
        } else if(session->context_length()>256 || session->count_tokens("Hi")>4)throw std::runtime_error("pilot resolved context/prompt refused");
        gp.stall_timeout_s=180;
        gp.on_status=[](const std::string& s){log_output("[cpu-pilot] status "+s+"\n");};
        auto result=session->generate(gp);
        if(!result.success)throw std::runtime_error("pilot generate: "+result.error_msg);
        if(!record && (result.n_eval!=2 || result.n_prompt<=0 || result.n_prompt>4))throw std::runtime_error("pilot ended before bounded two-sample/one-step contract");
        if(record && (!record->complete || result.n_prompt<=0 || result.n_prompt>static_cast<int>(configured?configured->config.prompt_cap:128) || result.n_eval>static_cast<int>(max_new)))throw std::runtime_error("first-answer native result mismatch");
        state->phase("after-prompt-step");
        // Same held FILE post-check, before owner/WriterGate release; no path reopen.
        if(io)io->set_phase(5);
        auto post=make_held_sha256_validator(expected,[state]{return state->abort->load();},state->deadline,sha_observer);
        if(record)record->post_sha_begin_ns=record->tick();
        (void)post(state->held,bytes);
        if(record)record->post_sha_end_ns=record->tick();
        state->phase("post-SHA");if(record)record->post_sha=true;success=true;
        log_output(record?(marker+" generation and postSHA complete; awaiting owner close/output commit\n"):"[cpu-pilot] prompt+step complete; not a context/quality/speed acceptance\n");
    }catch(const std::exception& e){if(record)record->fail(e.what());log_output(std::string("[cpu-pilot] STOP: ")+e.what()+"\n");state->request_stop();}
    catch(...){if(record)record->fail("unknown stage failure");log_output("[cpu-pilot] STOP unknown failure\n");state->request_stop();}
    // Monitor never outlives state. Session destruction drains actual native I/O
    // before file, writer permit and compute admission may be released.
    state->finish_monitor();
    // Joining seals the monitor's final outcome. Normal owner close requests Reader
    // cancellation itself, so consume the latched monitor abort BEFORE teardown.
    if(record) {
        const bool was_success=success;
        success=record->after_monitor_join(success,state->abort->load());
        if(was_success && !success)log_output(marker+" STOP final monitor outcome; partial diagnostics retained\n");
    }
    if(io){try{io->set_phase(6);}catch(const std::exception&e){success=false;if(record->error.empty())record->fail(e.what());}}
    hub.reset_locked();state->held=nullptr;state->limit_changed.revoke();
    if(record) {
        record->owner_closed=true;record->quota_limit=state->initial_limit;record->usage_max_sampled=state->max_sampled_usage.load();
        if(io){try{io->finish();if(success&&!io->healthy(bytes)){success=false;record->fail("IO coverage/clock/native error");}}catch(const std::exception&e){success=false;if(record->error.empty())record->fail(e.what());}}
        if(cache_profile) {
            try {
                if(!trace)throw std::runtime_error("trace allocation unavailable");
                std::vector<int32_t> ids;ids.reserve(record->samples.size());for(const auto& sample:record->samples)ids.push_back(sample.id);
                trace->seal(success,record->prompt_ids,ids);
                if(success && !trace->complete()){success=false;record->fail("trace is not complete replayable");}
                if(trace_file==INVALID_HANDLE_VALUE)throw std::runtime_error("trace output unavailable");
                DWORD written=0;
                if(!WriteFile(trace_file,trace->bytes(),static_cast<DWORD>(trace->file_size()),&written,nullptr) || written!=trace->file_size() || !FlushFileBuffers(trace_file))throw std::runtime_error("trace write/flush failed");
                if(!CloseHandle(trace_file)){trace_file=INVALID_HANDLE_VALUE;throw std::runtime_error("trace close failed");}trace_file=INVALID_HANDLE_VALUE;
                record->cache_ab_json=trace->json(true);record->trace_complete_committed=trace->complete();
                log_output(marker+(trace->complete()?" TRACE_COMMITTED complete replayable\n":" TRACE_COMMITTED partial not replayable\n"));
            }catch(const std::exception& e){success=false;
                if(record->error.empty())record->fail(e.what());else record->complete=false;
                record->cache_ab_json=trace?trace->json(false):ua50::CacheTrace::unavailable(cache_profile,configured?configured->config.slots:0,configured?configured->config.cache_ceiling:0);
                log_output(marker+" TRACE_FAILED: "+e.what()+"\n");}
            if(trace_file!=INVALID_HANDLE_VALUE){CloseHandle(trace_file);trace_file=INVALID_HANDLE_VALUE;}
        }

        try {
            if(configured){
             const auto quote=first_answer::quote;const auto e=native_effective?*native_effective:CpuExpertEffective{};
             std::string effective="null";
             if(native_effective&&e.context){
              if(e.context!=512||e.threads!=1||e.threads_batch!=1||e.slots!=configured->config.slots||e.ceiling!=configured->config.cache_ceiling||e.owned>e.ceiling)throw std::runtime_error("native effective mismatch");
              effective="{\"model_id\":\"lfm25-8b-a1b-q4-k-m\",\"model_sha256\":"+quote(sha)+",\"context_tokens\":"+std::to_string(e.context)+",\"prompt_tokens\":"+std::to_string(record->prompt_ids.size())+",\"max_new_tokens\":"+std::to_string(record->max_new())+",\"threads\":"+std::to_string(e.threads)+",\"threads_batch\":"+std::to_string(e.threads_batch)+",\"backend_id\":\"cpu_expert_streaming\",\"device_mode\":\"cpu\",\"main_gpu\":-1,\"gpu_layers\":0,\"offload_kqv\":false,\"op_offload\":false,\"expert_placement\":\"cpu_streamed\",\"compute_mode\":\"nonfused\",\"sampler_kind\":\"greedy\",\"cache_policy\":\"global_lru\",\"cache_slots\":"+std::to_string(e.slots)+",\"cache_owned_ceiling_bytes\":"+std::to_string(e.ceiling)+",\"cache_required_owned_bytes\":"+std::to_string(e.owned)+",\"cache_arena_bytes\":"+std::to_string(e.arena)+",\"cache_metadata_bytes\":"+std::to_string(e.metadata)+",\"stage_required_bytes\":"+std::to_string(planned_stage)+",\"instrumentation_owned_bytes\":"+std::to_string(charged_instrumentation)+"}";
             }
             const auto package=winrt::to_string(winrt::Windows::ApplicationModel::Package::Current().Id().FullName());
             record->runner_complete=success&&io->healthy(bytes)&&effective!="null";
             if(success&&!record->runner_complete){success=false;record->fail("runner effective/IO completion unavailable");}
             record->runner_json="{\"run_id\":"+quote(configured->config.id)+",\"selection_id\":"+quote(configured->ticket.selection)+",\"config_sha256\":"+quote(configured->config_sha)+",\"ticket_sha256\":"+quote(configured->ticket_sha)+",\"engine_contract_sha256\":"+quote(dev_run::engine_digest)+",\"native_package_full_name\":"+quote(package)+",\"requested\":"+configured->config.raw+",\"effective\":"+effective+",\"storage\":"+io->json()+"}";
            }
            if(output==INVALID_HANDLE_VALUE)throw std::runtime_error("private output was not created");
            first_answer::commit_record(*record,[&](const std::string& json){DWORD written=0;
            if(!WriteFile(output,json.data(),static_cast<DWORD>(json.size()),&written,nullptr) || written!=json.size() || !FlushFileBuffers(output))throw std::runtime_error("private output write/flush failure");
            if(!CloseHandle(output)){output=INVALID_HANDLE_VALUE;throw std::runtime_error("private output close failure");}output=INVALID_HANDLE_VALUE;
            });
            log_output(success?(marker+" OUTPUT_COMMITTED normal owner/file/writer closed\n"):(marker+" OUTPUT_COMMITTED partial/refused/error owner/file/writer closed\n"));
        }catch(const std::exception& e){success=false;log_output(std::string((marker+" OUTPUT_FAILED: "))+e.what()+"\n");}
        if(output!=INVALID_HANDLE_VALUE)CloseHandle(output);
        log_output(success?(marker+" terminal normal\n"):(marker+" terminal stopped; no retry\n"));
    }
    log_output(success?"[cpu-pilot] terminal normal owner/file/writer close\n":"[cpu-pilot] terminal stopped; no automatic retry\n");
}
void run_cpu_expert_pilot(){run_cpu_expert_stage(false);}
void run_cpu_expert_first_answer(){run_cpu_expert_stage(true);}
void run_cpu_expert_finished_answer(){run_cpu_expert_stage(true,true);}
void run_cpu_expert_configured(){
 try{Pilot input_probe;input_probe.deadline=ua50::uwp::Clock::now()+std::chrono::minutes(20);
  const auto quota=input_probe.snapshot("before-config-parse");check(quota,quota.limit,probe_allowance);
  const std::filesystem::path root(winrt::Windows::Storage::ApplicationData::Current().LocalFolder().Path().c_str());
  auto input=std::make_shared<DevRunInput>(root);run_cpu_expert_stage(true,true,3,std::move(input));
 }catch(const std::exception&e){log_output(std::string("[dev-run-v1] REFUSED: ")+e.what()+"\n");}
 catch(...){log_output("[dev-run-v1] REFUSED: unknown input failure\n");}
}
void run_cpu_expert_cache_a2(){run_cpu_expert_stage(true,true,1);}
void run_cpu_expert_cache_b512(){run_cpu_expert_stage(true,true,2);}
#include "gpu-proof-pilot.inc"
#include "generation-runner.inc"
#include "gpu-identity-pilot.inc"
} // namespace xllama::bridge
