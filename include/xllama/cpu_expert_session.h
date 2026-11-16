#pragma once
#include "llama.h"
#include "cpu_pilot_policy.h"
#include "ua50/owned_expert_load.h"
#include "windows_held_reader.h"
#include <atomic>
#include <functional>
#include <memory>
#include <vector>
#include <string>
#include <stdexcept>
namespace xllama {
// INTERNAL source candidate only. No UI/LAN/C ABI setting or automatic provider activation.
struct CpuExpertEffective {uint32_t context,threads,threads_batch;size_t slots,ceiling,owned,arena,metadata;};
struct CpuExpertOptIn {
 enum class PlanProfile { Smoke64, FirstAnswer512, FinishedAnswer512, GenerationBounded };
 uint32_t generation_context=512; // only new bounded generation profile; old defaults unchanged
 PlanProfile plan_profile=PlanProfile::Smoke64;
 uint32_t planning_context() const {
  switch(plan_profile){case PlanProfile::Smoke64:return 64;case PlanProfile::FirstAnswer512:case PlanProfile::FinishedAnswer512:return 512;case PlanProfile::GenerationBounded:if(generation_context==64||generation_context==128||generation_context==256||generation_context==512)return generation_context;break;}
  throw std::invalid_argument("unsupported CPU planning profile");
 }
 bool enabled=false;
 std::shared_ptr<ua50::ExpertGpu> hybrid_gpu; // explicit synchronous MMID bridge; CPU graph remains CPU
 size_t numerical_proof_owned_bytes=0; // separately admitted, never borrowed from collector
 llama_model_params model=llama_model_default_params();
 llama_context_params context=llama_context_default_params();
 size_t slots=2,cache_ceiling=16u*1024u*1024u,stage_ceiling=4u*1024u*1024u;
 std::function<ua50::uwp::WindowsInputCandidate()> admitted_source;
 // Mandatory app quota/context/compute admission, not guessed from file size.
 std::function<void(const ua50::ExpertLoadPlan&,const llama_context_params&)> admit_budget;
 // Optional for other internal clients; the explicit pilot supplies a mandatory real quota guard.
 std::function<void(const char*)> phase_check;
 std::function<void(uint32_t,uint64_t,uint64_t,uint64_t)> route_trace;
 // Only the named cache comparison installs this instrumentation.
 std::function<void(const char*,const ua50::Counters&)> cache_events;
 size_t instrumentation_owned_bytes=0;
 std::function<void(const CpuExpertEffective&)> effective_observer;
 std::function<void(uint32_t,const ua50::Tensor&,const char*)> hybrid_catalog;
};
class CpuExpertSession final:public std::enable_shared_from_this<CpuExpertSession> {
public:
 enum class State { Opening,Ready,Decoding,Invalid,Closing,Closed };
 static std::shared_ptr<CpuExpertSession> open_if_requested(const CpuExpertOptIn*);
 ~CpuExpertSession();
 void begin_fresh_turn(); // mandatory before each full plain prompt; no saved/raw-thinking KV reuse
 int decode(llama_batch,std::atomic<bool>* turn_abort=nullptr);
 void request_stop() noexcept;
 bool close() noexcept; // false only for an active decode/close: caller must retain owner
 std::vector<llama_token> tokenize(const std::string&) const;
 std::string piece(llama_token,bool special=false) const;
 llama_token sample(llama_sampler*);
 const llama_model* model_view() const; // template/sampler vocabulary only, never context
 uint32_t context_length() const;
 uint32_t batch_length() const;
 using Effective=CpuExpertEffective;
 Effective effective() const;
 void primitive_vector(const char* tensor_name,uint32_t expert,const unsigned char* q8,size_t bytes,float* cpu,float* gpu,std::shared_ptr<ua50::ExpertGpu> executor);
 void capture_logits(float* destination,size_t elements) const; // exact128000 online proof, owning thread only
 std::shared_ptr<ua50::ExpertGpu> hybrid_gpu()const{return gpu_;}
 void invalidate(const char*) noexcept;
 void log_drained_counters() const;
 State state() const noexcept{return state_.load(std::memory_order_acquire);}
 const char* first_failure_after_drain() const noexcept{return first_failure_;}
 CpuExpertSession(const CpuExpertSession&)=delete;
 CpuExpertSession& operator=(const CpuExpertSession&)=delete;
private:
 CpuExpertSession()=default;
 struct ModelDelete{void operator()(llama_model*p)const noexcept{llama_model_free(p);}};
 struct ContextDelete{void operator()(llama_context*p)const noexcept{llama_free(p);}};
 std::shared_ptr<ua50::ExpertGpu> gpu_;
 std::shared_ptr<ua50::uwp::WindowsInputCandidate> input_; // held beyond context/model/Reader
 std::unique_ptr<llama_model,ModelDelete> model_;
 std::unique_ptr<llama_context,ContextDelete> context_;
 std::atomic<State> state_{State::Opening};std::atomic<bool> stop_{false};
 std::atomic<bool>* turn_abort_=nullptr; // serialized decode scope; callback only while llama_decode runs
 std::function<void(const char*)> phase_check_;
 std::function<void(const char*,const ua50::Counters&)> cache_events_;
 void cache_event(const char*);
 size_t applied_cache_ceiling_=0;
 char first_failure_[256]{};
 static bool should_abort(void*) noexcept;
 void latch_failure(const char*) noexcept;
};
// Metadata/no-payload native plan; caller must hold the admitted FILE and exclude writers.
cpu_pilot::MemoryPlan plan_cpu_expert_memory(FILE*,const CpuExpertOptIn&,const ua50::ExpertLoadPlan&);
using HashReadObserver=std::function<void(size_t,size_t,int64_t)>;
// Mandatory full-held-file SHA callback implementation, Windows BCrypt system API.
// Expected bytes/SHA must come from trusted app catalogue, not a LAN supplied claim.
std::function<ua50::uwp::WindowsInputCandidate()> make_windows_expert_source(
 const std::wstring& trusted_local_path,const ua50::uwp::ExpectedFile& trusted_expected,
 std::shared_ptr<std::atomic<bool>> stop,ua50::uwp::Clock::time_point absolute_deadline,
 size_t stage_ceiling=4u*1024u*1024u,HashReadObserver observer={});
ua50::uwp::ValidateHeldFile make_held_sha256_validator(
 const ua50::uwp::ExpectedFile&,std::function<bool()> stop,
 ua50::uwp::Clock::time_point absolute_deadline,HashReadObserver observer={});
}
