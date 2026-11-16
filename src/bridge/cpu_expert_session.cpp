#include "xllama/cpu_expert_session.h"
#include "llama-context.h"
#include "llama-model.h"
#include "ggml-cpu.h"
#include "llama-ext.h"
#include "llama-memory.h"
#include "xllama/platform.h"
#include <stdexcept>
#include <cstdio>
#include <climits>
#include <cstring>
namespace xllama {
void CpuExpertSession::latch_failure(const char*s) noexcept {
 if(!first_failure_[0])std::snprintf(first_failure_,sizeof(first_failure_),"%s",s?s:"expert failure");
}
bool CpuExpertSession::should_abort(void*p) noexcept{auto*o=static_cast<CpuExpertSession*>(p);return o->stop_.load(std::memory_order_acquire)||(o->turn_abort_&&o->turn_abort_->load());}
std::shared_ptr<CpuExpertSession> CpuExpertSession::open_if_requested(const CpuExpertOptIn*opt){
 if(!opt || !opt->enabled)return {}; // ordinary caller factory path remains untouched
 if(!opt->admitted_source || !opt->admit_budget)throw std::invalid_argument("explicit source and app budget admission required");
 if(!opt->context.n_ctx || opt->context.offload_kqv || opt->context.op_offload || opt->context.flash_attn_type!=LLAMA_FLASH_ATTN_TYPE_DISABLED || opt->context.n_threads<=0 || opt->context.n_threads_batch<=0)throw std::invalid_argument("explicit CPU/context policy required");
 auto out=std::shared_ptr<CpuExpertSession>(new CpuExpertSession());
 try{
  out->phase_check_=opt->phase_check;out->cache_events_=opt->cache_events;
  if(out->phase_check_)out->phase_check_("before-source");
  auto candidate=opt->admitted_source();
  if(!candidate.file || !candidate.file_anchor || !candidate.admission_anchor || !candidate.validate_binding || !candidate.control || !candidate.make_reader || !candidate.admission.bytes || candidate.admission.sha256.size()!=64 || candidate.admission.identity.empty())throw std::invalid_argument("incomplete Windows admission");
  out->input_=std::make_shared<ua50::uwp::WindowsInputCandidate>(std::move(candidate));
  auto input=out->input_;input->validate_binding(input->file);
  if(out->phase_check_)out->phase_check_("before-metadata");
  auto plan=ua50::inspect_expert_input(input->file,opt->slots);
  if(!plan.full_payload || plan.available_file_bytes!=input->admission.bytes || plan.cache_owned_bytes>opt->cache_ceiling || plan.stage_bytes>opt->stage_ceiling)throw std::invalid_argument("expert full payload/component budget refused");
  out->applied_cache_ceiling_=opt->cache_ceiling;
  const bool generation=opt->plan_profile==CpuExpertOptIn::PlanProfile::GenerationBounded;
  const bool envelope=generation?(opt->slots>=2&&opt->slots<=512&&opt->context.n_ctx==opt->planning_context()&&opt->numerical_proof_owned_bytes==0):(opt->slots==2&&opt->context.n_ctx==512&&opt->numerical_proof_owned_bytes==(4u<<20));
  if(opt->hybrid_gpu && (!envelope || opt->context.n_threads!=1 || opt->context.n_threads_batch!=1 || plan.records!=114 || plan.experts!=64 || plan.layers!=40 || plan.max_matrix!=2580480))throw std::invalid_argument("hybrid exact tuple/metadata/profile budget");
  out->gpu_=opt->hybrid_gpu;
  opt->admit_budget(plan,opt->context); // mandatory BEFORE model/context allocation
  ua50::OwnedExpertLoad owned;owned.gpu=out->gpu_;if(owned.gpu)owned.admit_hybrid=[opt](const ua50::ExpertLoadPlan&p){opt->admit_budget(p,opt->context);};owned.slots=opt->slots;owned.cache_ceiling=opt->cache_ceiling;owned.stage_ceiling=opt->stage_ceiling;
  owned.input.file=input->file;owned.input.file_anchor=input->file_anchor;
  owned.input.contract=ua50::InputContract::windows_held_verified;
  owned.input.admission_anchor=input->admission_anchor;owned.input.validate_binding=input->validate_binding;
  owned.input.make_reader=[input](size_t stage)->std::shared_ptr<ua50::Reader>{return input->make_reader(stage);};
  if(out->phase_check_)out->phase_check_("before-model-load");
  out->model_.reset(ua50::load_owned_experts(owned,opt->model)); // common factory installs BEFORE load_tensors
  if(!out->model_)throw std::runtime_error("owned common model load refused");
  if(out->phase_check_)out->phase_check_("after-weights-before-context");
  if(opt->route_trace){auto* m=dynamic_cast<llama_model_base*>(out->model_.get());
   if(!m || !m->expert_provider())throw std::runtime_error("trace provider missing");
   m->expert_provider()->set_trace(opt->route_trace);}
  if(opt->cache_events){auto* m=dynamic_cast<llama_model_base*>(out->model_.get());
   if(!m || !m->expert_provider())throw std::runtime_error("cache events provider missing");
   m->expert_provider()->enable_timing();}
  if(opt->hybrid_catalog){auto*m=dynamic_cast<llama_model_base*>(out->model_.get());if(!m||!m->expert_provider())throw std::runtime_error("hybrid registry unavailable");m->expert_provider()->validate_hybrid_catalog();m->expert_provider()->visit_registry(opt->hybrid_catalog);}
  out->context_.reset(llama_init_from_model(out->model_.get(),opt->context));
  if(!out->context_)throw std::runtime_error("expert context allocation refused");
  auto sched=out->context_->get_sched();
  if(ggml_backend_sched_get_n_backends(sched)!=1 || !ggml_backend_is_cpu(ggml_backend_sched_get_backend(sched,0)))throw std::invalid_argument("one CPU backend required");
  ggml_backend_cpu_set_use_ref(ggml_backend_sched_get_backend(sched,0),true); // nonfused, not all-scalar
  llama_set_abort_callback(out->context_.get(),&CpuExpertSession::should_abort,out.get());
  if(out->phase_check_)out->phase_check_("context-ready");
  out->state_.store(State::Ready,std::memory_order_release);if(opt->effective_observer)opt->effective_observer(out->effective());return out;
 }catch(...){out->latch_failure("expert opening refused");out->state_.store(State::Invalid);out->close();throw;}
}
CpuExpertSession::~CpuExpertSession(){(void)close();}
void CpuExpertSession::begin_fresh_turn(){
 auto keep_alive=shared_from_this();State expected=State::Ready;
 if(!state_.compare_exchange_strong(expected,State::Decoding))throw std::logic_error("invalid expert turn owner");
 try{
  if(stop_.load())throw std::runtime_error("expert owner stopped");
  if(phase_check_)phase_check_("before-turn");
  llama_synchronize(context_.get());llama_memory_clear(llama_get_memory(context_.get()),true);
  state_.store(State::Ready,std::memory_order_release);
 }catch(...){latch_failure("fresh turn reset failed");state_.store(State::Invalid);close();throw;}
}
int CpuExpertSession::decode(llama_batch batch,std::atomic<bool>* turn_abort){
 auto keep_alive=shared_from_this(); // last external owner cannot destroy an active decode
 State expected=State::Ready;
 if(!state_.compare_exchange_strong(expected,State::Decoding))throw std::logic_error("expert owner not ready; fresh admission required");
 try{
  if(stop_.load())throw std::runtime_error("expert owner stopped");
  if(phase_check_)phase_check_("before-decode");
  cache_event("begin");
  turn_abort_=turn_abort;
  int rc=llama_decode(context_.get(),batch);llama_synchronize(context_.get());
  const bool cancelled=should_abort(this);turn_abort_=nullptr;
  auto model=dynamic_cast<llama_model_base*>(model_.get());auto provider=model?model->expert_provider():nullptr;
  if(!provider || provider->active_leases())throw std::runtime_error("expert graph lease barrier incomplete");
  if(gpu_&&(!gpu_->terminal()||gpu_->error()[0])){latch_failure(gpu_->error()[0]?gpu_->error():"hybrid terminal incomplete");throw std::runtime_error("hybrid terminal GPU failure");}
  if(provider->error()[0]){latch_failure(provider->error());throw std::runtime_error("expert provider terminal failure");}
  if(rc || cancelled){
   latch_failure(cancelled?"expert cancellation":"native decode nonzero; no continuation");state_.store(State::Invalid);close();return rc?rc:2;
  }
  cache_event("end");
  log_drained_counters();
  if(phase_check_)phase_check_("after-decode");
  state_.store(State::Ready,std::memory_order_release);return 0;
 }catch(...){turn_abort_=nullptr;latch_failure("expert decode exception; no continuation");state_.store(State::Invalid);close();throw;}
}
void CpuExpertSession::request_stop() noexcept{
 stop_.store(true,std::memory_order_release);
 auto gpu=std::atomic_load(&gpu_);if(gpu)gpu->request_stop();
 auto input=std::atomic_load(&input_);if(input&&input->control)input->control->request_stop();
}
bool CpuExpertSession::close() noexcept{
 State s=state_.load();
 for(;;){if(s==State::Closed)return true;if(s==State::Decoding || s==State::Closing)return false;if(state_.compare_exchange_weak(s,State::Closing))break;}
 request_stop();
 if(context_)llama_synchronize(context_.get()); // wait graph workers before their buffers are released
 if(gpu_){gpu_->drain();if(!gpu_->terminal())std::terminate();}
 auto input=std::atomic_load(&input_);if(input&&input->control)input->control->drain_to_terminal(); // may block forever; never early-free pending I/O
 // Snapshot after actual graph/I/O drain, before provider destruction. Callback
 // failure cannot skip owning teardown and will prevent complete trace sealing.
 try{cache_event("terminal");}catch(...){latch_failure("cache terminal telemetry failed");}
 context_.reset();model_.reset();std::atomic_store(&gpu_,std::shared_ptr<ua50::ExpertGpu>{}); // provider/Reader destroyed before admission/file anchors
 std::atomic_store(&input_,std::shared_ptr<ua50::uwp::WindowsInputCandidate>{});input.reset();
 state_.store(State::Closed,std::memory_order_release);return true;
}
const llama_model* CpuExpertSession::model_view() const{
 if(state()!=State::Ready || stop_.load())throw std::logic_error("invalid expert model view");
 return model_.get();
}
uint32_t CpuExpertSession::context_length() const{model_view();return llama_n_ctx(context_.get());}
uint32_t CpuExpertSession::batch_length() const{model_view();return llama_n_batch(context_.get());}
CpuExpertSession::Effective CpuExpertSession::effective() const {
 model_view();auto*m=dynamic_cast<llama_model_base*>(model_.get());auto p=m?m->expert_provider():nullptr;
 if(!p||p->active_leases())throw std::runtime_error("effective state before native drain");const auto&c=p->counters();
 const auto applied=cpu_pilot::applied_cache(p->slot_count(),applied_cache_ceiling_,p->reserved_budget(),p->requested_owned_bytes(),c.arena_bytes,c.metadata_bytes);
 return {llama_n_ctx(context_.get()),uint32_t(llama_n_threads(context_.get())),uint32_t(llama_n_threads_batch(context_.get())),applied.slots,applied.ceiling,applied.owned,applied.arena,applied.metadata};
}
void CpuExpertSession::primitive_vector(const char*name,uint32_t expert,const unsigned char*q8,size_t bytes,float*cpu,float*gpu,std::shared_ptr<ua50::ExpertGpu> executor){
 auto keep_alive=shared_from_this();State expected=State::Ready;if(!state_.compare_exchange_strong(expected,State::Decoding))throw std::logic_error("primitive owner not ready");
 std::shared_ptr<ua50::NativeProvider> provider;bool leased=false;
 try{
  if(!executor||!name||!cpu||!gpu||!q8||stop_.load())throw std::invalid_argument("primitive source/stop");
  auto*m=dynamic_cast<llama_model_base*>(model_.get());provider=m?m->expert_provider():nullptr;if(!provider)throw std::runtime_error("primitive provider missing");provider->validate_hybrid_catalog();
  auto*t=provider->tensor_named(name);if(!provider->acquire(t,expert))throw std::runtime_error(provider->error());leased=true;
  const auto quant=t->type==GGML_TYPE_Q4_K?ua50::Quant::q4_k:ua50::Quant::q6_k;
  ua50::ExpertVector request{quant,provider->role_id(t),expert,uint32_t(t->ne[0]),uint32_t(t->ne[1]),reinterpret_cast<const unsigned char*>(provider->data(t)),size_t(t->nb[2]),q8,bytes};ua50::check_vector(request);
  auto*trait=ggml_get_type_traits_cpu(t->type);if(!trait||trait->vec_dot_type!=GGML_TYPE_Q8_K||!trait->vec_dot)throw std::runtime_error("primitive actual CPU Q8 oracle unavailable");
  for(size_t row=0;row<request.n;++row)trait->vec_dot(int(request.k),cpu+row,0,request.weights+row*t->nb[1],0,q8,0,1);
  if(stop_.load())throw std::runtime_error("primitive stopped before GPU submit");
  const float*result=executor->execute(request);executor->drain();if(!executor->terminal()||executor->error()[0]||stop_.load())throw std::runtime_error("primitive GPU stopped/failed");
  std::memcpy(gpu,result,request.n*sizeof(float));provider->release();leased=false;state_.store(State::Ready);
 }catch(const std::exception&e){latch_failure(e.what());if(executor){executor->request_stop();executor->drain();}if(leased)provider->release();state_.store(State::Invalid);close();throw;}
 catch(...){latch_failure("primitive unknown terminal failure; no continuation");if(executor){executor->request_stop();executor->drain();}if(leased)provider->release();state_.store(State::Invalid);close();throw;}
}
void CpuExpertSession::capture_logits(float* destination,size_t elements) const {
 model_view();if(!destination||elements!=128000||llama_vocab_n_tokens(llama_model_get_vocab(model_.get()))!=128000)throw std::invalid_argument("numerical proof full-row extent");
 const float* row=llama_get_logits_ith(context_.get(),-1);if(!row)throw std::runtime_error("native logits unavailable");
 std::memcpy(destination,row,elements*sizeof(float));
}
std::vector<llama_token> CpuExpertSession::tokenize(const std::string&s) const{
 const auto*v=llama_model_get_vocab(model_view());
 if(s.size()>INT32_MAX)throw std::length_error("prompt bytes overflow");
 int n=llama_tokenize(v,s.data(),static_cast<int32_t>(s.size()),nullptr,0,true,true);
 if(n==INT32_MIN || n>=0)throw std::runtime_error("empty or invalid tokenization");
 std::vector<llama_token> out(static_cast<size_t>(-n));
 int got=llama_tokenize(v,s.data(),static_cast<int32_t>(s.size()),out.data(),-n,true,true);
 if(got<0 || got>-n)throw std::runtime_error("tokenization changed");
 out.resize(got);return out;
}
std::string CpuExpertSession::piece(llama_token t,bool special) const{
 const auto*v=llama_model_get_vocab(model_view());char small[256];
 int n=llama_token_to_piece(v,t,small,sizeof(small),0,special);
 if(n>=0)return std::string(small,n);
 if(n==INT32_MIN)throw std::runtime_error("piece overflow");
 std::string out(static_cast<size_t>(-n),'\0');int got=llama_token_to_piece(v,t,out.data(),-n,0,special);
 if(got<0 || got>-n)throw std::runtime_error("piece changed");
 out.resize(got);return out;
}
llama_token CpuExpertSession::sample(llama_sampler*s){
 if(phase_check_)phase_check_("before-sample");
 model_view();if(!s)throw std::invalid_argument("missing sampler");
 return llama_sampler_sample(s,context_.get(),-1); // raw context stays inside owning controller
}
void CpuExpertSession::invalidate(const char*why) noexcept{
 latch_failure(why);request_stop();State s=state();
 if(s==State::Ready || s==State::Opening)state_.store(State::Invalid);
 (void)close();
}
void CpuExpertSession::cache_event(const char* event) {
 if(!cache_events_)return;
 auto*m=dynamic_cast<llama_model_base*>(model_.get());auto p=m?m->expert_provider():nullptr;
 if(!p)return; // opening failure before provider construction has no requests
 if(p->active_leases())throw std::runtime_error("cache event before lease barrier");
 cache_events_(event,p->counters());
}
void CpuExpertSession::log_drained_counters() const {
 auto *m=dynamic_cast<llama_model_base*>(model_.get());auto provider=m?m->expert_provider():nullptr;
 if(!provider || provider->active_leases())throw std::runtime_error("pilot telemetry before drain");
 const auto& c=provider->counters();char line[512];
 std::snprintf(line,sizeof(line),"[cpu-pilot] cache hits=%llu misses=%llu evictions=%llu logical_reads=%llu logical_bytes=%llu failures=%llu arena=%zu metadata=%zu leases=0\n",
  static_cast<unsigned long long>(c.hits),static_cast<unsigned long long>(c.misses),static_cast<unsigned long long>(c.evictions),static_cast<unsigned long long>(c.read_calls),static_cast<unsigned long long>(c.read_bytes),static_cast<unsigned long long>(c.failures),c.arena_bytes,c.metadata_bytes);
 log_output(line);
}
cpu_pilot::MemoryPlan plan_cpu_expert_memory(FILE* file,const CpuExpertOptIn& opt,const ua50::ExpertLoadPlan& input) {
 cpu_pilot::exact_profile(input.available_file_bytes,input.layers,input.experts,input.records,input.max_matrix);
 if(!file || !input.full_payload || !input.routed_tensor_bytes || opt.context.n_ctx!=opt.planning_context() || opt.context.n_batch!=1 || opt.context.n_ubatch!=1 || opt.context.n_threads!=1 || opt.context.n_threads_batch!=1 || opt.context.n_seq_max!=1 || opt.context.n_outputs_max!=1 || opt.context.n_samplers || opt.context.embeddings)
  throw std::invalid_argument("pilot sizing only covers exact one-token CPU profile");
 struct Position { FILE*f;fpos_t pos;explicit Position(FILE*p):f(p){if(fgetpos(f,&pos))throw std::runtime_error("sizing FILE position");}~Position(){fsetpos(f,&pos);} } restore(file);
 if(fseek(file,0,SEEK_SET))throw std::runtime_error("sizing FILE rewind");
 auto mp=opt.model;mp.no_alloc=true;
 std::unique_ptr<llama_model,decltype(&llama_model_free)> model(llama_model_load_from_file_ptr(file,mp),&llama_model_free);
 if(!model || !model->hparams.no_alloc)throw std::runtime_error("native no_alloc sizing refused");
 std::unique_ptr<llama_context,decltype(&llama_free)> ctx(llama_init_from_model(model.get(),opt.context),&llama_free);
 if(!ctx)throw std::runtime_error("native context sizing refused");
 auto sched=ctx->get_sched();
 if(ggml_backend_sched_get_n_backends(sched)!=1 || !ggml_backend_is_cpu(ggml_backend_sched_get_backend(sched,0)))throw std::runtime_error("sizing requires one CPU");
 ggml_backend_cpu_set_use_ref(ggml_backend_sched_get_backend(sched,0),true);
 auto memory=ctx->get_memory();auto mctx=memory?memory->init_full():llama_memory_context_ptr{};
 size_t reference_compute=0;
 auto *graph=ctx->graph_reserve(1,1,1,mctx.get(),true,&reference_compute);
 if(!graph)throw std::runtime_error("nonfused graph sizing refused");
 auto work=ggml_graph_plan(graph,1,nullptr);
 cpu_pilot::MemoryPlan result;
 for(const auto& pair:llama_get_memory_breakdown(ctx.get())) {
  result.resident=cpu_pilot::add(result.resident,pair.second.model);
  result.kv_recurrent=cpu_pilot::add(result.kv_recurrent,pair.second.context);
  result.compute=cpu_pilot::add(result.compute,pair.second.compute);
 }
 if(result.resident<input.routed_tensor_bytes)throw std::runtime_error("sizing routed subtraction underflow");
 // Subtract packed routed bytes only: any backend padding remains a conservative charge.
 result.resident-=input.routed_tensor_bytes;
 result.compute=std::max<uint64_t>(result.compute,reference_compute);
 result.cpu_workspace=work.work_size;
 result.output=cpu_pilot::add(0,uint64_t(llama_vocab_n_tokens(llama_model_get_vocab(model.get())))*sizeof(float));
 // Charge bounded collector, JSON/hex serialization and fixed template storage separately.
 if(opt.plan_profile!=CpuExpertOptIn::PlanProfile::Smoke64)result.output=cpu_pilot::add(result.output,
  (opt.plan_profile==CpuExpertOptIn::PlanProfile::FinishedAnswer512||opt.plan_profile==CpuExpertOptIn::PlanProfile::GenerationBounded)?2u*1024u*1024u:1024u*1024u);
 result.output=cpu_pilot::add(result.output,opt.instrumentation_owned_bytes);
 if(opt.numerical_proof_owned_bytes){if(opt.numerical_proof_owned_bytes!=(4u<<20))throw std::invalid_argument("proof fixed owned cap");result.output=cpu_pilot::add(result.output,opt.numerical_proof_owned_bytes);}
 if(opt.hybrid_gpu){if(!opt.hybrid_gpu->terminal())throw std::runtime_error("GPU plan before terminal");result.output=cpu_pilot::add(result.output,opt.hybrid_gpu->required_owned_bytes());}
 result.cache=input.cache_owned_bytes;result.stage=input.stage_bytes;
 if(fsetpos(file,&restore.pos))throw std::runtime_error("sizing FILE restore");
 (void)result.total();return result;
}

}
