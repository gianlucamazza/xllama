// Internal opt-in Session adapter. Ordinary Session factory remains unchanged when null/disabled.
// No raw context or ordinary decode loop is reachable in this branch.
class CpuExpertAppSession final : public Session {
    std::shared_ptr<CpuExpertSession> owner_;
public:
    explicit CpuExpertAppSession(std::shared_ptr<CpuExpertSession> owner):owner_(std::move(owner)){}
    ~CpuExpertAppSession() override { owner_->request_stop(); (void)owner_->close(); }
    bool reusable() const override { return owner_->state()==CpuExpertSession::State::Ready; }
    void request_stop() noexcept override { owner_->request_stop(); }
    bool close_expert_terminal() noexcept override { owner_->request_stop();return owner_->close(); }
    int context_length() const override { return static_cast<int>(owner_->context_length()); }
    const void* resident_llama_model() const override { return owner_->model_view(); }
    int count_tokens(const std::string& s) override {
        try { return static_cast<int>(owner_->tokenize(s).size()); }
        catch(...) { owner_->invalidate("token count failed"); throw; }
    }
    InferenceResult generate(const GenerateParams& gp) override {
        InferenceResult result;
        try {
            const auto trace=gp.expert_answer;
            if(trace && (gp.n_predict!=static_cast<int>(trace->max_new()) || gp.temperature!=0 || !gp.stop_sequences.empty()))throw std::invalid_argument("first-answer exact greedy/length contract");
            // First candidate is full plain prompt only: never silently degrade tools/KV modes.
            if(gp.reuse_kv || gp.reset_kv || !gp.grammar.empty() || !gp.preserved_tokens.empty() || gp.n_predict<=0)
                throw std::invalid_argument("expert Session requires full plain prompt; KV/tools unsupported");
            if(gp.abort_flag && gp.abort_flag->load())throw std::runtime_error("turn cancelled");
            std::atomic<bool> local_abort{false};
            auto* active_abort=gp.abort_flag?gp.abort_flag:&local_abort;
            auto progress=gp.progress();
            if(progress.stall_timeout_s<=0 || progress.stall_timeout_s>180)throw std::invalid_argument("expert watchdog must remain bounded <=180s");
            InferenceWatchdog watchdog(progress,active_abort);
            owner_->begin_fresh_turn();
            auto tokens=owner_->tokenize(gp.prompt); // ordinary Session flags: add_special=true, parse_special=true
            const size_t capacity=owner_->context_length();
            if(tokens.empty() || tokens.size()>=capacity || static_cast<size_t>(gp.n_predict)>capacity-tokens.size())
                throw std::length_error("full prompt plus output does not fit; no trim");
            const auto* vocab=llama_model_get_vocab(owner_->model_view());
            LlamaSamplerPtr sampler(llama_sampler_chain_init(llama_sampler_chain_default_params()));
            if(!sampler)throw std::runtime_error("sampler allocation failed");
            add_sampler_stages(sampler.get(),gp.sampling(),vocab); // same sampler stages as ordinary Session
            struct Batch {
                llama_batch value;
                explicit Batch(int n):value(llama_batch_init(n,0,1)) {
                    if(!value.token || !value.pos || !value.n_seq_id || !value.seq_id || !value.logits){llama_batch_free(value);throw std::bad_alloc();}
                }
                ~Batch(){llama_batch_free(value);}
            } batch(static_cast<int>(std::min<size_t>(tokens.size(),owner_->batch_length())));
            auto decode=[&](const llama_token* ids,size_t n,size_t position){
                if(active_abort->load())throw std::runtime_error("turn cancelled");
                batch.value.n_tokens=static_cast<int32_t>(n);
                for(size_t i=0;i<n;++i){
                    batch.value.token[i]=ids[i];batch.value.pos[i]=static_cast<llama_pos>(position+i);
                    batch.value.n_seq_id[i]=1;batch.value.seq_id[i][0]=0;batch.value.logits[i]=(i+1==n);
                }
                if(owner_->decode(batch.value,active_abort)!=0)throw std::runtime_error("terminal expert decode failure");
            };
            if(trace)trace->begin(gp.prompt,tokens,static_cast<int>(capacity));
            const auto start=std::chrono::steady_clock::now();
            if(gp.on_status)gp.on_status("cpu-expert prefill");
            watchdog.enter_prefill(static_cast<int>(tokens.size()));
            for(size_t p=0;p<tokens.size();){size_t n=std::min<size_t>(owner_->batch_length(),tokens.size()-p);decode(tokens.data()+p,n,p);p+=n;watchdog.note_prefill_chunk(static_cast<int>(p));}
            const auto prefilled=std::chrono::steady_clock::now();
            if(trace)trace->prefilled();
            result.n_prompt=result.n_p_eval=static_cast<int>(tokens.size());
            result.t_p_eval_ms=std::chrono::duration<double,std::milli>(prefilled-start).count();
            watchdog.prefill_done(result.n_prompt,result.t_p_eval_ms);watchdog.enter_decode(gp.n_predict);
            if(gp.on_status)gp.on_status("cpu-expert decode");
            const char* trace_stop="length";
            for(int i=0;i<gp.n_predict;++i){
                if(active_abort->load())throw std::runtime_error("turn cancelled");
                const auto sample_begin=trace?trace->tick():int64_t(-1);
                const llama_token token=owner_->sample(sampler.get());
                const auto sample_end=trace?trace->tick():int64_t(-1);
                const bool eog=llama_vocab_is_eog(vocab,token);
                std::string piece;
                if(trace){piece=owner_->piece(token,true);trace->sampled(token,piece,eog,sample_begin,sample_end);}
                if(eog){trace_stop="eos_or_eot";break;}
                if(!trace)piece=owner_->piece(token);
                result.output_text+=piece;++result.n_eval;watchdog.on_decode_token(result.n_eval);
                if(result.n_eval==1)result.t_first_token_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
                bool stopped=false;
                for(const auto& stop:gp.stop_sequences){
                    if(stop.empty())continue;
                    const auto at=result.output_text.find(stop);
                    if(at!=std::string::npos){result.output_text.resize(at);result.ended_with_stop=true;stopped=true;break;}
                }
                if(stopped){trace_stop="stop_sequence";break;}
                // Enabled expert mode publishes only a completed successful turn.
                // llama_sampler_sample already accepts the token in the pinned API.
                if(i+1<gp.n_predict){
                    if(trace)trace->decode_begin();
                    decode(&token,1,tokens.size()+static_cast<size_t>(i));
                    if(trace)trace->decode_end();
                }
            }
            result.t_eval_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-prefilled).count();
            if(active_abort->load())throw std::runtime_error("turn cancelled before publication");
            if(gp.on_token && !result.output_text.empty())gp.on_token(result.output_text);
            if(trace)trace->finish(trace_stop);
            result.success=true;
        }catch(const std::exception& e){
            if(gp.expert_answer)gp.expert_answer->fail(e.what());
            owner_->invalidate(e.what());result.success=false;result.output_text.clear();result.error_msg=e.what();
        }catch(...){
            if(gp.expert_answer)gp.expert_answer->fail("unknown Session failure");
            owner_->invalidate("unknown Session failure");result.success=false;result.output_text.clear();result.error_msg="unknown expert Session failure";
        }
        return result;
    }
};

