bool Session::expert_requested(const SessionParams& sp) noexcept {
#if defined(XLLAMA_UWP) && defined(XLLAMA_USE_LLAMA)
    return sp.cpu_experts && sp.cpu_experts->enabled;
#else
    (void)sp;
    return false;
#endif
}

std::unique_ptr<Session> Session::create(const SessionParams& sp, std::string* err) {
    if(sp.cpu_experts) {
#if defined(XLLAMA_UWP) && defined(XLLAMA_USE_LLAMA)
        if(sp.cpu_experts->enabled) {
            try {
                if(sp.backend!=Backend::LlamaCpp || sp.n_gpu_layers!=0 || !sp.lora_path.empty() || sp.kv_q8 || sp.prompt_lookup || sp.n_ctx<=0 ||
                   static_cast<uint32_t>(sp.n_ctx)!=sp.cpu_experts->context.n_ctx || sp.n_threads!=sp.cpu_experts->context.n_threads ||
                   (sp.n_batch>0 && static_cast<uint32_t>(sp.n_batch)!=sp.cpu_experts->context.n_batch) ||
                   (sp.n_ubatch>0 && static_cast<uint32_t>(sp.n_ubatch)!=sp.cpu_experts->context.n_ubatch))
                    throw std::invalid_argument("expert Session load policy mismatch");
                auto owner=CpuExpertSession::open_if_requested(sp.cpu_experts.get());
                if(!owner)throw std::runtime_error("enabled expert owner absent");
                return std::make_unique<CpuExpertAppSession>(std::move(owner));
            }catch(const std::exception& e){if(err)*err=e.what();return nullptr;}
            catch(...){if(err)*err="unknown expert opening failure";return nullptr;}
        }
#else
        // Non-null opt-in cannot silently select an unsupported backend/build.
        if(err)*err="CPU expert input is unsupported by this build";
        return nullptr;
#endif
    }

#if defined(XLLAMA_USE_ORT) && defined(XLLAMA_USE_LLAMA)
    Backend b = sp.backend;
    if (b == Backend::Auto) {
        // Layout-aware: uses suffix fast-path + resolve + directory scan so that
        // bare catalogue names (e.g. "qwen35-0.8b" pointing at a dir with .gguf)
        // are classified correctly. Explicit Backend from manifest (MainPage) or
        // tests still takes precedence.
        b = model_uses_llama_backend(sp.model_path) ? Backend::LlamaCpp : Backend::OrtGenAI;
    }
    return b == Backend::LlamaCpp ? detail::create_llama(sp, err) : detail::create_ort(sp, err);
#elif defined(XLLAMA_USE_ORT)
    return detail::create_ort(sp, err);
#else // XLLAMA_USE_LLAMA
    return detail::create_llama(sp, err);
#endif
}

