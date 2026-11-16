#pragma once
#include <cstdint>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>

namespace xllama::cpu_pilot {
// Built-in developer pilot pin. This is not a signed/updateable catalogue.
inline constexpr char id[] = "lfm25-8b-a1b-q4-k-m";
inline constexpr wchar_t file[] = L"LFM2.5-8B-A1B-Q4_K_M.gguf";
inline constexpr uint64_t bytes = 5155564768ull;
inline constexpr char sha[] = "4923ec14f06b968b74d663e5949867d2d9c3bf13a20b8be1a9f9af39989b2bb0";
inline constexpr uint64_t reserve = 512ull*1024*1024;
inline constexpr uint64_t probe_allowance = 256ull*1024*1024;
inline uint64_t add(uint64_t a,uint64_t b) {
    if(b>std::numeric_limits<uint64_t>::max()-a)throw std::overflow_error("pilot memory sum");
    return a+b;
}
struct Quota { uint64_t limit=0,usage=0; bool pressure=false; };
// Every read is an in-process MemoryManager snapshot, never Portal WS/host RAM.
inline void check(const Quota& q,uint64_t initial_limit,uint64_t pending=0) {
    if(!initial_limit || q.limit!=initial_limit || q.pressure || q.usage>q.limit)
        throw std::runtime_error("pilot quota unknown/changed/pressure");
    if(add(pending,reserve)>q.limit-q.usage)
        throw std::runtime_error("pilot joint memory headroom refused");
}
struct MemoryPlan {
    uint64_t resident=0,kv_recurrent=0,compute=0,cpu_workspace=0,output=0,cache=0,stage=0;
    uint64_t total() const {
        uint64_t n=0;
        for(auto x:{resident,kv_recurrent,compute,cpu_workspace,output,cache,stage})n=add(n,x);
        return n;
    }
};
struct AppliedCache {size_t slots,ceiling,owned,arena,metadata;};
// The provider reserves its computed owned bytes; policy ceiling is a separate
// enforced admission limit. Never mistake equality to the ceiling for application.
inline AppliedCache applied_cache(size_t slots,size_t policy_ceiling,size_t provider_budget,size_t owned,size_t arena,size_t metadata){
 if(!slots||slots>1024||!policy_ceiling||!owned||provider_budget<owned||provider_budget>policy_ceiling||add(arena,metadata)>owned)
  throw std::invalid_argument("constructed cache budget/policy mismatch");
 return {slots,policy_ceiling,owned,arena,metadata};
}
inline void exact_profile(uint64_t size,uint32_t layers,uint32_t experts,size_t roles,size_t matrix) {
    if(size!=bytes || layers!=40 || experts!=64 || roles!=114 || matrix!=2580480)
        throw std::invalid_argument("pilot exact LFM24 metadata mismatch");
}
} // namespace xllama::cpu_pilot
