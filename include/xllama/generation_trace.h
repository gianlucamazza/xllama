#pragma once
#include "ua50/expert_gpu.h"
#include "generation_profile.h"
#include <array>
#include <cstring>
namespace xllama::generation {
class Trace {
 static constexpr size_t roles=256,frames=roles+114*16,keys=frames+511*96,footer=keys+233016*2;
 std::array<unsigned char,footer+128> data_{};std::array<bool,114> known_{};
 size_t nf_=0,nk_=0;bool inside_=false,sealed_=false,overflow_=false,hybrid_;ua50::Counters before_{};ua50::GpuCounts gpu_before_{};
 void put(size_t at,uint64_t v,size_t n){if(at+n>data_.size())throw std::length_error("generation trace bound");for(size_t i=0;i<n;++i)data_[at+i]=static_cast<unsigned char>(v>>(8*i));}
 uint64_t get(size_t at,size_t n)const{uint64_t v=0;for(size_t i=0;i<n;++i)v|=uint64_t(data_[at+i])<<(8*i);return v;}
 void hex(size_t at,const std::string&s,size_t n){if(!dev_run::hex(s,n*2))throw std::invalid_argument("generation trace digest");for(size_t i=0;i<n;++i){auto d=[](char c){return c<='9'?c-'0':c-'a'+10;};data_[at+i]=static_cast<unsigned char>(d(s[i*2])*16+d(s[i*2+1]));}}
 static uint64_t delta(uint64_t a,uint64_t b){if(b<a)throw std::runtime_error("generation counter reset");return b-a;}
public:
 static constexpr size_t file_bytes=footer+128,cap=524288;
 static size_t owned_bytes(){return sizeof(Trace);}
 Trace(uint32_t pid,const Request&r,const std::string&config,const std::string&permit):hybrid_(r.profile.hybrid){
  std::memcpy(data_.data(),"UA68GEN1",8);put(8,1,4);put(12,file_bytes,4);put(16,pid,4);put(20,hybrid_?1:0,4);put(24,r.profile.context,4);put(28,1,4);put(32,r.profile.slots,4);put(36,r.profile.max_new,4);hex(64,r.id,16);hex(80,config,32);hex(112,permit,32);hex(144,contract_sha,32);
 }
 void effective_gpu(const ua50::GpuSnapshot&g){if(!hybrid_)throw std::logic_error("CPU trace GPU identity");auto s=g.adapter_luid;dev_run::need(s.size()==17&&s[8]==':',"trace actual adapter");hex(176,s.substr(0,8)+s.substr(9),8);hex(184,g.shader_sha256,32);}
 void role(uint32_t id,const ua50::Tensor&t,const char*){if(sealed_||id>=114||known_[id]||t.experts!=64||!t.nb02)throw std::runtime_error("generation role registry");known_[id]=true;put(roles+id*16,id,4);put(roles+id*16+4,t.quant==ua50::Quant::q4_k?1:t.quant==ua50::Quant::q6_k?2:0,4);put(roles+id*16+8,t.nb02,8);}
 void begin(const ua50::Counters&c,const ua50::GpuCounts&g,bool prefill){if(sealed_||inside_||nf_==511)throw std::length_error("generation frame bound");inside_=true;before_=c;gpu_before_=g;size_t a=frames+nf_*96;put(a,nf_,4);put(a+4,prefill?0:1,4);put(a+8,nk_,4);}
 void key(uint32_t role,uint64_t expert,uint64_t bytes){if(!inside_||sealed_||role>=114||expert>=64||!known_[role]||bytes!=get(roles+role*16+8,8)||nk_==233016){overflow_=true;throw std::runtime_error("generation trace key coverage/bound");}put(keys+nk_++*2,role*64+expert,2);}
 void end(const ua50::Counters&c,const ua50::GpuCounts&g){if(!inside_||sealed_)throw std::logic_error("generation trace end");size_t a=frames+nf_*96;auto start=get(a+8,4),n=nk_-start;if(n!=456||delta(before_.requests,c.requests)!=n)throw std::runtime_error("generation full request coverage");
  std::array<unsigned,114>counts{};std::array<int,114>last{};last.fill(-1);for(size_t i=start;i<nk_;++i){auto k=get(keys+i*2,2),r=k/64,e=k%64;if(last[r]>=0&&e<=uint64_t(last[r]))throw std::runtime_error("generation ascending experts");last[r]=int(e);++counts[r];}for(auto x:counts)if(x!=4)throw std::runtime_error("generation role coverage");
  uint64_t hit=delta(before_.hits,c.hits),miss=delta(before_.misses,c.misses);if(hit+miss!=456)throw std::runtime_error("generation hit conservation");put(a+12,n,4);size_t off=16;for(auto x:{hit,miss,delta(before_.read_calls,c.read_calls),delta(before_.read_bytes,c.read_bytes),delta(before_.reader_ns_total,c.reader_ns_total),delta(before_.acquire_ns_total,c.acquire_ns_total),delta(gpu_before_.submits,g.submits),delta(gpu_before_.completed,g.completed),delta(gpu_before_.published,g.published)}){put(a+off,x,8);off+=8;}
  if(hybrid_&&(get(a+64,8)!=456||get(a+72,8)!=456||get(a+80,8)!=456||!g.terminal))throw std::runtime_error("generation GPU frame actual coverage");if(!hybrid_&&(get(a+64,8)||get(a+72,8)||get(a+80,8)))throw std::runtime_error("CPU generation GPU counter drift");put(a+88,1,8);inside_=false;++nf_;
 }
 void seal(bool normal,uint32_t prompt,uint32_t sampled){if(sealed_)throw std::logic_error("generation trace reseal");bool registry=true;for(auto x:known_)registry=registry&&x;bool full=normal&&registry&&!inside_&&!overflow_&&prompt>0&&sampled>0&&nf_==uint64_t(prompt)+sampled-1&&nk_==nf_*456;put(40,prompt,4);put(44,sampled,4);put(footer,full?1:0,8);put(footer+8,nf_,8);put(footer+16,nk_,8);put(footer+24,overflow_?1:0,8);sealed_=true;}
 bool complete()const{return sealed_&&get(footer,8)==1;}
 const unsigned char*data()const{return data_.data();}
 uint64_t fnv()const{uint64_t h=14695981039346656037ull;for(auto c:data_){h^=c;h*=1099511628211ull;}return h;}
};
static_assert(Trace::file_bytes==517296&&sizeof(Trace)<524288,"generation bounded trace allocation");
}
