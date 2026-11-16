#pragma once
#include "ua50/expert_gpu.h"
#include <array>
#include <cstring>
#include <string>
namespace xllama::gpu_proof {
// New binary, no CPU UA50TRC1/2 bytes or reserved-field reinterpretation.
class Trace {
 static constexpr size_t header=512,roles=header,frames=roles+114*48,keys=frames+258*48,primitives=keys+117648*2,footer=primitives+16*32;
 std::array<unsigned char,footer+256> data_{};std::array<bool,114> known_{};
 size_t frame_count_=0,key_count_=0,primitive_count_=0;int current_=-1;uint32_t phase_=0;bool overflow_=false,sealed_=false;
 ua50::GpuCounts before_{};
 void put(size_t at,uint64_t v,size_t bytes){if(at+bytes>data_.size())throw std::length_error("GPU trace bounds");for(size_t i=0;i<bytes;++i)data_[at+i]=static_cast<unsigned char>(v>>(8*i));}
 uint64_t get(size_t at,size_t bytes)const{uint64_t n=0;for(size_t i=0;i<bytes;++i)n|=uint64_t(data_[at+i])<<(8*i);return n;}
 void digest(size_t at,const std::string&s,size_t bytes){if(s.size()!=bytes*2)throw std::invalid_argument("trace identity digest");for(size_t i=0;i<bytes;++i){unsigned n=0;for(size_t j=0;j<2;++j){char c=s[i*2+j];unsigned d=c>='0'&&c<='9'?c-'0':c>='a'&&c<='f'?c-'a'+10:99;if(d>15)throw std::invalid_argument("trace identity hex");n=n*16+d;}data_[at+i]=static_cast<unsigned char>(n);}}
 static uint64_t fnv(const unsigned char*p,size_t n){uint64_t h=14695981039346656037ull;for(size_t i=0;i<n;++i){h^=p[i];h*=1099511628211ull;}return h;}
public:
 static constexpr size_t file_bytes=footer+256,capacity_bytes=524288;
 Trace(uint32_t pid,uint32_t selected_stage,const std::string&run,const std::string&config,const std::string&ticket,const std::string&engine,const std::string&shader){
  if(!pid||(selected_stage!=3&&selected_stage!=4))throw std::invalid_argument("GPU trace stage/PID");std::memcpy(data_.data(),"UA68TRC1",8);put(8,1,4);put(12,file_bytes,4);put(16,pid,4);put(20,selected_stage,4);put(24,512,4);put(28,1,4);put(32,2,4);put(36,114,4);digest(64,run,16);digest(80,config,32);digest(112,ticket,32);digest(144,engine,32);digest(176,shader,32);
 }
 static constexpr size_t required_owned_bytes(){return sizeof(Trace);}
 void role(uint32_t id,const ua50::Tensor&t,const char*name){
  if(sealed_||id>=114||t.experts!=64||!name)throw std::invalid_argument("GPU trace registry");size_t a=roles+id*48;std::array<unsigned char,48> prior{};std::memcpy(prior.data(),data_.data()+a,48);
  put(a,id,2);put(a+2,t.quant==ua50::Quant::q4_k?1:t.quant==ua50::Quant::q6_k?2:0,2);put(a+4,t.columns,4);put(a+8,t.rows,4);put(a+12,t.experts,4);put(a+16,t.offset,8);put(a+24,t.nb02,8);const auto name_hash=fnv(reinterpret_cast<const unsigned char*>(name),std::strlen(name));for(size_t other=0;other<114;++other)if(other!=id&&known_[other]&&get(roles+other*48+32,8)==name_hash)throw std::invalid_argument("trace name hash collision");put(a+32,name_hash,8);put(a+40,1,8);
  if(known_[id]&&std::memcmp(prior.data(),data_.data()+a,48))throw std::invalid_argument("CPU/hybrid actual registry differs");known_[id]=true;
 }
 void begin(uint32_t phase,uint32_t position,ua50::GpuCounts before={}){
  if(sealed_||current_>=0||frame_count_==258||(phase!=1&&phase!=2))throw std::length_error("GPU frame phase/bounds");phase_=phase;current_=int(frame_count_++);before_=before;auto a=frames+size_t(current_)*48;put(a,phase,4);put(a+4,position,4);put(a+8,key_count_,4);
 }
 void key(uint32_t id,uint64_t expert,uint64_t offset,uint64_t bytes){
  if(sealed_||id>=114||expert>=64||!known_[id]||offset!=get(roles+id*48+16,8)+expert*get(roles+id*48+24,8)||bytes!=get(roles+id*48+24,8))throw std::invalid_argument("GPU trace exact acquire extent");
  if(phase_==3)return; // primitive request recorded explicitly after its terminal comparator row
  if(current_<0||key_count_==117648){overflow_=true;throw std::length_error("GPU ordered key cap");}put(keys+key_count_*2,id*64+expert,2);++key_count_;
 }
 uint32_t id_for(const char*name)const{uint64_t h=fnv(reinterpret_cast<const unsigned char*>(name),std::strlen(name));for(uint32_t i=0;i<114;++i)if(known_[i]&&get(roles+i*48+32,8)==h)return i;throw std::invalid_argument("named primitive role absent");}
 void primitive_begin(){if(current_>=0||sealed_)throw std::logic_error("primitive phase busy");phase_=3;}
 void primitive_end(uint32_t ordinal,uint32_t role,uint32_t expert,uint32_t pattern,uint32_t elements,const ua50::GpuCounts&before,const ua50::GpuCounts&after){
  if(primitive_count_==16||ordinal!=primitive_count_||after.submits-before.submits!=1||after.completed-before.completed!=1||!after.terminal)throw std::runtime_error("primitive actual dispatch/terminal mismatch");size_t a=primitives+primitive_count_++*32;put(a,ordinal,4);put(a+4,role*64+expert,4);put(a+8,pattern,4);put(a+12,elements,4);put(a+16,1,8);put(a+24,1,8);
 }
 void end(const ua50::GpuCounts&after={}){
  if(current_<0)throw std::logic_error("GPU frame end without start");size_t a=frames+size_t(current_)*48;const auto n=key_count_-get(a+8,4);put(a+12,n,4);if(n!=456)throw std::runtime_error("GPU full role route coverage");
  if(phase_==2){if(after.submits<before_.submits||after.completed<before_.completed||after.published<before_.published||after.submits-before_.submits!=456||after.completed-before_.completed!=456||after.published-before_.published!=456||!after.terminal)throw std::runtime_error("GPU actual dispatched role coverage");put(a+16,456,8);put(a+24,456,8);put(a+32,456,8);}
  put(a+40,1,8);current_=-1;
 }
 struct Routes {bool available=false,equal=false;uint32_t frame=0,key=0,cpu_key=0,gpu_key=0;};
 Routes compare_model_routes(uint32_t prompt_count)const{
  Routes out;const size_t n=size_t(prompt_count)+1;if(!prompt_count||prompt_count>128||frame_count_!=2*n||current_>=0)return out;
  out.available=true;out.equal=true;for(size_t i=0;i<n;++i){size_t a=frames+i*48,b=frames+(i+n)*48;
   if(get(a,4)!=1||get(b,4)!=2||get(a+4,4)!=get(b+4,4)||get(a+12,4)!=456||get(b+12,4)!=456){out.available=false;out.equal=false;return out;}
   const auto ka=get(a+8,4),kb=get(b+8,4);for(size_t j=0;j<456;++j)if(get(keys+(ka+j)*2,2)!=get(keys+(kb+j)*2,2)){out.equal=false;out.frame=uint32_t(i);out.key=uint32_t(j);out.cpu_key=uint32_t(get(keys+(ka+j)*2,2));out.gpu_key=uint32_t(get(keys+(kb+j)*2,2));return out;}}
  return out;
 }
 void seal(bool normal,uint32_t prompt_count,int32_t t0,int32_t t1,bool proof_complete){
  if(sealed_)throw std::logic_error("trace seal twice");bool registry=true;for(bool k:known_)registry=registry&&k;
  const uint32_t stage=uint32_t(get(20,4));const bool shape=stage==3?(primitive_count_==16&&frame_count_==0&&key_count_==0):(primitive_count_==0&&prompt_count>0&&prompt_count<=128&&frame_count_==2*(prompt_count+1)&&key_count_==frame_count_*456);
  const auto route=compare_model_routes(prompt_count);const bool route_ok=stage==3||(route.available&&route.equal);
  const bool complete=normal&&proof_complete&&registry&&shape&&route_ok&&!overflow_&&current_<0;
  put(40,prompt_count,4);put(44,uint32_t(t0),4);put(48,uint32_t(t1),4);put(footer,complete?1:0,8);put(footer+8,frame_count_,8);put(footer+16,key_count_,8);put(footer+24,primitive_count_,8);put(footer+32,overflow_?1:0,8);sealed_=true;
 }
 bool complete()const{return sealed_&&get(footer,8)==1;}
 const unsigned char*data()const{return data_.data();}
 uint64_t whole_fnv()const{return fnv(data_.data(),data_.size());}
};
static_assert(Trace::file_bytes<=Trace::capacity_bytes,"GPU trace cap");
}
