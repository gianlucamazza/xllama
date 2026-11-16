#pragma once
#include <array>
#include <algorithm>
#include <utility>
#include <cmath>
#include <cstdio>
#include <charconv>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string_view>
namespace xllama::gpu_proof {
static_assert(sizeof(float)==4&&std::numeric_limits<float>::is_iec559,"proof requires IEEE F32");
inline constexpr size_t vocab=128000,primitive_rows=16,model_rows=2,max_rows=18,proof_cap=4u<<20,summary_cap=65536;
inline constexpr double absolute_tolerance=.001,relative_tolerance=.001;
struct Witness {uint32_t index=0,cpu_bits=0,gpu_bits=0;double error=0,tolerance=0;bool valid=false,nonfinite=false;};
struct Row {
 uint32_t stage=0,ordinal=0,role=0,expert=0,pattern=0,position=0;
 uint64_t expected=0,visited=0,compared=0,finite_cpu=0,finite_gpu=0,finite_pairs=0,violations=0;
 double maxabs=0,squared=0,compensation=0;Witness worst[2]{},first{};bool complete=false;
};
class SummaryWriter {
 char*p_;size_t size_=0,cap_;
public:
 SummaryWriter(char*p,size_t cap):p_(p),cap_(cap){}
 void text(std::string_view s){if(s.size()>cap_-size_)throw std::length_error("numeric summary cap");std::memcpy(p_+size_,s.data(),s.size());size_+=s.size();}
 void number(uint64_t n){char x[32];auto s=std::snprintf(x,sizeof(x),"%llu",static_cast<unsigned long long>(n));if(s<0||size_t(s)>=sizeof(x))throw std::runtime_error("numeric serialization");text({x,size_t(s)});}
 void real(double d){if(!std::isfinite(d)){text("null");return;}char x[64];auto r=std::to_chars(x,x+sizeof(x),d,std::chars_format::general,17);if(r.ec!=std::errc{})throw std::runtime_error("numeric serialization");text({x,size_t(r.ptr-x)});}
 void quoted_hex32(uint32_t b){char x[11];std::snprintf(x,sizeof(x),"\"%08x\"",b);text(x);}
 std::string_view result()const{return {p_,size_};}
};
// One concrete storage owner. Native context/output/GPU allocation is charged separately.
enum class Stage{Primitive16,Model2};
class Proof final {
 Stage stage_;
 std::array<float,vocab*2> reference_{};
 std::array<float,vocab> working_{};
 std::array<float,4096> primitive_{};
 std::array<float,2048> f32_input_{};
 alignas(64) std::array<unsigned char,2336> q8_{};
 std::array<char,summary_cap> serialized_{};
 std::array<Row,max_rows> rows_{};size_t count_=0;bool failed_=false,greedy_equal_=true;uint32_t greedy_checks_=0;
 static uint32_t bits(float f){uint32_t u;std::memcpy(&u,&f,4);return u;}
 static bool worse(const Witness&a,const Witness&b){return !b.valid||(a.nonfinite&&!b.nonfinite)||(a.nonfinite==b.nonfinite&&(a.error>b.error||(a.error==b.error&&a.index<b.index)));}
 static void witness(SummaryWriter&w,const Witness&v){if(!v.valid){w.text("null");return;}w.text("{\"index\":");w.number(v.index);w.text(",\"cpu_bits\":");w.quoted_hex32(v.cpu_bits);w.text(",\"gpu_bits\":");w.quoted_hex32(v.gpu_bits);w.text(",\"nonfinite\":");w.text(v.nonfinite?"true":"false");w.text(",\"abs_error\":");if(v.nonfinite)w.text("null");else w.real(v.error);w.text(",\"tolerance\":");if(v.nonfinite)w.text("null");else w.real(v.tolerance);w.text("}");}
public:
 explicit Proof(Stage stage):stage_(stage){if(sizeof(*this)>proof_cap)throw std::length_error("proof actual sizeof over cap");}
 Proof(const Proof&)=delete;Proof&operator=(const Proof&)=delete;
 float* reference(size_t i){if(i>=2)throw std::out_of_range("reference row");return reference_.data()+i*vocab;}
 float* working(){return working_.data();}float* primitive_cpu(){return primitive_.data();}float* primitive_gpu(){return primitive_.data()+2048;}
 float* f32_input(){return f32_input_.data();}unsigned char*q8(){return q8_.data();}
 static constexpr size_t required_owned_bytes(){return proof_cap;} // conservative reserve includes control block/serialization/metadata
 void greedy(bool same){if(stage_!=Stage::Model2||greedy_checks_==2)throw std::logic_error("greedy proof envelope");++greedy_checks_;greedy_equal_=greedy_equal_&&same;if(!same)failed_=true;}
 void compare(uint32_t stage,uint32_t ordinal,uint32_t role,uint32_t expert,uint32_t pattern,uint32_t position,const float*cpu,const float*gpu,size_t n){
  if(count_==max_rows||!cpu||!gpu||!n||(stage==3&&(ordinal>=16||n>2048))||(stage==4&&(ordinal>=2||n!=vocab))||(stage!=3&&stage!=4))throw std::invalid_argument("proof fixed row envelope");
  if((stage_==Stage::Primitive16&&stage!=3)||(stage_==Stage::Model2&&stage!=4)||ordinal!=count_||(stage==3&&n!=(ordinal/4<2?1536u:2048u)))throw std::invalid_argument("numeric fixed stage/order/row size");
  for(size_t k=0;k<count_;++k)if(rows_[k].stage==stage&&rows_[k].ordinal==ordinal)throw std::invalid_argument("duplicate numeric row");
  Row&r=rows_[count_++];r.stage=stage;r.ordinal=ordinal;r.role=role;r.expert=expert;r.pattern=pattern;r.position=position;r.expected=n;
  for(size_t i=0;i<n;++i){++r.visited;const bool a=std::isfinite(cpu[i]),b=std::isfinite(gpu[i]);r.finite_cpu+=a;r.finite_gpu+=b;
   Witness v;v.valid=true;v.index=uint32_t(i);v.cpu_bits=bits(cpu[i]);v.gpu_bits=bits(gpu[i]);v.nonfinite=!(a&&b);
   if(a&&b){++r.finite_pairs;++r.compared;v.error=std::abs(double(gpu[i])-double(cpu[i]));v.tolerance=absolute_tolerance+relative_tolerance*std::abs(double(cpu[i]));
    if(!std::isfinite(v.error)||!std::isfinite(v.tolerance))throw std::overflow_error("numeric difference overflow");
    r.maxabs=std::max(r.maxabs,v.error);double y=v.error*v.error-r.compensation,t=r.squared+y;r.compensation=(t-r.squared)-y;r.squared=t;if(!std::isfinite(t))throw std::overflow_error("numeric RMS overflow");}
   const bool violation=v.nonfinite||v.error>v.tolerance;if(violation){++r.violations;failed_=true;if(!r.first.valid)r.first=v;}
   if(worse(v,r.worst[0])){r.worst[1]=r.worst[0];r.worst[0]=v;}else if(worse(v,r.worst[1]))r.worst[1]=v;
  }
  r.complete=r.visited==r.expected;
 }
 bool complete()const{size_t a=0,b=0;uint64_t na=0,nb=0;for(size_t i=0;i<count_;++i){auto&r=rows_[i];if(!r.complete)return false;if(r.stage==3){++a;na+=r.expected;}else{++b;nb+=r.expected;}}return stage_==Stage::Primitive16?(a==16&&b==0&&na==28672):(a==0&&b==2&&nb==256000);}
 bool passed()const{return complete()&&!failed_&&greedy_equal_&&(stage_==Stage::Primitive16||greedy_checks_==2);}
 // Rawrows are transient only. Returned summary stays owned until caller copies/commits or destroys Proof.
 std::string_view summary(bool owner_terminal,bool final_gate=false){
  SummaryWriter w(serialized_.data(),serialized_.size());w.text("{\"schema\":\"ua68-online-numerical-proof-v1\",\"raw_rows_saved\":false,\"offline_rawrow_replay\":false,\"complete\":");w.text(complete()?"true":"false");w.text(",\"normal_numerical_proof\":");w.text(passed()&&owner_terminal&&final_gate?"true":"false");w.text(",\"owner_terminal\":");w.text(owner_terminal?"true":"false");w.text(",\"greedy_equal\":");if(stage_==Stage::Primitive16||!greedy_checks_)w.text("null");else w.text(greedy_equal_?"true":"false");w.text(",\"greedy_checks\":");w.number(greedy_checks_);w.text(",\"threshold\":{\"absolute\":0.001,\"relative_to_cpu\":0.001},\"expected_rows\":");w.number(stage_==Stage::Primitive16?16:2);w.text(",\"expected_pairs\":");w.number(stage_==Stage::Primitive16?28672:256000);w.text(",\"rows\":[");
  uint64_t compared=0,visited=0,violations=0,fc=0,fg=0,completed_rows=0;double sum=0,comp=0,ma=0;Witness first{};uint32_t firstrow=0;
  for(size_t i=0;i<count_;++i){auto&r=rows_[i];if(i)w.text(",");w.text("{\"stage\":");w.number(r.stage);w.text(",\"ordinal\":");w.number(r.ordinal);w.text(",\"role\":");w.number(r.role);w.text(",\"expert\":");w.number(r.expert);w.text(",\"pattern\":");w.number(r.pattern);w.text(",\"position\":");w.number(r.position);
   for(auto x:{std::pair<const char*,uint64_t>{"expected",r.expected},{"visited",r.visited},{"compared",r.compared},{"finite_cpu",r.finite_cpu},{"finite_gpu",r.finite_gpu},{"finite_pairs",r.finite_pairs},{"violations",r.violations}}){w.text(",\"");w.text(x.first);w.text("\":");w.number(x.second);}
   w.text(",\"maxabs\":");if(r.compared)w.real(r.maxabs);else w.text("null");w.text(",\"rms\":");if(r.compared)w.real(std::sqrt(r.squared/double(r.compared)));else w.text("null");w.text(",\"first_divergence\":");witness(w,r.first);w.text(",\"worst\":[");witness(w,r.worst[0]);w.text(",");witness(w,r.worst[1]);w.text("]}");
   completed_rows+=r.complete;compared+=r.compared;visited+=r.visited;violations+=r.violations;fc+=r.finite_cpu;fg+=r.finite_gpu;double y=r.squared-comp,t=sum+y;comp=(t-sum)-y;sum=t;ma=std::max(ma,r.maxabs);if(!first.valid&&r.first.valid){first=r.first;firstrow=uint32_t(i);}}
  w.text("],\"totals\":{\"completed_rows\":");w.number(completed_rows);w.text(",\"attempted_rows\":");w.number(count_);w.text(",\"visited\":");w.number(visited);w.text(",\"compared\":");w.number(compared);w.text(",\"finite_cpu\":");w.number(fc);w.text(",\"finite_gpu\":");w.number(fg);w.text(",\"violations\":");w.number(violations);w.text(",\"maxabs\":");if(compared)w.real(ma);else w.text("null");w.text(",\"rms\":");if(compared&&std::isfinite(sum))w.real(std::sqrt(sum/double(compared)));else w.text("null");w.text("},\"first_divergence_row\":");if(first.valid)w.number(firstrow);else w.text("null");w.text(",\"first_divergence\":");witness(w,first);w.text("}");return w.result();
 }
};
static_assert(sizeof(Proof)<proof_cap,"all reference/comparator storage must fit separate4MiB");
}
