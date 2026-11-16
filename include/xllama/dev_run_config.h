#pragma once
#include "first_answer_record.h"
#include <vector>
#include <utility>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <algorithm>
namespace xllama::dev_run {
inline constexpr char engine_digest[]="6baa8156f7f07e9974e3ee32280d2367811026df427fcad13074c359a992b366";
inline constexpr char capabilities[]=R"CAP({"cache_owned_ceiling_max":1342177280,"cache_slots_max":512,"cache_slots_min":2,"collector_charge_bytes":2097152,"config_max_bytes":16384,"config_schema":"xllama-dev-run-config-v1","context":512,"contract":"xllama-cpu-dev-engine-contract-v1","json_max_depth":8,"json_max_nodes":256,"main_cap_bytes":262144,"model_bytes":14526258464,"model_id":"lfm25-24b-a2b-q4-k-m","model_sha256":"d3334405b96228eb9908cdebb5c6194908f2da64282bda852afc3a3910c64e7d","native_wall_seconds":1200,"new_tokens_max":384,"output_schema":"xllama-dev-run-output-v1","probe_bytes":268435456,"prompt_tokens_max":128,"prompt_utf8_bytes_max":4096,"reader":"windows_buffered_overlapped/exact_tensor_extent/normal/one_inflight/no_prefetch","reserve_bytes":536870912,"scenario":"cpu_single_turn/plain_greedy_trace_v1","stage_ceiling_bytes":4194304,"storage_observability":"phase_decode_aggregate/slow_samples_disabled","template_sha256":"ea663864491de7ade391839479860ca95541f892f72665c73251fbd4643b1bef","threads":1,"ticket_future_ms_max":1800000,"ticket_max_bytes":4096,"ticket_schema":"xllama-dev-run-ticket-v1","trace_capacity_bytes":524288,"trace_file_bytes":500032,"trace_schema":"UA50TRC2","unsupported":["gpu","fused","threads_not_1","nongreedy","history","tools","lifecycle_sequence","ui","context_not_512","prefetch","alternate_io_hints","slow_read_samples","plugins"],"watchdog_seconds":180}
)CAP";
inline constexpr size_t config_cap=16384,ticket_cap=4096;
inline void need(bool v,const char* cause){if(!v)throw std::invalid_argument(cause);}
inline bool hex(const std::string&s,size_t n){return s.size()==n&&std::all_of(s.begin(),s.end(),[](char x){return (x>='0'&&x<='9')||(x>='a'&&x<='f');});}
inline bool run_id(const std::string&s){return hex(s,32)&&s!=std::string(32,'0');}
// Small closed-schema parser. No lenient escape repair, numeric coercion or map overwrite.
struct Json {
 enum class Type{object,array,string,integer,boolean,null};Type type=Type::null;
 std::string text;int64_t number=0;bool flag=false;
 std::vector<std::pair<std::string,Json>> object;std::vector<Json> array;
 size_t owned()const{size_t n=sizeof(Json)+text.capacity()+object.capacity()*sizeof(std::pair<std::string,Json>)+array.capacity()*sizeof(Json);for(const auto&p:object)n+=p.first.capacity()+p.second.owned();for(const auto&p:array)n+=p.owned();return n;}
 const Json& at(const char*k)const{need(type==Type::object,"object required");for(const auto&p:object)if(p.first==k)return p.second;throw std::invalid_argument("missing field");}
 void keys(std::initializer_list<const char*> names)const{need(type==Type::object&&object.size()==names.size(),"unknown/missing field");for(auto name:names)(void)at(name);}
 void eq(const char*s)const{need(type==Type::string&&text==s,"unsupported string capability");}
 void eq(int64_t n)const{need(type==Type::integer&&number==n,"unsupported integer capability");}
 void eq(bool b)const{need(type==Type::boolean&&flag==b,"unsupported boolean capability");}
 uint64_t range(uint64_t lo,uint64_t hi)const{need(type==Type::integer&&number>=0&&uint64_t(number)>=lo&&uint64_t(number)<=hi,"integer bound/type");return uint64_t(number);}
};
class Parser {
 const std::string&s;size_t p=0,nodes=0;
 void ws(){while(p<s.size()&&(s[p]==' '||s[p]=='\n'||s[p]=='\r'||s[p]=='\t'))++p;}
 char take(){need(p<s.size(),"truncated JSON");return s[p++];}
 void expect(char c){need(take()==c,"JSON syntax");}
 unsigned hex4(){unsigned v=0;for(int i=0;i<4;++i){char c=take();unsigned d=c>='0'&&c<='9'?c-'0':c>='a'&&c<='f'?c-'a'+10:c>='A'&&c<='F'?c-'A'+10:99;need(d<16,"unicode escape");v=v*16+d;}return v;}
 void codepoint(std::string&out,unsigned cp){if(cp<128)out+=char(cp);else if(cp<2048){out+=char(0xc0|(cp>>6));out+=char(0x80|(cp&63));}else if(cp<65536){out+=char(0xe0|(cp>>12));out+=char(0x80|((cp>>6)&63));out+=char(0x80|(cp&63));}else{out+=char(0xf0|(cp>>18));out+=char(0x80|((cp>>12)&63));out+=char(0x80|((cp>>6)&63));out+=char(0x80|(cp&63));}}
 std::string string(){expect('"');std::string o;for(;;){char c=take();if(c=='"')break;need(static_cast<unsigned char>(c)>=32,"raw string control");if(c!='\\'){o+=c;continue;}char e=take();switch(e){case'"':case'\\':case'/':o+=e;break;case'b':o+='\b';break;case'f':o+='\f';break;case'n':o+='\n';break;case'r':o+='\r';break;case't':o+='\t';break;case'u':{unsigned v=hex4();if(v>=0xd800&&v<=0xdbff){expect('\\');expect('u');unsigned low=hex4();need(low>=0xdc00&&low<=0xdfff,"lone high surrogate");v=65536+((v-0xd800)<<10)+(low-0xdc00);}else need(v<0xdc00||v>0xdfff,"lone low surrogate");codepoint(o,v);break;}default:throw std::invalid_argument("unknown JSON escape");}}need(first_answer::utf8(o),"decoded UTF8");return o;}
 Json value(size_t depth){need(depth<=8&&++nodes<=256,"JSON depth/nodes");ws();Json v;need(p<s.size(),"missing value");char ch=s[p];
  if(ch=='{'){++p;v.type=Json::Type::object;ws();if(p<s.size()&&s[p]=='}'){++p;return v;}for(;;){ws();std::string key=string();for(const auto&q:v.object)need(q.first!=key,"duplicate JSON key");ws();expect(':');auto child=value(depth+1);v.object.emplace_back(std::move(key),std::move(child));ws();char c=take();if(c=='}')break;need(c==',',"object separator");}return v;}
  if(ch=='['){++p;v.type=Json::Type::array;ws();if(p<s.size()&&s[p]==']'){++p;return v;}for(;;){v.array.push_back(value(depth+1));ws();char c=take();if(c==']')break;need(c==',',"array separator");}return v;}
  if(ch=='"'){v.type=Json::Type::string;v.text=string();return v;}
  if(ch=='t'||ch=='f'||ch=='n'){const char*lit=ch=='t'?"true":ch=='f'?"false":"null";while(*lit)expect(*lit++);v.type=ch=='n'?Json::Type::null:Json::Type::boolean;v.flag=ch=='t';return v;}
  v.type=Json::Type::integer;bool negative=ch=='-';if(negative)++p;need(p<s.size()&&s[p]>='0'&&s[p]<='9',"integer required; float/nonfinite unsupported");bool zero=s[p]=='0';uint64_t n=0;size_t digits=0;while(p<s.size()&&s[p]>='0'&&s[p]<='9'){unsigned d=s[p++]-'0';need(n<=(uint64_t(INT64_MAX)-d)/10,"integer overflow");n=n*10+d;++digits;}need(!zero||digits==1,"leading integer zero");need(!(negative&&n==0),"negative zero");v.number=negative?-int64_t(n):int64_t(n);return v;
 }
public:
 explicit Parser(const std::string& raw):s(raw){}
 Json parse(size_t cap){need(!s.empty()&&s.size()<=cap&&first_answer::utf8(s),"raw JSON size/UTF8");auto v=value(1);ws();need(p==s.size(),"trailing JSON/numeric suffix");return v;}
};
struct Config {
 std::string raw,id,prompt;uint32_t prompt_cap=128,max_new=384,slots=2;uint64_t cache_ceiling=16777216;size_t parser_charge=0;
 size_t owned_bytes()const{return sizeof(*this)+raw.capacity()+id.capacity()+prompt.capacity()+parser_charge;}
 std::string leaf(const char*suffix)const{need(run_id(id),"unsafe run identity");return "ua50-dev-run-v1."+id+"."+suffix;}
};
inline Config parse_config(std::string raw){
 auto j=Parser(raw).parse(config_cap);j.keys({"schema","engine_contract_sha256","run_id","scenario","model","input","context","generation","execution","cache","storage","diagnostics"});
 j.at("schema").eq("xllama-dev-run-config-v1");j.at("engine_contract_sha256").eq(engine_digest);Config c;c.raw=std::move(raw);c.parser_charge=j.owned();need(j.at("run_id").type==Json::Type::string&&run_id(j.at("run_id").text),"run_id");c.id=j.at("run_id").text;
 const auto&scenario=j.at("scenario");scenario.keys({"kind","recipe"});scenario.at("kind").eq("cpu_single_turn");scenario.at("recipe").eq("plain_greedy_trace_v1");
 const auto&m=j.at("model");m.keys({"id","content_sha256"});m.at("id").eq("lfm25-24b-a2b-q4-k-m");m.at("content_sha256").eq("d3334405b96228eb9908cdebb5c6194908f2da64282bda852afc3a3910c64e7d");
 const auto&input=j.at("input");input.keys({"messages"});const auto&messages=input.at("messages");need(messages.type==Json::Type::array&&messages.array.size()==1,"single user message");const auto&msg=messages.array[0];msg.keys({"role","content"});msg.at("role").eq("user");need(msg.at("content").type==Json::Type::string,"prompt string");c.prompt=msg.at("content").text;need(!c.prompt.empty()&&c.prompt.size()<=4096&&c.prompt.find("<|im_start|>")==std::string::npos&&c.prompt.find("<|im_end|>")==std::string::npos,"prompt boundary");for(unsigned char x:c.prompt)need(x>=32||x=='\n'||x=='\t',"prompt control");
 const auto&ctx=j.at("context");ctx.keys({"requested_tokens","max_prompt_tokens"});ctx.at("requested_tokens").eq(int64_t(512));c.prompt_cap=uint32_t(ctx.at("max_prompt_tokens").range(1,128));
 const auto&g=j.at("generation");g.keys({"max_new_tokens","sampler"});c.max_new=uint32_t(g.at("max_new_tokens").range(1,384));g.at("sampler").keys({"kind"});g.at("sampler").at("kind").eq("greedy");
 const auto&e=j.at("execution");e.keys({"backend_id","device_mode","adapter_id","cpu","gpu","experts","compute_mode"});e.at("backend_id").eq("cpu_expert_streaming");e.at("device_mode").eq("cpu");need(e.at("adapter_id").type==Json::Type::null,"no adapter");e.at("cpu").keys({"threads","threads_batch"});e.at("cpu").at("threads").eq(int64_t(1));e.at("cpu").at("threads_batch").eq(int64_t(1));e.at("gpu").keys({"main_gpu","gpu_layers","offload_kqv","op_offload"});e.at("gpu").at("main_gpu").eq(int64_t(-1));e.at("gpu").at("gpu_layers").eq(int64_t(0));e.at("gpu").at("offload_kqv").eq(false);e.at("gpu").at("op_offload").eq(false);e.at("experts").keys({"placement"});e.at("experts").at("placement").eq("cpu_streamed");e.at("compute_mode").eq("nonfused");
 const auto&cache=j.at("cache");cache.keys({"policy","slots","owned_ceiling_bytes"});cache.at("policy").eq("global_lru");c.slots=uint32_t(cache.at("slots").range(2,512));c.cache_ceiling=cache.at("owned_ceiling_bytes").range(1,1342177280);
 const auto&storage=j.at("storage");storage.keys({"reader","observability"});const auto&rd=storage.at("reader");rd.keys({"mode","inflight_limit","request_policy","access_hint","prefetch"});rd.at("mode").eq("windows_buffered_overlapped");rd.at("inflight_limit").eq(int64_t(1));rd.at("request_policy").eq("exact_tensor_extent");rd.at("access_hint").eq("normal");rd.at("prefetch").eq("none");const auto&o=storage.at("observability");o.keys({"mode","slow_read_samples"});o.at("mode").eq("phase_decode_aggregate");o.at("slow_read_samples").keys({"enabled","max_events"});o.at("slow_read_samples").at("enabled").eq(false);o.at("slow_read_samples").at("max_events").eq(int64_t(0));j.at("diagnostics").keys({"trace"});j.at("diagnostics").at("trace").eq("ordered_expert_keys");return c;
}
struct Ticket {std::string raw,selection,id,config_sha;uint64_t expires=0;};
inline Ticket parse_ticket(std::string raw,uint64_t now){auto j=Parser(raw).parse(ticket_cap);j.keys({"schema","selection_id","run_id","config_sha256","engine_contract_sha256","expires_at_epoch_ms"});j.at("schema").eq("xllama-dev-run-ticket-v1");j.at("engine_contract_sha256").eq(engine_digest);Ticket t;t.raw=std::move(raw);need(j.at("run_id").type==Json::Type::string&&run_id(j.at("run_id").text),"ticket run_id");t.id=j.at("run_id").text;need(j.at("config_sha256").type==Json::Type::string&&hex(j.at("config_sha256").text,64),"ticket configSHA");t.config_sha=j.at("config_sha256").text;need(j.at("selection_id").type==Json::Type::string,"ticket selection");t.selection=j.at("selection_id").text;need(!t.selection.empty()&&t.selection.size()<=64&&std::all_of(t.selection.begin(),t.selection.end(),[](char x){return (x>='a'&&x<='z')||(x>='0'&&x<='9')||x=='-';}),"selection characters");t.expires=j.at("expires_at_epoch_ms").range(1,INT64_MAX);need(t.expires>now&&t.expires-now<=1800000,"expired/future ticket");return t;}
// Actual Windows calls are supplied by the source-owned held-handle adapter.
// No pathname cleanup/retry can be added by this state machine after consumption.
struct OneUse {
 enum class State{validated,claim_attempted,claimed,consume_attempted,consumed};State state=State::validated;
 template<class CreateClaim,class ConsumeHeld>void spend(CreateClaim claim,ConsumeHeld consume){need(state==State::validated,"ticket already spent");state=State::claim_attempted;claim();state=State::claimed;state=State::consume_attempted;consume();state=State::consumed;}
 bool may_start_model()const{return state==State::consumed;}
};
}
