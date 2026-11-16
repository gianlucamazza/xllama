#pragma once
#include "dev_run_config.h"
#include <mutex>
#include <atomic>
#include <memory>
#include <functional>
namespace xllama::generation {
inline constexpr char contract_sha[]="9a0861e803fbd0a302bae54f67c7694dae9834758ced824735634cbd84fbc335";
inline constexpr size_t request_cap=16384,permit_cap=4096,result_cap=262144;
struct Profile {bool hybrid=false;uint32_t workers=1,context=512,prompt_cap=128,max_new=128,slots=2;uint64_t ceiling=16u<<20;};
struct Request {std::string raw,id,prompt,permit_sha;Profile profile;size_t parsed_owned=0;
 size_t owned_bytes()const{return sizeof(*this)+raw.capacity()+id.capacity()+prompt.capacity()+permit_sha.capacity()+parsed_owned;}
 std::string leaf(const char*s)const{dev_run::need(dev_run::run_id(id),"generation run id");return "ua68-generation-v1."+id+"."+s;}
};
inline Request parse(std::string raw){auto j=dev_run::Parser(raw).parse(request_cap);j.keys({"schema","contract_sha256","run_id","permit_sha256","prompt","profile"});j.at("schema").eq("ua68-generation-request-v1");j.at("contract_sha256").eq(contract_sha);Request r;r.parsed_owned=j.owned();r.raw=std::move(raw);
 dev_run::need(j.at("run_id").type==dev_run::Json::Type::string&&dev_run::run_id(j.at("run_id").text),"generation run id");r.id=j.at("run_id").text;dev_run::need(j.at("permit_sha256").type==dev_run::Json::Type::string&&dev_run::hex(j.at("permit_sha256").text,64),"generation permit SHA");r.permit_sha=j.at("permit_sha256").text;
 dev_run::need(j.at("prompt").type==dev_run::Json::Type::string,"generation prompt type");r.prompt=j.at("prompt").text;dev_run::need(!r.prompt.empty()&&r.prompt.size()<=4096,"generation prompt bytes");for(auto x:{"<|im_start|>","<|im_end|>","<|im_sep|>","<|fim_prefix|>","<|fim_middle|>","<|fim_suffix|>"})dev_run::need(r.prompt.find(x)==std::string::npos,"generation role delimiter refused");
 auto&p=j.at("profile");p.keys({"compute","workers","context","prompt_cap","max_new","cache_slots","cache_owned_ceiling","sampler","output_max_bytes"});p.at("workers").eq(int64_t(1));p.at("sampler").eq("greedy");p.at("output_max_bytes").eq(int64_t(result_cap));dev_run::need(p.at("compute").type==dev_run::Json::Type::string,"generation compute type");if(p.at("compute").text=="cpu")r.profile.hybrid=false;else if(p.at("compute").text=="hybrid_selected_expert_sync")r.profile.hybrid=true;else throw std::invalid_argument("generation compute unsupported");
 r.profile.context=uint32_t(p.at("context").range(64,512));dev_run::need(r.profile.context==64||r.profile.context==128||r.profile.context==256||r.profile.context==512,"generation context unsupported");r.profile.prompt_cap=uint32_t(p.at("prompt_cap").range(1,128));r.profile.max_new=uint32_t(p.at("max_new").range(1,384));dev_run::need(r.profile.max_new<r.profile.context,"generation no room for prompt");r.profile.slots=uint32_t(p.at("cache_slots").range(2,512));r.profile.ceiling=p.at("cache_owned_ceiling").range(16u<<20,1280ull<<20);return r;}
struct Permit {std::string raw,package,artifact,primitive,model2,luid,shader;uint64_t expires=0;size_t parsed_owned=0;
 size_t owned_bytes()const{return sizeof(*this)+raw.capacity()+package.capacity()+artifact.capacity()+primitive.capacity()+model2.capacity()+luid.capacity()+shader.capacity()+parsed_owned;}
};
inline Permit permit(std::string raw,uint64_t now,const std::string&package){auto j=dev_run::Parser(raw).parse(permit_cap);j.keys({"schema","contract_sha256","native_package_full_name","signed_artifact_sha256","primitive_acceptance_sha256","model2_acceptance_sha256","adapter_luid","compiled_shader_sha256","expires_at_epoch_ms"});j.at("schema").eq("ua68-generation-permit-v1");j.at("contract_sha256").eq(contract_sha);Permit p;p.raw=std::move(raw);p.parsed_owned=j.owned();
 auto str=[&](const char*k){const auto&v=j.at(k);dev_run::need(v.type==dev_run::Json::Type::string,"generation permit field type");return v.text;};p.package=str("native_package_full_name");dev_run::need(!package.empty()&&package.size()<=1024&&p.package==package,"generation permit actual package mismatch");p.artifact=str("signed_artifact_sha256");p.primitive=str("primitive_acceptance_sha256");p.model2=str("model2_acceptance_sha256");p.shader=str("compiled_shader_sha256");p.luid=str("adapter_luid");for(const auto&s:{p.artifact,p.primitive,p.model2,p.shader})dev_run::need(dev_run::hex(s,64),"generation proof/artifact/identity prerequisite missing");dev_run::need(p.luid.size()==17&&p.luid[8]==':'&&dev_run::hex(p.luid.substr(0,8),8)&&dev_run::hex(p.luid.substr(9),8),"generation LUID missing");p.expires=j.at("expires_at_epoch_ms").range(1,INT64_MAX);dev_run::need(now<p.expires&&p.expires-now<=3600000,"generation permit expired/future");return p;
}
// One actual production active-run state shared by run/status/stop; no queue/reconfigure/reuse.
struct Active {const std::string id,config_sha;std::atomic<bool> stop_requested{false};std::mutex mutex;std::string phase="admission";std::function<void()> stop_native;bool publication_decided=false;
 Active(std::string i,std::string s):id(std::move(i)),config_sha(std::move(s)){}
 void set_phase(const char*s){std::lock_guard<std::mutex>l(mutex);phase=s;}
 void bind_stop(std::function<void()>f){bool stopped;{std::lock_guard<std::mutex>l(mutex);stop_native=std::move(f);stopped=stop_requested.load();}if(stopped)stop();}
 bool stop(){std::function<void()> f;{std::lock_guard<std::mutex>l(mutex);if(publication_decided)return false;stop_requested.store(true);f=stop_native;}if(f)f();return true;}
 // Linearizes final buffered publication against the manual stop endpoint AFTER terminal drain.
 bool decide_publication(){std::lock_guard<std::mutex>l(mutex);if(publication_decided)throw std::logic_error("generation publication repeated");publication_decided=true;phase="closed-publication";return !stop_requested.load();}
};
struct Registry {std::mutex mutex;std::shared_ptr<Active> active;
 std::shared_ptr<Active> start(const std::string&id,const std::string&sha){std::lock_guard<std::mutex>l(mutex);if(active)throw std::runtime_error("generation busy");active=std::make_shared<Active>(id,sha);return active;}
 std::shared_ptr<Active> get(const std::string&id){std::lock_guard<std::mutex>l(mutex);return active&&active->id==id?active:nullptr;}
 void closed(const std::shared_ptr<Active>&a){std::lock_guard<std::mutex>l(mutex);if(active!=a)throw std::logic_error("generation registry owner mismatch");active.reset();}
};
struct TextView {std::string classification;bool valid=false,closed=false;std::string reasoning,final;};
inline TextView classify(const std::string&raw,const std::string&stop,bool normal){TextView v;v.valid=first_answer::utf8(raw);if(!v.valid){v.classification="invalid_utf8";return v;}auto at=raw.find("</think>");v.closed=at!=std::string::npos;v.reasoning=v.closed?raw.substr(0,at):raw;v.final=v.closed?raw.substr(at+8):std::string();bool meaningful=std::any_of(v.final.begin(),v.final.end(),[](unsigned char c){return c!=' '&&c!='\t'&&c!='\r'&&c!='\n';});v.classification=!normal?"partial_error":!v.closed?"reasoning_only":!meaningful?"closed_thinking_no_final":stop=="eos_or_eot"?"terminated_final_text":"length_capped_final_text";return v;}
}
