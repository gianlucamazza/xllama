#pragma once
#include "dev_run_config.h"
#include <array>
#include <functional>
namespace xllama::gpu_identity {
inline constexpr char contract_sha[]="155a1a305cddb3e925cf2b52fd1f5bc9b8132b20c85a8fca325e2ceadd768f0a";
inline constexpr size_t request_cap=4096,result_cap=16384,serialization_capacity_cap=65536;
inline constexpr uint64_t reserve=512ull<<20,probe=256ull<<20;
struct Request {std::string raw,id,selection;uint64_t expires=0;size_t parser_owned=0;
 size_t owned_bytes()const{return sizeof(*this)+raw.capacity()+id.capacity()+selection.capacity()+parser_owned;}
 std::string leaf(const char*s)const{dev_run::need(dev_run::run_id(id),"identity run id");return "ua68-gpu-identity-v1."+id+"."+s;}
};
inline Request parse(std::string raw,uint64_t now){auto j=dev_run::Parser(raw).parse(request_cap);j.keys({"schema","contract_sha256","run_id","selection_id","expires_at_epoch_ms"});j.at("schema").eq("ua68-gpu-identity-request-v1");j.at("contract_sha256").eq(contract_sha);Request r;r.parser_owned=j.owned();r.raw=std::move(raw);
 dev_run::need(j.at("run_id").type==dev_run::Json::Type::string&&dev_run::run_id(j.at("run_id").text),"identity run id");r.id=j.at("run_id").text;
 dev_run::need(j.at("selection_id").type==dev_run::Json::Type::string,"identity selection type");r.selection=j.at("selection_id").text;dev_run::need(!r.selection.empty()&&r.selection.size()<=64&&std::all_of(r.selection.begin(),r.selection.end(),[](char c){return(c>='a'&&c<='z')||(c>='0'&&c<='9')||c=='-';}),"identity selection characters");r.expires=j.at("expires_at_epoch_ms").range(1,INT64_MAX);dev_run::need(r.expires>now&&r.expires-now<=300000,"identity strict expiry");return r;}
struct Quota {uint64_t limit=0,usage=0;uint32_t level=0;}; // level0Low1Medium2High3OverLimit; other refuses
struct Observation {std::string luid,shader;uint32_t vendor=0,device=0,subsystem=0,revision=0,flags=0,nodes=0;bool uma=false,coherent=false;uint64_t budget=0,usage=0,reservation=0,available_reservation=0,observed_init_delta=0;Quota before{},after_device{},after_close{};
 size_t owned_bytes()const{return sizeof(*this)+luid.capacity()+shader.capacity();}
};
// Query returns only after temporary COM/hash owners have actually released.
// callback checks stop/original deadline/native quota; absence refuses.
Observation query(const std::function<void()>& check,uint64_t metadata_charge);
}
