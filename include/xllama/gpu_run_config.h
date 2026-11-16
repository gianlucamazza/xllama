#pragma once
#include "dev_run_config.h"
#include "gpu_numeric_proof.h"
#include "first_answer_template.h"
namespace xllama::gpu_run {
inline constexpr char engine_sha[]="2b904dcee7f3dfc51f05724a1e493b6fdc22b333410608ad8321e170a8b98bd4";
inline constexpr char capabilities[]=R"UA68({"CPU_threads":1,"GPU_timestamps":"not_collected_first_slice","activation_tag":"gpu-expert-proof-v1","batch":1,"collector_charge":2097152,"config_max_bytes":16384,"config_schema":"ua68-gpu-proof-config-v1","context":512,"contract":"ua68-expert-sync-contract-v1","driver_probe":268435456,"experts":64,"gpu_outstanding":1,"gpu_slots":1,"host_cache_slots":2,"main_cap":262144,"model_pairs":256000,"model_rows":2,"model_sha256":"d3334405b96228eb9908cdebb5c6194908f2da64282bda852afc3a3910c64e7d","native_wall_seconds":1200,"no_fallback":true,"output_schema":"ua68-gpu-proof-output-v1","primitive_pairs":28672,"primitive_rows":16,"proof_owned_cap":4194304,"quants":["Q4_K","Q6_K"],"raw_rows_saved":false,"reserve":536870912,"roles":114,"run_prefix":"ua68-gpu-proof-v1.","stages":["primitive16","model2"],"summary_cap":65536,"threshold":{"absolute":0.001,"relative_to_cpu":0.001},"ticket_leaf":"ua68-gpu-proof-v1.ticket.json","ticket_max_bytes":4096,"ticket_schema":"ua68-gpu-proof-ticket-v1","trace_cap":524288,"trace_file_bytes":254432,"trace_schema":"UA68TRC1","tuple":"cpu_graph_d3d12_expert_sync","unsupported":["fused","GPU_cache","async","prefetch","threads_not_1","alternate_model","history","UI","context_not_512","performance_sweep"],"watchdog_seconds":180}
)UA68";
struct Config {
 std::string raw,id,adapter_luid,shader_sha,primitive_acceptance_sha;dev_run::Config cpu;size_t parser_charge=0;
 gpu_proof::Stage stage=gpu_proof::Stage::Primitive16;
 size_t owned_bytes()const{return sizeof(*this)+raw.capacity()+id.capacity()+adapter_luid.capacity()+shader_sha.capacity()+primitive_acceptance_sha.capacity()+cpu.owned_bytes()+parser_charge;}
 std::string leaf(const char* suffix)const{dev_run::need(dev_run::run_id(id),"GPU run id");return "ua68-gpu-proof-v1."+id+"."+suffix;}
};
inline Config parse(std::string raw){
 auto j=dev_run::Parser(raw).parse(16384);j.keys({"schema","engine_contract_sha256","run_id","stage","base_cpu_json","execution","prerequisite_primitive_acceptance_sha256"});j.at("schema").eq("ua68-gpu-proof-config-v1");j.at("engine_contract_sha256").eq(engine_sha);
 Config c;c.raw=std::move(raw);c.parser_charge=j.owned();dev_run::need(j.at("run_id").type==dev_run::Json::Type::string&&dev_run::run_id(j.at("run_id").text),"GPU run id");c.id=j.at("run_id").text;
 const auto& stage=j.at("stage");dev_run::need(stage.type==dev_run::Json::Type::string,"GPU stage");if(stage.text=="primitive16")c.stage=gpu_proof::Stage::Primitive16;else if(stage.text=="model2")c.stage=gpu_proof::Stage::Model2;else throw std::invalid_argument("GPU stage unsupported");
 auto&base=j.at("base_cpu_json");dev_run::need(base.type==dev_run::Json::Type::string,"explicit immutable CPU envelope required");c.cpu=dev_run::parse_config(base.text);
 dev_run::need(c.cpu.id==c.id&&c.cpu.slots==2&&c.cpu.cache_ceiling==(16u<<20)&&c.cpu.prompt_cap==128&&c.cpu.max_new==2&&c.cpu.prompt==first_answer::greeting,"GPU exact fixed reference profile");
 const auto&e=j.at("execution");e.keys({"kind","adapter_luid","compiled_shader_sha256"});e.at("kind").eq("cpu_graph_d3d12_expert_sync");
 dev_run::need(e.at("adapter_luid").type==dev_run::Json::Type::string&&e.at("compiled_shader_sha256").type==dev_run::Json::Type::string,"GPU identity types");c.adapter_luid=e.at("adapter_luid").text;c.shader_sha=e.at("compiled_shader_sha256").text;
 dev_run::need(c.adapter_luid.size()==17&&c.adapter_luid[8]==':'&&dev_run::hex(c.adapter_luid.substr(0,8),8)&&dev_run::hex(c.adapter_luid.substr(9),8)&&dev_run::hex(c.shader_sha,64),"GPU exact identity");
 auto&pr=j.at("prerequisite_primitive_acceptance_sha256");if(c.stage==gpu_proof::Stage::Model2){dev_run::need(pr.type==dev_run::Json::Type::string&&dev_run::hex(pr.text,64),"model2 externally accepted primitive prerequisite");c.primitive_acceptance_sha=pr.text;}else dev_run::need(pr.type==dev_run::Json::Type::null,"primitive has no prerequisite override");
 return c;
}
struct Ticket {std::string raw,selection,id,config_sha;uint64_t expires=0;size_t parser_charge=0;};
inline Ticket ticket(std::string raw,uint64_t now){auto j=dev_run::Parser(raw).parse(4096);j.keys({"schema","selection_id","run_id","config_sha256","engine_contract_sha256","expires_at_epoch_ms"});j.at("schema").eq("ua68-gpu-proof-ticket-v1");j.at("engine_contract_sha256").eq(engine_sha);Ticket t;t.raw=std::move(raw);t.parser_charge=j.owned();
 dev_run::need(j.at("run_id").type==dev_run::Json::Type::string&&dev_run::run_id(j.at("run_id").text),"GPU ticket id");t.id=j.at("run_id").text;dev_run::need(j.at("config_sha256").type==dev_run::Json::Type::string&&dev_run::hex(j.at("config_sha256").text,64),"GPU ticket hash");t.config_sha=j.at("config_sha256").text;
 dev_run::need(j.at("selection_id").type==dev_run::Json::Type::string,"GPU ticket selection");t.selection=j.at("selection_id").text;dev_run::need(!t.selection.empty()&&t.selection.size()<=64&&std::all_of(t.selection.begin(),t.selection.end(),[](char x){return (x>='a'&&x<='z')||(x>='0'&&x<='9')||x=='-';}),"GPU selection chars");t.expires=j.at("expires_at_epoch_ms").range(1,INT64_MAX);dev_run::need(t.expires>now&&t.expires-now<=1800000,"GPU original ticket expiry");return t;}
}
