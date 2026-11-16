#pragma once
#include "dev-run-input.h"
#include "xllama/gpu_run_config.h"
namespace xllama::bridge {
struct GpuProofInput {
 DevHandle held_ticket;gpu_run::Config config;gpu_run::Ticket ticket;dev_run::OneUse use;std::string config_sha,ticket_sha;std::filesystem::path root;
 explicit GpuProofInput(std::filesystem::path base):root(std::move(base)){
  open_dev_file(held_ticket,root/L"ua68-gpu-proof-v1.ticket.json",true);auto traw=read_dev_file(held_ticket,4096);ticket_sha=dev_sha(traw);ticket=gpu_run::ticket(std::move(traw),dev_utc_ms());
  DevHandle cfg;const auto leaf="ua68-gpu-proof-v1."+ticket.id+".config.json";open_dev_file(cfg,root/std::wstring(leaf.begin(),leaf.end()),false);auto raw=read_dev_file(cfg,16384);config_sha=dev_sha(raw);config=gpu_run::parse(std::move(raw));cfg.close();dev_run::need(config.id==ticket.id&&config_sha==ticket.config_sha,"GPU ticket consumed bytes");
 }
 std::filesystem::path path(const char*suffix)const{auto leaf=config.leaf(suffix);return root/std::wstring(leaf.begin(),leaf.end());}
 size_t owned_bytes()const{return sizeof(*this)+config.owned_bytes()+ticket.raw.capacity()+ticket.selection.capacity()+ticket.id.capacity()+ticket.config_sha.capacity()+ticket.parser_charge+config_sha.capacity()+ticket_sha.capacity()+4096;}
 void spend(){dev_run::need(ticket.expires>dev_utc_ms(),"GPU expiry before claim");use.spend([&]{DevHandle c(CreateFile2(path("claim").c_str(),GENERIC_WRITE,0,CREATE_NEW,nullptr));if(c.value==INVALID_HANDLE_VALUE)throw std::runtime_error("GPU claim already used/unavailable");auto s=ticket_sha+"\n";DWORD n=0;if(!WriteFile(c.value,s.data(),DWORD(s.size()),&n,nullptr)||n!=s.size()||!FlushFileBuffers(c.value))throw std::runtime_error("GPU claim durable write");c.close();},[&]{dev_run::need(ticket.expires>dev_utc_ms(),"GPU expiry before consume");FILE_DISPOSITION_INFO d{};d.DeleteFile=TRUE;if(!SetFileInformationByHandle(held_ticket.value,FileDispositionInfo,&d,sizeof(d)))throw std::runtime_error("GPU held ticket disposition failed");held_ticket.close();});dev_run::need(use.may_start_model(),"GPU ticket not consumed");}
};
}
