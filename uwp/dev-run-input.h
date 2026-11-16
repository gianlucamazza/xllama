#pragma once
#include "xllama/dev_run_config.h"
#include <windows.h>
#include <bcrypt.h>
#include <array>
#include <filesystem>
#include <cstring>
namespace xllama::bridge {
struct DevHandle {
 HANDLE value=INVALID_HANDLE_VALUE;
 DevHandle()=default;explicit DevHandle(HANDLE v):value(v){}
 DevHandle(const DevHandle&)=delete;DevHandle&operator=(const DevHandle&)=delete;
 ~DevHandle(){if(value!=INVALID_HANDLE_VALUE)CloseHandle(value);}
 void close(){HANDLE v=value;value=INVALID_HANDLE_VALUE;if(v!=INVALID_HANDLE_VALUE&&!CloseHandle(v))throw std::runtime_error("native handle close failure");}
};
inline uint64_t dev_utc_ms(){FILETIME f{};GetSystemTimeAsFileTime(&f);ULARGE_INTEGER v{};v.LowPart=f.dwLowDateTime;v.HighPart=f.dwHighDateTime;return (v.QuadPart-116444736000000000ull)/10000;}
inline std::string dev_sha(const std::string& bytes){
 BCRYPT_ALG_HANDLE alg=nullptr;BCRYPT_HASH_HANDLE hash=nullptr;std::vector<unsigned char> object;std::array<unsigned char,32> sum{};
 auto check=[](NTSTATUS s){if(s<0)throw std::runtime_error("config BCrypt SHA");};
 try{check(BCryptOpenAlgorithmProvider(&alg,BCRYPT_SHA256_ALGORITHM,nullptr,0));DWORD size=0,got=0;check(BCryptGetProperty(alg,BCRYPT_OBJECT_LENGTH,reinterpret_cast<PUCHAR>(&size),sizeof(size),&got,0));if(got!=sizeof(size)||!size||size>65536)throw std::runtime_error("config SHA object bound");object.resize(size);check(BCryptCreateHash(alg,&hash,object.data(),size,nullptr,0,0));check(BCryptHashData(hash,reinterpret_cast<PUCHAR>(const_cast<char*>(bytes.data())),ULONG(bytes.size()),0));check(BCryptFinishHash(hash,sum.data(),ULONG(sum.size()),0));BCryptDestroyHash(hash);hash=nullptr;BCryptCloseAlgorithmProvider(alg,0);alg=nullptr;
 }catch(...){if(hash)BCryptDestroyHash(hash);if(alg)BCryptCloseAlgorithmProvider(alg,0);throw;}
 const char*d="0123456789abcdef";std::string out;out.reserve(64);for(auto x:sum){out+=d[x>>4];out+=d[x&15];}return out;
}
inline void open_dev_file(DevHandle& h,const std::filesystem::path& path,bool ticket){CREATEFILE2_EXTENDED_PARAMETERS p{};p.dwSize=sizeof(p);p.dwFileFlags=FILE_FLAG_OPEN_REPARSE_POINT;h.value=CreateFile2(path.c_str(),GENERIC_READ|(ticket?DELETE:0),ticket?0:FILE_SHARE_READ,OPEN_EXISTING,&p);if(h.value==INVALID_HANDLE_VALUE)throw std::runtime_error("config/ticket held open refused");}
inline std::string read_dev_file(DevHandle& h,size_t cap){
 FILE_STANDARD_INFO before{};FILE_ATTRIBUTE_TAG_INFO attr{};FILE_ID_INFO id{},postid{};
 if(GetFileType(h.value)!=FILE_TYPE_DISK||!GetFileInformationByHandleEx(h.value,FileStandardInfo,&before,sizeof(before))||!GetFileInformationByHandleEx(h.value,FileAttributeTagInfo,&attr,sizeof(attr))||!GetFileInformationByHandleEx(h.value,FileIdInfo,&id,sizeof(id))||before.NumberOfLinks!=1||before.Directory||before.DeletePending||(attr.FileAttributes&FILE_ATTRIBUTE_REPARSE_POINT)||before.EndOfFile.QuadPart<=0||uint64_t(before.EndOfFile.QuadPart)>cap)throw std::runtime_error("config/ticket file shape/bound");
 std::string raw(size_t(before.EndOfFile.QuadPart),'\0');DWORD got=0;if(!ReadFile(h.value,raw.data(),DWORD(raw.size()),&got,nullptr)||got!=raw.size())throw std::runtime_error("config/ticket short read");unsigned char extra=0;DWORD trailing=0;if(!ReadFile(h.value,&extra,1,&trailing,nullptr)||trailing)throw std::runtime_error("config/ticket trailing read");FILE_STANDARD_INFO after{};if(!GetFileInformationByHandleEx(h.value,FileStandardInfo,&after,sizeof(after))||!GetFileInformationByHandleEx(h.value,FileIdInfo,&postid,sizeof(postid))||after.EndOfFile.QuadPart!=before.EndOfFile.QuadPart||memcmp(&id,&postid,sizeof(id)))throw std::runtime_error("config/ticket held identity changed");return raw;
}
struct DevRunInput {
 DevHandle held_ticket;dev_run::Ticket ticket;dev_run::Config config;dev_run::OneUse use;std::string config_sha,ticket_sha;std::filesystem::path root;
 explicit DevRunInput(std::filesystem::path base):root(std::move(base)){
  open_dev_file(held_ticket,root/L"ua50-dev-run-v1.ticket.json",true);auto ticket_bytes=read_dev_file(held_ticket,dev_run::ticket_cap);ticket_sha=dev_sha(ticket_bytes);ticket=dev_run::parse_ticket(std::move(ticket_bytes),dev_utc_ms());
  DevHandle cfg;const std::string leaf="ua50-dev-run-v1."+ticket.id+".config.json";open_dev_file(cfg,root/std::wstring(leaf.begin(),leaf.end()),false);auto bytes=read_dev_file(cfg,dev_run::config_cap);config_sha=dev_sha(bytes);config=dev_run::parse_config(std::move(bytes));cfg.close();dev_run::need(ticket.id==config.id&&ticket.config_sha==config_sha,"ticket/config binding");
 }
 size_t owned_bytes()const{return sizeof(*this)+config.owned_bytes()+ticket.raw.capacity()+ticket.selection.capacity()+ticket.id.capacity()+ticket.config_sha.capacity()+config_sha.capacity()+ticket_sha.capacity()+4096;}
 std::filesystem::path path(const char*suffix)const{auto leaf=config.leaf(suffix);return root/std::wstring(leaf.begin(),leaf.end());}
 void spend(){dev_run::need(ticket.expires>dev_utc_ms(),"ticket expired before claim");use.spend([&]{
   DevHandle claim(CreateFile2(path("claim").c_str(),GENERIC_WRITE,0,CREATE_NEW,nullptr));if(claim.value==INVALID_HANDLE_VALUE)throw std::runtime_error("run identity already used/claim unavailable");
   const auto bytes=ticket_sha+"\n";DWORD n=0;if(!WriteFile(claim.value,bytes.data(),DWORD(bytes.size()),&n,nullptr)||n!=bytes.size()||!FlushFileBuffers(claim.value))throw std::runtime_error("claim durable write refused");claim.close();
  },[&]{dev_run::need(ticket.expires>dev_utc_ms(),"ticket expired before consumption");FILE_DISPOSITION_INFO disposition{};disposition.DeleteFile=TRUE;if(!SetFileInformationByHandle(held_ticket.value,FileDispositionInfo,&disposition,sizeof(disposition)))throw std::runtime_error("held ticket consumption failed");held_ticket.close();
  });dev_run::need(use.may_start_model(),"ticket not consumed");
 }
};
}
