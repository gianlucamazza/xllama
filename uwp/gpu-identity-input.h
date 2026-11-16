#pragma once
#include "dev-run-input.h"
#include "xllama/gpu_identity.h"
namespace xllama::bridge {
inline std::string identity_request_sha(const std::string&raw){
 BCRYPT_ALG_HANDLE a=nullptr;BCRYPT_HASH_HANDLE h=nullptr;std::vector<unsigned char> object;unsigned char sum[32]{};
 auto check=[](NTSTATUS s){if(s<0)throw std::runtime_error("identity request SHA/close failed");};
 try{check(BCryptOpenAlgorithmProvider(&a,BCRYPT_SHA256_ALGORITHM,nullptr,0));DWORD size=0,got=0;check(BCryptGetProperty(a,BCRYPT_OBJECT_LENGTH,reinterpret_cast<PUCHAR>(&size),sizeof(size),&got,0));if(got!=sizeof(size)||!size||size>65536)throw std::runtime_error("identity SHA object bound");object.resize(size);check(BCryptCreateHash(a,&h,object.data(),size,nullptr,0,0));check(BCryptHashData(h,reinterpret_cast<PUCHAR>(const_cast<char*>(raw.data())),ULONG(raw.size()),0));check(BCryptFinishHash(h,sum,sizeof(sum),0));auto dh=BCryptDestroyHash(h);h=nullptr;check(dh);auto da=BCryptCloseAlgorithmProvider(a,0);a=nullptr;check(da);
 }catch(...){if(h)BCryptDestroyHash(h);if(a)BCryptCloseAlgorithmProvider(a,0);throw;}
 std::string out;out.reserve(64);const char*hex="0123456789abcdef";for(auto b:sum){out+=hex[b>>4];out+=hex[b&15];}return out;
}
struct GpuIdentityInput {
 DevHandle held;gpu_identity::Request request;std::string request_sha;std::filesystem::path root;dev_run::OneUse use;
 explicit GpuIdentityInput(std::filesystem::path r):root(std::move(r)){
  open_dev_file(held,root/L"ua68-gpu-identity-v1.ticket.json",true);auto raw=read_dev_file(held,gpu_identity::request_cap);request_sha=identity_request_sha(raw);request=gpu_identity::parse(std::move(raw),dev_utc_ms());
 }
 std::filesystem::path path(const char*s)const{auto x=request.leaf(s);return root/std::wstring(x.begin(),x.end());}
 size_t owned_bytes()const{return sizeof(*this)+request.owned_bytes()+request_sha.capacity()+root.native().capacity()*sizeof(wchar_t)+4096;}
 void spend(){dev_run::need(request.expires>dev_utc_ms(),"identity expired before claim");use.spend([&]{DevHandle c(CreateFile2(path("claim").c_str(),GENERIC_WRITE,0,CREATE_NEW,nullptr));if(c.value==INVALID_HANDLE_VALUE)throw std::runtime_error("identity claim unavailable/already spent");const auto b=request_sha+"\n";DWORD n=0;if(!WriteFile(c.value,b.data(),DWORD(b.size()),&n,nullptr)||n!=b.size()||!FlushFileBuffers(c.value))throw std::runtime_error("identity claim durable write");c.close();},[&]{dev_run::need(request.expires>dev_utc_ms(),"identity expired before consume");FILE_DISPOSITION_INFO d{};d.DeleteFile=TRUE;if(!SetFileInformationByHandle(held.value,FileDispositionInfo,&d,sizeof(d)))throw std::runtime_error("identity held consumption failed");held.close();});dev_run::need(use.may_start_model(),"identity ticket not consumed");}
};
}
