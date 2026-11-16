#if !defined(_WIN32)
#error "Windows BCrypt source candidate requires Windows/UWP validation"
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <bcrypt.h>
#include "xllama/cpu_expert_session.h"
#include <array>
#include <vector>
#include <algorithm>
#include <limits>
#include <utility>
#include <stdexcept>
namespace xllama {
namespace {
void checked(NTSTATUS status){if(status<0)throw std::runtime_error("BCrypt SHA256 failed");}
struct Algorithm{BCRYPT_ALG_HANDLE p=nullptr;~Algorithm(){if(p)BCryptCloseAlgorithmProvider(p,0);}};
// Hash handle dies before its backing object storage. No whole-file allocation.
struct Hash{std::vector<unsigned char> object;BCRYPT_HASH_HANDLE p=nullptr;~Hash(){if(p)BCryptDestroyHash(p);}};
struct Position{FILE*f;fpos_t pos;explicit Position(FILE*v):f(v){if(!f || fgetpos(f,&pos))throw std::runtime_error("held SHA position");}~Position(){fsetpos(f,&pos);}};
}
ua50::uwp::ValidateHeldFile make_held_sha256_validator(
 const ua50::uwp::ExpectedFile& expected,std::function<bool()> stop,ua50::uwp::Clock::time_point deadline,HashReadObserver observer){
 if(!expected.bytes || expected.bytes>uint64_t(INT64_MAX) || expected.sha256.size()!=64 || !stop || deadline<=ua50::uwp::Clock::now() || !std::all_of(expected.sha256.begin(),expected.sha256.end(),[](char c){return(c>='0'&&c<='9')||(c>='a'&&c<='f');}))throw std::invalid_argument("trusted exact bytes/SHA/deadline/stop required");
 return [expected,stop,deadline,observer](FILE*file,uint64_t admitted_bytes)->ua50::uwp::ValidationResult{
  if(admitted_bytes!=expected.bytes)throw std::invalid_argument("held SHA expected byte mismatch");
  auto gate=[&]{if(stop() || ua50::uwp::Clock::now()>=deadline)throw std::runtime_error("held SHA stopped/deadline");};
  gate();Position saved(file);if(_fseeki64(file,0,SEEK_SET))throw std::runtime_error("held SHA rewind");
  Algorithm algorithm;checked(BCryptOpenAlgorithmProvider(&algorithm.p,BCRYPT_SHA256_ALGORITHM,nullptr,0));
  DWORD object_bytes=0,hash_bytes=0,got=0;
  checked(BCryptGetProperty(algorithm.p,BCRYPT_OBJECT_LENGTH,reinterpret_cast<PUCHAR>(&object_bytes),sizeof(object_bytes),&got,0));
  if(got!=sizeof(object_bytes) || !object_bytes || object_bytes>65536)throw std::runtime_error("bounded BCrypt object refused");
  checked(BCryptGetProperty(algorithm.p,BCRYPT_HASH_LENGTH,reinterpret_cast<PUCHAR>(&hash_bytes),sizeof(hash_bytes),&got,0));
  if(got!=sizeof(hash_bytes) || hash_bytes!=32)throw std::runtime_error("SHA256 length refused");
  Hash hash;hash.object.resize(object_bytes);checked(BCryptCreateHash(algorithm.p,&hash.p,hash.object.data(),object_bytes,nullptr,0,0));
  std::array<unsigned char,65536> chunk{};uint64_t count=0;
  auto read=[&](unsigned char* destination,size_t want){const auto before=ua50::uwp::Clock::now();size_t n=fread(destination,1,want,file);const auto after=ua50::uwp::Clock::now();if(observer)observer(want,n,std::chrono::duration_cast<std::chrono::nanoseconds>(after-before).count());return n;};
  while(count<expected.bytes){
   gate();const size_t want=size_t(std::min<uint64_t>(chunk.size(),expected.bytes-count));
   const size_t n=read(chunk.data(),want);
   if(n!=want || ferror(file))throw std::runtime_error("held SHA short/error read");
   checked(BCryptHashData(hash.p,chunk.data(),ULONG(n),0));count+=n;
  }
  gate();unsigned char extra=0;if(read(&extra,1)!=0 || ferror(file))throw std::runtime_error("held SHA trailing bytes/error");
  std::array<unsigned char,32> sum{};checked(BCryptFinishHash(hash.p,sum.data(),ULONG(sum.size()),0));
  static constexpr char digits[]="0123456789abcdef";std::string hex;hex.reserve(64);
  for(auto v:sum){hex.push_back(digits[v>>4]);hex.push_back(digits[v&15]);}
  gate();if(count!=expected.bytes || hex!=expected.sha256)throw std::runtime_error("full held SHA mismatch");
  if(fsetpos(file,&saved.pos))throw std::runtime_error("held SHA restore");
  return {count,std::move(hex)};
 };
}
std::function<ua50::uwp::WindowsInputCandidate()> make_windows_expert_source(
 const std::wstring& path,const ua50::uwp::ExpectedFile& expected,
 std::shared_ptr<std::atomic<bool>> stop,ua50::uwp::Clock::time_point deadline,size_t stage_ceiling,HashReadObserver observer){
 if(path.empty() || !stop || !stage_ceiling || stage_ceiling>4u*1024u*1024u)throw std::invalid_argument("trusted local source/control/stage required");
 // This binds the implemented full-held SHA callback, not a caller's receipt claim.
 auto validate=make_held_sha256_validator(expected,[stop]{return stop->load(std::memory_order_acquire);},deadline,std::move(observer));
 return [path,expected,stop,deadline,stage_ceiling,validate]{
  if(stop->load(std::memory_order_acquire))throw std::runtime_error("source stopped before open");
  return ua50::uwp::open_windows_input_candidate(path,expected,validate,deadline,stage_ceiling);
 };
}

}
