#pragma once
#include "windows_held_reader.h"
#include <array>
#include <functional>
namespace ua50::uwp {
struct RunIo {
 using Row=std::array<uint64_t,15>;
 std::array<Row,8> phases{};std::array<Row,511> decodes{};
 std::array<uint64_t,8> phase_ns{};
 size_t phase=7,decode_count=0;int current=-1;uint64_t clock_errors=0;uint32_t first_native_error=0;
 bool closed=false,drain_seen=false;int64_t changed=-1,last=-1;
 std::function<int64_t()> clock=[] {return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();};
 std::weak_ptr<WindowsReader> observed;
 Ledger previous{};uint64_t expected_reads=0,expected_read_bytes=0;
 int64_t tick(){int64_t t=-1;try{t=clock();}catch(...){++clock_errors;throw;}if(t<0||t<last){++clock_errors;throw std::runtime_error("IO telemetry monotonic clock");}last=t;return t;}
 static uint64_t plus(uint64_t a,uint64_t b){if(a>UINT64_MAX-b)throw std::overflow_error("IO counter overflow");return a+b;}
 void add(size_t col,uint64_t n,bool frame=true){phases[phase][col]=plus(phases[phase][col],n);if(frame&&current>=0)decodes[size_t(current)][col]=plus(decodes[size_t(current)][col],n);}
 void set_phase(size_t p){if(p>=8)throw std::invalid_argument("IO phase");const auto t=tick();if(changed>=0)phase_ns[phase]=plus(phase_ns[phase],uint64_t(t-changed));phase=p;changed=t;}
 void begin_frame(bool prefill){if(current>=0||decode_count>=511)throw std::runtime_error("IO frame bound/state");set_phase(prefill?3:4);current=int(decode_count++);}
 void end_frame(){if(current<0)throw std::runtime_error("IO frame missing");current=-1;}
 void ledger(const Ledger& now){
  const uint64_t a[]={now.submissions,now.terminal_operations,now.terminal_bytes,now.polls,now.cancellations,now.commits};
  const uint64_t b[]={previous.submissions,previous.terminal_operations,previous.terminal_bytes,previous.polls,previous.cancellations,previous.commits};
  for(size_t i=0;i<6;++i){if(a[i]<b[i])throw std::runtime_error("IO ledger reset");add(4+i,a[i]-b[i]);}
  if(!first_native_error)first_native_error=now.first_native_error;previous=now;
 }
 void after_drain(uint64_t reads,uint64_t bytes){auto r=observed.lock();if(!r)throw std::runtime_error("Reader missing at actual drain");ledger(r->ledger_after_drain());expected_reads=reads;expected_read_bytes=bytes;current=-1;drain_seen=true;}
 void finish(){current=-1;const auto t=tick();if(changed>=0)phase_ns[phase]=plus(phase_ns[phase],uint64_t(t-changed));changed=t;closed=true;}
 void sha(size_t requested,size_t returned,int64_t elapsed){add(11,1,false);add(12,requested,false);add(13,returned,false);if(elapsed<0){++clock_errors;throw std::runtime_error("SHA read monotonic clock");}add(14,uint64_t(elapsed),false);}
 size_t owned_bytes()const{return sizeof(*this);}
 bool healthy(uint64_t sha_bytes=0)const{
  if(!closed||!drain_seen||clock_errors||first_native_error)return false;
  Row totals{},frames{};for(const auto&r:phases)for(size_t i=0;i<15;++i)totals[i]=plus(totals[i],r[i]);for(size_t n=0;n<decode_count;++n)for(size_t i=0;i<11;++i)frames[i]=plus(frames[i],decodes[n][i]);
  for(size_t i=0;i<11;++i)if(frames[i]!=plus(phases[3][i],phases[4][i]))return false;
  if(totals[0]!=expected_reads||totals[4]!=expected_reads||totals[5]!=expected_reads||totals[9]!=expected_reads||totals[1]!=expected_read_bytes||totals[2]!=expected_read_bytes||totals[6]!=expected_read_bytes||totals[8]||totals[10])return false;
  for(size_t i=0;i<11;++i)if(totals[i]!=frames[i])return false;
  if(sha_bytes){const auto calls=(sha_bytes+65535)/65536+1;for(size_t phase:{size_t(0),size_t(5)})if(phases[phase][11]!=calls||phases[phase][12]!=sha_bytes+1||phases[phase][13]!=sha_bytes)return false;}
  return true;
 }
 std::string json()const{
  auto row=[](const Row&r){std::string s="[";for(size_t i=0;i<r.size();++i){if(i)s+=',';s+=std::to_string(r[i]);}return s+"]";};
  Row total{};for(const auto&r:phases)for(size_t i=0;i<15;++i)total[i]=plus(total[i],r[i]);
  std::string s="{\"schema\":\"xllama-reader-aggregates-v1\",\"coverage\":{\"reader_calls\":\"complete_for_wrapped_provider_reader\",\"sha_calls\":\"complete_for_instrumented_held_SHA_fread_including_failures\",\"unobserved_model_file_io_phases\":[\"metadata_plan\",\"load\"],\"all_model_file_io_complete\":false,\"io_detail\":\"disabled_not_collected\",\"physical_ssd_metrics\":\"unavailable\",\"covered_reader_complete\":"+std::string(closed&&drain_seen?"true":"false")+",\"io_clock_errors\":"+std::to_string(clock_errors)+",\"first_native_error\":"+std::to_string(first_native_error)+"},\"phases\":[";
  for(size_t i=0;i<8;++i){if(i)s+=',';s+=row(phases[i]);}s+="],\"decode\":[";for(size_t i=0;i<decode_count;++i){if(i)s+=',';s+=row(decodes[i]);}s+="],\"totals\":"+row(total)+",\"phase_elapsed_ns\":[";for(size_t i=0;i<8;++i){if(i)s+=',';s+=std::to_string(phase_ns[i]);}return s+"],\"optional_system_metrics\":null}";
 }
};
// Delegates actual native I/O/lifetime. No polling from memory monitor.
class ObservedWindowsReader final:public WindowsReader {
 std::shared_ptr<WindowsReader> reader_;std::shared_ptr<RunIo> io_;
public:
 ObservedWindowsReader(std::shared_ptr<WindowsReader> r,std::shared_ptr<RunIo> io):reader_(std::move(r)),io_(std::move(io)){if(!reader_||!io_)throw std::invalid_argument("observed Reader ownership");}
 size_t read(uint64_t offset,unsigned char* dst,size_t n)override{
  const auto start=io_->tick();io_->add(0,1);io_->add(1,n);
  size_t got=0;try{got=reader_->read(offset,dst,n);}catch(...){
   io_->ledger(reader_->ledger_after_drain());io_->add(10,1);const auto end=io_->tick();io_->add(3,uint64_t(end-start));throw;
  }
  io_->ledger(reader_->ledger_after_drain());io_->add(2,got);const auto end=io_->tick();io_->add(3,uint64_t(end-start));return got;
 }
 void request_stop()noexcept override{reader_->request_stop();}
 void drain_to_terminal()noexcept override{reader_->drain_to_terminal();}
 Ledger ledger_after_drain()const override{return reader_->ledger_after_drain();}
 Cause cause_after_drain()const override{return reader_->cause_after_drain();}
};
}
