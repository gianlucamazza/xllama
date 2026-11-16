// Included once by ggml_d3d12.cpp inside Windows xllama namespace; not a TU.
namespace {
using ExpertClock=std::chrono::steady_clock;
uint64_t expert_ns(ExpertClock::time_point a){auto n=std::chrono::duration_cast<std::chrono::nanoseconds>(ExpertClock::now()-a).count();if(n<0)throw std::runtime_error("GPU monotonic clock");return uint64_t(n);}
class ExpertD3DQueue final:public ua50::ExpertQueue {
 Gpu& g_;ExpertGpuAdmission admission_;ua50::GpuSnapshot info_;ua50::GpuCounts counts_;
 ComPtr<ID3D12Resource> weight_,q8_,out_;ComPtr<ID3D12Fence> fence_;d3d12c::FenceEvent event_;
 unsigned char*q8_ptr_=nullptr;float*out_ptr_=nullptr;UINT64 value_=0;bool inflight_=false,terminal_=true,weight_ready_=false;
 std::unique_lock<std::mutex> lock_;char cause_[256]{};
 void fail(const char*s)noexcept{if(!cause_[0])std::snprintf(cause_,sizeof(cause_),"%s",s);}
 void check(HRESULT hr,const char*what){if(FAILED(hr)){fail(d3d12c::hr_message(what,hr).c_str());throw std::runtime_error(cause_);}}
 uint64_t allocation(const D3D12_RESOURCE_DESC& d){auto a=g_.device->GetResourceAllocationInfo(0,1,&d);if(a.SizeInBytes==UINT64_MAX||!a.SizeInBytes||!a.Alignment)throw std::runtime_error("GPU allocation size unknown");return a.SizeInBytes;}
 D3D12_RESOURCE_DESC desc(uint64_t bytes,D3D12_RESOURCE_FLAGS f){D3D12_RESOURCE_DESC d{};d.Dimension=D3D12_RESOURCE_DIMENSION_BUFFER;d.Width=bytes;d.Height=1;d.DepthOrArraySize=1;d.MipLevels=1;d.SampleDesc.Count=1;d.Layout=D3D12_TEXTURE_LAYOUT_ROW_MAJOR;d.Flags=f;return d;}
 ua50::GpuSnapshot current()const{
  auto x=info_;x.app_limit=winrt::Windows::System::MemoryManager::AppMemoryUsageLimit();x.app_usage=winrt::Windows::System::MemoryManager::AppMemoryUsage();
  ComPtr<IDXGIFactory4> factory;ComPtr<IDXGIAdapter3> adapter;DXGI_QUERY_VIDEO_MEMORY_INFO m{};
  if(FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))||FAILED(factory->EnumAdapterByLuid(g_.luid,IID_PPV_ARGS(&adapter)))||FAILED(adapter->QueryVideoMemoryInfo(0,DXGI_MEMORY_SEGMENT_GROUP_LOCAL,&m)))throw std::runtime_error("GPU DXGI placement unknown");
  DXGI_ADAPTER_DESC1 descriptor{};if(FAILED(adapter->GetDesc1(&descriptor))||(descriptor.Flags&DXGI_ADAPTER_FLAG_SOFTWARE))throw std::runtime_error("actual hardware GPU adapter required");
  x.dxgi_budget=m.Budget;x.dxgi_usage=m.CurrentUsage;return x;
 }
 void quota(uint64_t pending){auto x=current();auto level=winrt::Windows::System::MemoryManager::AppMemoryUsageLevel();
  if(!x.app_limit||x.app_limit!=info_.app_limit||x.app_usage>x.app_limit||(level!=winrt::Windows::System::AppMemoryUsageLevel::Low&&level!=winrt::Windows::System::AppMemoryUsageLevel::Medium)||ua50::add(pending,512ull<<20)>x.app_limit-x.app_usage||!x.dxgi_budget||x.dxgi_usage>x.dxgi_budget||pending>x.dxgi_budget-x.dxgi_usage)throw std::runtime_error("hybrid fresh quota/placement/limit-change refused");
  admission_.admit(x,pending);
 }
 void barrier(D3D12_RESOURCE_STATES from,D3D12_RESOURCE_STATES to){D3D12_RESOURCE_BARRIER b{};b.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;b.Transition.pResource=weight_.Get();b.Transition.Subresource=D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;b.Transition.StateBefore=from;b.Transition.StateAfter=to;g_.cl->ResourceBarrier(1,&b);}
public:
 explicit ExpertD3DQueue(Gpu&g,const ExpertGpuAdmission&a,uint64_t init_delta):g_(g),admission_(a){
  if(a.adapter_luid.empty()||a.shader_sha256.empty()||!a.admit||!g.ok)throw std::invalid_argument("GPU exact identity/admission missing");
  std::lock_guard<std::mutex> guard(g.mu);char luid[32];std::snprintf(luid,sizeof(luid),"%08x:%08x",unsigned(g.luid.HighPart),unsigned(g.luid.LowPart));info_.adapter_luid=luid;info_.shader_sha256=expert_shader_sha();info_.cache_coherent_uma=true;
  info_.init_observed_app_delta=init_delta;info_.app_limit=winrt::Windows::System::MemoryManager::AppMemoryUsageLimit();
  if(info_.adapter_luid!=a.adapter_luid||info_.shader_sha256!=a.shader_sha256)throw std::runtime_error("constructed GPU identity mismatch");
  auto wd=desc(2580480+16,D3D12_RESOURCE_FLAG_NONE),qd=desc(1u<<20,D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),od=desc(8192+64,D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
  auto &l=info_.resources;
  for(auto b:{std::pair<const char*,ID3D12Resource*>{"stock_upload",g.staging.Get()}, {"stock_readback",g.readback.Get()}}){if(!b.second)throw std::runtime_error("stock resource absent");l.add(b.first,allocation(b.second->GetDesc()),true);}
  if(g.xq)l.add("stock_activation",allocation(g.xq->GetDesc()),true);
  if(g.ts_rb)l.add("stock_timestamps",allocation(g.ts_rb->GetDesc()),true);
  l.add("expert_weight",allocation(wd));l.add("expert_q8",allocation(qd));l.add("expert_output",allocation(od));
  // Queue/PSO/descriptor/driver allocation is not fabricated from sizeof: reserve full probe and check actual init delta.
  quota(l.total());const auto before_owned=current().app_usage;std::string err;
  weight_=d3d12c::create_buffer(g.device.Get(),wd.Width,D3D12_HEAP_TYPE_DEFAULT,D3D12_RESOURCE_FLAG_NONE,D3D12_RESOURCE_STATE_COPY_DEST,"ua68_expert_weight",&err);
  D3D12_HEAP_PROPERTIES hp{};hp.Type=D3D12_HEAP_TYPE_CUSTOM;hp.CPUPageProperty=D3D12_CPU_PAGE_PROPERTY_WRITE_BACK;hp.MemoryPoolPreference=D3D12_MEMORY_POOL_L0;
  q8_=d3d12c::create_buffer_props(g.device.Get(),qd.Width,hp,qd.Flags,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,"ua68_q8",&err);
  out_=d3d12c::create_buffer_props(g.device.Get(),od.Width,hp,od.Flags,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,"ua68_out",&err);
  if(!weight_||!q8_||!out_)throw std::runtime_error("expert resource creation: "+err);
  void*v=nullptr;check(q8_->Map(0,nullptr,&v),"Map expertQ8");q8_ptr_=static_cast<unsigned char*>(v);check(out_->Map(0,nullptr,&v),"Map expertOutput");out_ptr_=static_cast<float*>(v);
  check(g.device->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&fence_)),"Create expert fence");event_.h=CreateEventW(nullptr,FALSE,FALSE,nullptr);if(!event_.h)throw std::runtime_error("expert fence event unavailable");quota(0);
  const auto after_owned=current().app_usage;if(after_owned<before_owned||after_owned-before_owned>info_.resources.total())throw std::runtime_error("GPU owned creation/probe observation refused");info_.owned_creation_app_delta=after_owned-before_owned;
 }
 ~ExpertD3DQueue(){while(inflight_&&!wait_terminal())std::this_thread::yield();if(q8_ptr_)q8_->Unmap(0,nullptr);if(out_ptr_)out_->Unmap(0,nullptr);}
 void stage_submit(const ua50::ExpertVector&x)override{
  if(inflight_||cause_[0])throw std::runtime_error("GPU queue terminal/busy");quota(0);ua50::check_vector(x);
  lock_=std::unique_lock<std::mutex>(g_.mu);quota(0);auto t=ExpertClock::now();
  std::memcpy(g_.staging_ptr,x.weights,x.weight_bytes);std::memset(g_.staging_ptr+x.weight_bytes,0,16);std::memcpy(q8_ptr_,x.q8,x.q8_bytes);
  const uint32_t sentinel=0x7fc0dead;for(size_t i=0;i<2048;++i)std::memcpy(out_ptr_+i,&sentinel,4);std::memset(reinterpret_cast<unsigned char*>(out_ptr_)+8192,0xa5,64);counts_.copy_cpu_ns=ua50::add(counts_.copy_cpu_ns,expert_ns(t));
  check(g_.alloc->Reset(),"Reset expert allocator");check(g_.cl->Reset(g_.alloc.Get(),nullptr),"Reset expert commands");
  if(weight_ready_)barrier(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_COPY_DEST);
  g_.cl->CopyBufferRegion(weight_.Get(),0,g_.staging.Get(),0,x.weight_bytes+16);barrier(D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);weight_ready_=true;
  uint32_t constants[8]={x.n,x.k,x.k/256,uint32_t(ua50::block(x.quant).bytes*(x.k/256)),uint32_t(x.q8_bytes),uint32_t(x.n),0,0};
  g_.cl->SetComputeRootSignature(g_.root.Get());g_.cl->SetPipelineState(g_.pso[x.quant==ua50::Quant::q4_k?kPsoQ4K:kPsoQ6K][0].Get());
  g_.cl->SetComputeRoot32BitConstants(0,8,constants,0);g_.cl->SetComputeRootShaderResourceView(1,weight_->GetGPUVirtualAddress());g_.cl->SetComputeRootUnorderedAccessView(2,out_->GetGPUVirtualAddress());g_.cl->SetComputeRootUnorderedAccessView(3,q8_->GetGPUVirtualAddress());
  g_.cl->Dispatch((x.n+3)/4,1,1);D3D12_RESOURCE_BARRIER visibility{};visibility.Type=D3D12_RESOURCE_BARRIER_TYPE_UAV;visibility.UAV.pResource=out_.Get();g_.cl->ResourceBarrier(1,&visibility);check(g_.cl->Close(),"Close expert commands");
  ID3D12CommandList*lists[]={g_.cl.Get()};inflight_=true;terminal_=false;++counts_.submits;g_.queue->ExecuteCommandLists(1,lists);
  ++value_;counts_.fence_value=value_;counts_.terminal=false;
  HRESULT hr=g_.queue->Signal(fence_.Get(),value_);if(FAILED(hr)){fail(d3d12c::hr_message("expert Signal",hr).c_str());value_=UINT64_MAX;}
 }
 bool wait_terminal()noexcept override{
  if(!inflight_)return true;
  struct PollTimer{ExpertD3DQueue*self;ExpertClock::time_point start=ExpertClock::now();~PollTimer(){try{self->counts_.fence_wait_ns=ua50::add(self->counts_.fence_wait_ns,expert_ns(start));}catch(...){self->fail("expert wait clock overflow");}}} timer{this};
  auto removed=g_.device->GetDeviceRemovedReason();if(FAILED(removed)){fail(d3d12c::hr_message("expert device removed",removed).c_str());terminal_=true;return true;}
  if(value_==UINT64_MAX)return false; // failed signal: NO terminal proof while live device
  UINT64 done=fence_->GetCompletedValue();if(done==UINT64_MAX){fail("expert fence invalid; waiting device terminal");return false;}
  if(done<value_){HRESULT hr=fence_->SetEventOnCompletion(value_,event_.h);if(FAILED(hr)){fail(d3d12c::hr_message("expert SetEventOnCompletion",hr).c_str());return false;}DWORD w=WaitForSingleObject(event_.h,25);if(w==WAIT_FAILED){fail("expert WaitForSingleObject failed");return false;}if(w!=WAIT_OBJECT_0&&w!=WAIT_TIMEOUT){fail("expert wait unexpected result");return false;}done=fence_->GetCompletedValue();if(done==UINT64_MAX||done<value_)return false;}
  terminal_=true;counts_.terminal=true;counts_.completed++;
  return true;
 }
 const float*output()override{
  quota(0);if(inflight_||!terminal_||cause_[0])throw std::runtime_error("GPU output before actual terminal");
  const auto*guard=reinterpret_cast<const unsigned char*>(out_ptr_)+8192;for(size_t i=0;i<64;++i)if(guard[i]!=0xa5)throw std::runtime_error("GPU output guard changed");return out_ptr_;
 }
 void release_terminal()noexcept override{if(!terminal_)std::terminate();inflight_=false;counts_.terminal=true;if(lock_.owns_lock())lock_.unlock();}
 bool submitted()const noexcept override{return inflight_;}
 const char*error()const noexcept override{return cause_;}
 ua50::GpuSnapshot snapshot()const override{return current();}
 ua50::GpuCounts counts()const override{return counts_;} // timestamps explicitly unavailable; no invented zero duration
 size_t owned_bytes()const override{return size_t(ua50::add(ua50::add(sizeof(*this)+4096,info_.adapter_luid.capacity()+info_.shader_sha256.capacity()+admission_.adapter_luid.capacity()+admission_.shader_sha256.capacity()),info_.resources.total()));}
};
} // anonymous
std::shared_ptr<ua50::ExpertGpu> make_expert_gpu_d3d12(const ExpertGpuAdmission&a){
 if(!a.admit)throw std::invalid_argument("GPU native admission required");
 const uint64_t before=winrt::Windows::System::MemoryManager::AppMemoryUsage();const uint64_t limit=winrt::Windows::System::MemoryManager::AppMemoryUsageLimit();
 if(!limit||before>limit||limit-before<(512ull<<20)+(256ull<<20))throw std::runtime_error("GPU initialization reserve/probe refused");
 auto&g=gpu();const uint64_t after=winrt::Windows::System::MemoryManager::AppMemoryUsage();
 if(!g.ok||winrt::Windows::System::MemoryManager::AppMemoryUsageLimit()!=limit||after<before||after-before>(256ull<<20)||after>limit-(512ull<<20))throw std::runtime_error("GPU actual initialization/probe refused");
 return std::make_shared<ua50::SynchronousExpertGpu>(std::make_shared<ExpertD3DQueue>(g,a,after-before));
}
