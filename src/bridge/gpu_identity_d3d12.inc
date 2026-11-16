// Included by existing Windows D3D bridge TU. No global gpu() use.
namespace {
std::string expert_shader_sha(){
 BCRYPT_ALG_HANDLE a=nullptr;BCRYPT_HASH_HANDLE h=nullptr;DWORD len=0,got=0;std::vector<unsigned char> obj;unsigned char out[32]{};
 auto ck=[](NTSTATUS x){if(x<0)throw std::runtime_error("GPU shader BCrypt identity failed");};
 try{ck(BCryptOpenAlgorithmProvider(&a,BCRYPT_SHA256_ALGORITHM,nullptr,0));ck(BCryptGetProperty(a,BCRYPT_OBJECT_LENGTH,reinterpret_cast<PUCHAR>(&len),sizeof(len),&got,0));if(got!=sizeof(len)||!len||len>65536)throw std::runtime_error("GPU hash object bound");obj.resize(len);ck(BCryptCreateHash(a,&h,obj.data(),len,nullptr,0,0));
  // Domain + explicit byte lengths + ACTUAL compiled-in DXIL, not requested digest.
  const unsigned char domain[]="ua68-q4k-q6k-t64-dxil-v1";ck(BCryptHashData(h,const_cast<PUCHAR>(domain),sizeof(domain),0));
  for(auto b:{std::pair<const unsigned char*,size_t>{kGgmlD3d12MmvQ4KT64Dxil,kGgmlD3d12MmvQ4KT64DxilSize},std::pair<const unsigned char*,size_t>{kGgmlD3d12MmvQ6KT64Dxil,kGgmlD3d12MmvQ6KT64DxilSize}}){unsigned char size[8]{};for(unsigned i=0;i<8;++i)size[i]=static_cast<unsigned char>(uint64_t(b.second)>>(8*i));ck(BCryptHashData(h,size,8,0));ck(BCryptHashData(h,const_cast<PUCHAR>(b.first),ULONG(b.second),0));}
  ck(BCryptFinishHash(h,out,sizeof(out),0));NTSTATUS dh=BCryptDestroyHash(h);h=nullptr;ck(dh);NTSTATUS da=BCryptCloseAlgorithmProvider(a,0);a=nullptr;ck(da);
 }catch(...){if(h)BCryptDestroyHash(h);if(a)BCryptCloseAlgorithmProvider(a,0);throw;}
 const char*hex="0123456789abcdef";std::string s;for(auto x:out){s+=hex[x>>4];s+=hex[x&15];}return s;
}

gpu_identity::Quota identity_quota(){
 using M=winrt::Windows::System::MemoryManager;auto limit=M::AppMemoryUsageLimit();auto usage=M::AppMemoryUsage();auto level=M::AppMemoryUsageLevel();
 uint32_t code=level==winrt::Windows::System::AppMemoryUsageLevel::Low?0:level==winrt::Windows::System::AppMemoryUsageLevel::Medium?1:level==winrt::Windows::System::AppMemoryUsageLevel::High?2:level==winrt::Windows::System::AppMemoryUsageLevel::OverLimit?3:99;
 if(!limit||limit!=M::AppMemoryUsageLimit()||usage>limit||code>1)throw std::runtime_error("identity unknown/changing/pressured quota");return {limit,usage,code};
}
void identity_admit(const gpu_identity::Quota&q,uint64_t initial,uint64_t metadata){
 if(q.limit!=initial||metadata>UINT64_MAX-gpu_identity::probe-gpu_identity::reserve||q.limit-q.usage<metadata+gpu_identity::probe+gpu_identity::reserve)throw std::runtime_error("identity reserve/probe/metadata admission refused");
}
}
namespace gpu_identity {
Observation query(const std::function<void()>&check,uint64_t metadata){
 if(!check)throw std::invalid_argument("identity check callback required");check();Observation o;o.before=identity_quota();identity_admit(o.before,o.before.limit,metadata);
 {
  // EXACT same selection helper as init_gpu; no queue/PSO/resource/submit.
  std::string error;auto device=d3d12c::create_device(&error);if(!device)throw std::runtime_error("identity temporary device: "+error);check();
  auto luid=device->GetAdapterLuid();char text[18];std::snprintf(text,sizeof(text),"%08x:%08x",unsigned(luid.HighPart),unsigned(luid.LowPart));o.luid=text;
  D3D12_FEATURE_DATA_ARCHITECTURE1 arch{};if(FAILED(device->CheckFeatureSupport(D3D12_FEATURE_ARCHITECTURE1,&arch,sizeof(arch))))throw std::runtime_error("identity architecture query failed");o.uma=arch.UMA!=FALSE;o.coherent=arch.CacheCoherentUMA!=FALSE;o.nodes=device->GetNodeCount();if(!o.nodes)throw std::runtime_error("identity node count unknown");
  ComPtr<IDXGIFactory4> factory;ComPtr<IDXGIAdapter3> adapter;
  if(FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))||FAILED(factory->EnumAdapterByLuid(luid,IID_PPV_ARGS(&adapter))))throw std::runtime_error("identity actual adapter query failed");
  DXGI_ADAPTER_DESC1 desc{};if(FAILED(adapter->GetDesc1(&desc))||(desc.Flags&DXGI_ADAPTER_FLAG_SOFTWARE))throw std::runtime_error("identity actual hardware required");o.vendor=desc.VendorId;o.device=desc.DeviceId;o.subsystem=desc.SubSysId;o.revision=desc.Revision;o.flags=desc.Flags;
  DXGI_QUERY_VIDEO_MEMORY_INFO mem{};if(FAILED(adapter->QueryVideoMemoryInfo(0,DXGI_MEMORY_SEGMENT_GROUP_LOCAL,&mem))||!mem.Budget||mem.CurrentUsage>mem.Budget)throw std::runtime_error("identity DXGI memory snapshot unknown");o.budget=mem.Budget;o.usage=mem.CurrentUsage;o.reservation=mem.CurrentReservation;o.available_reservation=mem.AvailableForReservation;
  check();o.shader=expert_shader_sha();check(); // same helper/actual compiled DXIL bytes as proof
  o.after_device=identity_quota();identity_admit(o.after_device,o.before.limit,metadata);o.observed_init_delta=o.after_device.usage>o.before.usage?o.after_device.usage-o.before.usage:0;if(o.observed_init_delta>probe)throw std::runtime_error("identity observed initialization exceeds probe");
  adapter.Reset();factory.Reset();device.Reset(); // temporary COM release before native result return
 }
 o.after_close=identity_quota();identity_admit(o.after_close,o.before.limit,metadata);check();return o;
}
}
