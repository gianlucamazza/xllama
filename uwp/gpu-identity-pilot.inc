// Identity only; included once by existing pilot TU. Never enters model/catalogue/Reader.
void run_gpu_capability_identity(){
 const std::string tag="[gpu-identity-v1]";char first[256]{};auto fail=[&](const char*s){if(!first[0])std::snprintf(first,sizeof(first),"%s",s?s:"unknown identity failure");};
 bool normal=false,resources_closed=true;auto limit_changed=std::make_shared<std::atomic<bool>>(false);uint64_t metadata=0,initial_limit=0;std::optional<gpu_identity::Observation> observation;
 std::shared_ptr<GpuIdentityInput> input;std::string package;auto wall=ua50::uwp::Clock::now()+std::chrono::seconds(30);
 // RAII for parse/claim failure; no output path is trusted until strict parse returns.
 try{
  auto compute=try_acquire_compute_job();if(!compute)throw std::runtime_error("identity compute lane busy");
  auto limit_event=winrt::Windows::System::MemoryManager::AppMemoryUsageLimitChanging(winrt::auto_revoke,[limit_changed](const auto&,const auto&)noexcept{limit_changed->store(true);});
  auto q=winrt::Windows::System::MemoryManager::AppMemoryUsageLimit();auto usage=winrt::Windows::System::MemoryManager::AppMemoryUsage();initial_limit=q;
  if(!q||usage>q||q-usage<gpu_identity::reserve+gpu_identity::probe+(256u<<10))throw std::runtime_error("identity preparse quota refused");
  const std::filesystem::path root(winrt::Windows::Storage::ApplicationData::Current().LocalFolder().Path().c_str());input=std::make_shared<GpuIdentityInput>(root);
  auto check=[&]{
   if(limit_changed->load()||ua50::uwp::Clock::now()>=wall||dev_utc_ms()>=input->request.expires)throw std::runtime_error("identity original deadline/expiry stop");
   const auto stop=input->path("stop");DWORD a=GetFileAttributesW(stop.c_str());if(a!=INVALID_FILE_ATTRIBUTES||GetLastError()!=ERROR_FILE_NOT_FOUND)throw std::runtime_error("identity stop requested/query unknown");
   using M=winrt::Windows::System::MemoryManager;const auto l=M::AppMemoryUsageLimit(),u=M::AppMemoryUsage();const auto level=M::AppMemoryUsageLevel();
   if(l!=initial_limit||!l||u>l||(level!=winrt::Windows::System::AppMemoryUsageLevel::Low&&level!=winrt::Windows::System::AppMemoryUsageLevel::Medium)||l!=M::AppMemoryUsageLimit()||metadata>UINT64_MAX-gpu_identity::reserve-gpu_identity::probe||l-u<metadata+gpu_identity::reserve+gpu_identity::probe)throw std::runtime_error("identity limit/usage/pressure/metadata stop");
  };
  metadata=add(input->owned_bytes(),sizeof(gpu_identity::Observation)+gpu_identity::serialization_capacity_cap+65536+8192);check();
  const auto output=input->path("output.json");DWORD old=GetFileAttributesW(output.c_str());if(old!=INVALID_FILE_ATTRIBUTES||GetLastError()!=ERROR_FILE_NOT_FOUND)throw std::runtime_error("identity old output/query unknown");
  package=winrt::to_string(winrt::Windows::ApplicationModel::Package::Current().Id().FullName());if(package.empty()||package.size()>1024)throw std::runtime_error("identity package bound");
  input->spend();check();resources_closed=false;
  observation=gpu_identity::query(check,metadata);resources_closed=true;check();normal=true;
 }catch(const std::exception&e){fail(e.what());}catch(...){fail("identity unknown failure");}
 // All query temporaries unwind before this serializer. Refused query retains conservative closure uncertainty.
 if(input){try{
  auto q=[](const gpu_identity::Quota&a){return std::string("{\"limit\":")+std::to_string(a.limit)+",\"usage\":"+std::to_string(a.usage)+",\"level\":"+std::to_string(a.level)+"}";};
  const auto quote=first_answer::quote;std::string body;body.reserve(gpu_identity::result_cap);
  body="{\"schema\":\"ua68-gpu-identity-result-v1\",\"contract_sha256\":"+quote(gpu_identity::contract_sha)+",\"normal\":"+(normal?"true":"false")+",\"status\":"+quote(normal?"IDENTITY_OBSERVED":"REFUSED")+",\"run_id\":"+quote(input->request.id)+",\"selection_id\":"+quote(input->request.selection)+",\"request_sha256\":"+quote(input->request_sha)+",\"native_package_full_name\":"+(package.empty()?std::string("null"):quote(package))+",\"pid\":"+std::to_string(GetCurrentProcessId())+",\"request_expires_at_epoch_ms\":"+std::to_string(input->request.expires)+",\"error\":"+quote(first)+",\"native_resources_closed\":"+(resources_closed?"true":"false")+",\"observation\":";
  if(normal&&observation){const auto&o=*observation;body+="{\"adapter_luid\":"+quote(o.luid)+",\"compiled_shader_sha256\":"+quote(o.shader);
   for(auto kv:{std::pair<const char*,uint64_t>{"vendor_id",o.vendor},{"device_id",o.device},{"subsystem_id",o.subsystem},{"revision",o.revision},{"adapter_flags",o.flags},{"node_count",o.nodes},{"DXGI_budget",o.budget},{"DXGI_usage",o.usage},{"DXGI_current_reservation",o.reservation},{"DXGI_available_for_reservation",o.available_reservation},{"observed_initialization_app_usage_delta",o.observed_init_delta}})body+=",\""+std::string(kv.first)+"\":"+std::to_string(kv.second);
   body+=",\"UMA\":"+std::string(o.uma?"true":"false")+",\"CacheCoherentUMA\":"+(o.coherent?"true":"false")+",\"quota_before\":"+q(o.before)+",\"quota_after_device\":"+q(o.after_device)+",\"quota_after_close\":"+q(o.after_close)+"}";
  }else body+="null";
  body+=",\"accounting\":{\"request_owned_bytes\":"+std::to_string(input->owned_bytes())+",\"metadata_charge_bytes\":"+std::to_string(metadata)+",\"serialization_capacity_cap\":65536,\"reserve_bytes\":536870912,\"probe_bytes\":268435456,\"sampled_not_peak\":true,\"UMA_extra_RAM\":false,\"model_plan_admitted\":false},\"model_opened\":false,\"model_bytes_read\":0,\"GPU_submits\":0,\"signed_artifact_authenticated_by_native\":false,\"driver_JIT_machinecode_identity_available\":false}";
  if(body.size()>gpu_identity::result_cap||body.capacity()>gpu_identity::serialization_capacity_cap)throw std::length_error("identity serialized output bound");
  // Even final serialization delay cannot publish normal after expiry/stop. Recheck without a new query.
  if(normal){DWORD s=GetFileAttributesW(input->path("stop").c_str());if(s!=INVALID_FILE_ATTRIBUTES||GetLastError()!=ERROR_FILE_NOT_FOUND||limit_changed->load()||ua50::uwp::Clock::now()>=wall||dev_utc_ms()>=input->request.expires||winrt::Windows::System::MemoryManager::AppMemoryUsageLimit()!=initial_limit)throw std::runtime_error("identity final publication stop");}
  if(input->held.value!=INVALID_HANDLE_VALUE){input->held.close();if(normal)throw std::runtime_error("identity unconsumed ticket cannot be normal");}
  DevHandle out(CreateFile2(input->path("output.json").c_str(),GENERIC_WRITE,FILE_SHARE_READ,CREATE_NEW,nullptr));if(out.value==INVALID_HANDLE_VALUE)throw std::runtime_error("identity output CREATE_NEW failed");DWORD n=0;if(!WriteFile(out.value,body.data(),DWORD(body.size()),&n,nullptr)||n!=body.size()||!FlushFileBuffers(out.value))throw std::runtime_error("identity output write/flush failed");out.close();
  log_output(tag+(normal?" OUTPUT_COMMITTED identity-only temporary resources closed\n":" OUTPUT_COMMITTED refused\n"));
 }catch(const std::exception&e){fail(e.what());normal=false;log_output(tag+" OUTPUT_FAILED: "+first+"\n");}catch(...){fail("identity unknown output failure");normal=false;log_output(tag+" OUTPUT_FAILED: "+first+"\n");}}
 else log_output(tag+" REFUSED: "+first+"\n");
 log_output(tag+(normal?" terminal identity-only\n":" terminal refused; no retry\n"));
}
