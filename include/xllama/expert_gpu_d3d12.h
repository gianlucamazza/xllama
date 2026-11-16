#pragma once
#include "../../llama.cpp/src/ua50/expert_gpu.h"
#include <functional>
namespace xllama {
struct ExpertGpuAdmission {
 std::string adapter_luid,shader_sha256;
 // Actual caller's native joint plan/quota guard; absent => refusal before resource allocation.
 std::function<void(const ua50::GpuSnapshot&,uint64_t pending_bytes)> admit;
};
// No fallback/non-Windows emulation. Caller must bind exact actual identity.
std::shared_ptr<ua50::ExpertGpu> make_expert_gpu_d3d12(const ExpertGpuAdmission&);
}
