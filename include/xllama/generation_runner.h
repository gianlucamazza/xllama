#pragma once
#include <string>
namespace xllama::bridge {
struct GenerationReply {int http_status=500;std::string body;};
GenerationReply run_configured_generation(std::string raw);
GenerationReply configured_generation_status(const std::string&run_id);
GenerationReply stop_configured_generation(const std::string&run_id);
std::string configured_generation_capabilities();
}
