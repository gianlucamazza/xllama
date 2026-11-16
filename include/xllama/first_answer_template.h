#pragma once
#include <stdexcept>
#include <string>
namespace xllama::first_answer {
inline constexpr char template_sha[]="6d65c8804847ad74eea912dd7eca3dc1cf7a457b53a77f47d841a14121910963";
inline constexpr char embedded_template[]=R"LFM24({{- bos_token -}}
{%- set preserve_thinking = preserve_thinking | default(false) -%}

{%- macro format_arg_value(arg_value) -%}
    {%- if arg_value is string -%}
        {{- "'" + arg_value + "'" -}}
    {%- elif arg_value is mapping -%}
        {{- arg_value | tojson -}}
    {%- else -%}
        {{- arg_value | string -}}
    {%- endif -%}
{%- endmacro -%}

{%- macro parse_content(content) -%}
    {%- if content is string -%}
        {{- content -}}
    {%- else -%}
        {%- set _ns = namespace(result="") -%}
        {%- for item in content -%}
            {%- if item["type"] == "image" -%}
                {%- set _ns.result = _ns.result + "<image>" -%}
            {%- elif item["type"] == "text" -%}
                {%- set _ns.result = _ns.result + item["text"] -%}
            {%- else -%}
                {%- set _ns.result = _ns.result + item | tojson -%}
            {%- endif -%}
        {%- endfor -%}
        {{- _ns.result -}}
    {%- endif -%}
{%- endmacro -%}

{%- macro render_tool_calls(tool_calls) -%}
    {%- set tool_calls_ns = namespace(tool_calls=[]) -%}
    {%- for tool_call in tool_calls -%}
        {%- set func_name = tool_call["function"]["name"] -%}
        {%- set func_args = tool_call["function"]["arguments"] -%}
        {%- set args_ns = namespace(arg_strings=[]) -%}
        {%- for arg_name, arg_value in func_args.items() -%}
            {%- set args_ns.arg_strings = args_ns.arg_strings + [arg_name + "=" + format_arg_value(arg_value)] -%}
        {%- endfor -%}
        {%- set tool_calls_ns.tool_calls = tool_calls_ns.tool_calls + [func_name + "(" + (args_ns.arg_strings | join(", ")) + ")"] -%}
    {%- endfor -%}
    {{- "<|tool_call_start|>[" + (tool_calls_ns.tool_calls | join(", ")) + "]<|tool_call_end|>" -}}
{%- endmacro -%}

{%- set ns = namespace(system_prompt="", last_user_index=-1) -%}
{%- if messages[0]["role"] == "system" -%}
    {%- if messages[0].get("content") -%}
        {%- set ns.system_prompt = parse_content(messages[0]["content"]) -%}
    {%- endif -%}
    {%- set messages = messages[1:] -%}
{%- endif -%}
{%- if tools -%}
    {%- set ns.system_prompt = ns.system_prompt + ("\n" if ns.system_prompt else "") + "List of tools: [" -%}
    {%- for tool in tools -%}
        {%- if tool is not string -%}
            {%- set tool = tool | tojson -%}
        {%- endif -%}
        {%- set ns.system_prompt = ns.system_prompt + tool -%}
        {%- if not loop.last -%}
            {%- set ns.system_prompt = ns.system_prompt + ", " -%}
        {%- endif -%}
    {%- endfor -%}
    {%- set ns.system_prompt = ns.system_prompt + "]" -%}
{%- endif -%}
{%- if ns.system_prompt -%}
    {{- "<|im_start|>system\n" + ns.system_prompt + "<|im_end|>\n" -}}
{%- endif -%}
{%- for message in messages -%}
    {%- if message["role"] == "user" -%}
        {%- set ns.last_user_index = loop.index0 -%}
    {%- endif -%}
{%- endfor -%}
{%- for message in messages -%}
    {{- "<|im_start|>" + message.role + "\n" -}}
    {%- if message.role == "assistant" -%}
        {%- generation -%}
        {%- if message.thinking is defined and (preserve_thinking or loop.index0 > ns.last_user_index) -%}
            {{- "<think>" + message.thinking + "</think>" -}}
        {%- endif -%}
        {%- set _cfm_tag = "CONTINUE_FINAL_MESSAGE_TAG " -%}
        {%- set _has_cfm = false -%}
        {%- if message.content is defined -%}
            {%- set content = parse_content(message.content) -%}
            {%- if not (preserve_thinking or loop.index0 > ns.last_user_index) -%}
                {%- if "</think>" in content -%}
                    {%- set content = content.split("</think>")[-1] | trim -%}
                {%- endif -%}
            {%- endif -%}
            {%- if message.tool_calls is defined and content.endswith(_cfm_tag) -%}
                {%- set _has_cfm = true -%}
                {%- set _trunc_len = (content | length) - (_cfm_tag | length) -%}
                {{- content[:_trunc_len] -}}
            {%- else -%}
                {{- content -}}
            {%- endif -%}
        {%- endif -%}
        {%- if message.tool_calls is defined -%}
            {{- render_tool_calls(message.tool_calls) -}}
        {%- endif -%}
        {%- if _has_cfm -%}
            {{- _cfm_tag -}}
        {%- endif -%}
        {{- "<|im_end|>\n" -}}
        {%- endgeneration -%}
    {%- else %}
        {%- if message.get("content") -%}
            {{- parse_content(message["content"]) -}}
        {%- endif -%}
        {{- "<|im_end|>\n" -}}
    {%- endif %}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{- "<|im_start|>assistant\n" -}}
{%- endif -%})LFM24";
inline constexpr char greeting[]=u8"Привет! Представься в одном-двух предложениях и скажи, чем ты можешь помочь. Ответь кратко на русском.";
inline std::string render_user(const char* actual_template,bool add_bos,int bos,const std::string& user) {
 if(!actual_template || std::string(actual_template)!=embedded_template || !add_bos || bos<0)throw std::runtime_error("configured exact template/BOS mismatch");
 return std::string("<|im_start|>user\n")+user+"<|im_end|>\n<|im_start|>assistant\n<think>";
}
inline std::string render(const char* actual_template,bool add_bos,int bos) {
 if(!actual_template || std::string(actual_template)!=embedded_template || !add_bos || bos<0)throw std::runtime_error("first-answer exact template/BOS mismatch");
 // Exact one-user/no-system/no-tools subset of the pinned Jinja. BOS is tokenizer-owned.
 return std::string("<|im_start|>user\n")+greeting+"<|im_end|>\n<|im_start|>assistant\n<think>";
}
}
