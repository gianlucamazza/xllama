// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// The only translation unit that includes llama.cpp common/chat.h. Tool
// proposals render and parse through the GGUF Jinja template. This file does
// not execute a tool, start a process, or open a socket.

#include "xllama/tool_template.h"

#include "xllama/platform.h"

#include <string>
#include <utility>
#include <vector>

#ifdef XLLAMA_USE_LLAMA

    #include "chat.h"
    #include "common.h" // regex_escape

    #include <nlohmann/json.hpp>

    #include "llama.h"

namespace xllama {
namespace {

bool arguments_complete(const std::string& arguments) {
    if (arguments.empty())
        return false;
    try {
        const nlohmann::json parsed = nlohmann::json::parse(arguments);
        return !parsed.is_discarded();
    } catch (const std::exception&) {
        return false;
    }
}

common_chat_tool_choice map_choice(ToolChoice choice) {
    switch (choice) {
    case ToolChoice::Required:
    case ToolChoice::Named:
        // common/chat has no named-function choice. The named tool is the only
        // schema passed in, and REQUIRED asks the template for a call.
        return COMMON_CHAT_TOOL_CHOICE_REQUIRED;
    case ToolChoice::None:
        return COMMON_CHAT_TOOL_CHOICE_NONE;
    case ToolChoice::Auto:
        return COMMON_CHAT_TOOL_CHOICE_AUTO;
    }
    return COMMON_CHAT_TOOL_CHOICE_AUTO;
}

struct Applied {
    bool ok = false;
    std::string error;
    common_chat_params params;
};

size_t count_occurrences(const std::string& text, const std::string& needle) {
    size_t n = 0;
    for (size_t at = text.find(needle); at != std::string::npos;
         at = text.find(needle, at + needle.size()))
        ++n;
    return n;
}

bool template_renders_tools(const std::string& prompt, const std::vector<ToolSpec>& offered,
                            const std::vector<common_chat_msg>& messages) {
    for (const ToolSpec& tool : offered) {
        size_t in_transcript = 0;
        for (const common_chat_msg& message : messages) {
            in_transcript += count_occurrences(message.content, tool.name);
            in_transcript += count_occurrences(message.tool_name, tool.name);
            for (const common_chat_tool_call& call : message.tool_calls)
                in_transcript += count_occurrences(call.name, tool.name) +
                                 count_occurrences(call.arguments, tool.name);
        }
        if (count_occurrences(prompt, tool.name) > in_transcript)
            return true;
    }
    return false;
}

// Triggers in the form llama_sampler_init_grammar_lazy_patterns takes, built
// exactly as common/sampling.cpp does for llama-server.
GrammarSpec grammar_spec(const common_chat_params& params) {
    GrammarSpec spec;
    spec.gbnf = params.grammar;
    spec.lazy = params.grammar_lazy;
    spec.generation_prompt = params.generation_prompt;
    for (const common_grammar_trigger& trigger : params.grammar_triggers) {
        switch (trigger.type) {
        case COMMON_GRAMMAR_TRIGGER_TYPE_WORD:
            spec.trigger_patterns.push_back(regex_escape(trigger.value));
            break;
        case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN:
            spec.trigger_patterns.push_back(trigger.value);
            break;
        case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL: {
            const std::string& p = trigger.value;
            std::string anchored = "^$";
            if (!p.empty())
                anchored = (p.front() != '^' ? "^" : "") + p + (p.back() != '$' ? "$" : "");
            spec.trigger_patterns.push_back(anchored);
            break;
        }
        case COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN:
            spec.trigger_tokens.push_back(trigger.token);
            break;
        }
    }
    return spec;
}

Applied apply_template(const void* llama_model, const std::string& jinja_override,
                       const std::vector<ChatMessage>& messages, const std::vector<ToolSpec>& tools,
                       ToolChoice choice, const std::string& forced_name) {
    Applied applied;
    if (llama_model == nullptr && jinja_override.empty()) {
        applied.error = "tools require a GGUF chat template";
        return applied;
    }

    std::vector<ToolSpec> offered = tools;
    if (choice == ToolChoice::Named) {
        offered.clear();
        for (const ToolSpec& tool : tools) {
            if (tool.name == forced_name)
                offered.push_back(tool);
        }
        if (offered.empty()) {
            applied.error = "tool_choice names an unknown tool";
            return applied;
        }
    }

    common_chat_templates_ptr templates;
    try {
        if (llama_model != nullptr) {
            templates =
                common_chat_templates_init(static_cast<const struct llama_model*>(llama_model), "");
        } else {
            templates = common_chat_templates_init(nullptr, jinja_override);
        }
    } catch (const std::exception& ex) {
        applied.error = std::string("model chat template failed: ") + ex.what();
        return applied;
    }
    if (!templates ||
        (llama_model != nullptr && !common_chat_templates_was_explicit(templates.get()))) {
        applied.error = "model has no chat template";
        return applied;
    }

    common_chat_templates_inputs inputs;
    inputs.add_generation_prompt = true;
    inputs.use_jinja = true;
    inputs.parallel_tool_calls = true;
    inputs.reasoning_format = COMMON_REASONING_FORMAT_AUTO;
    inputs.enable_thinking = true;
    inputs.tool_choice = map_choice(choice);
    for (const ToolSpec& tool : offered) {
        common_chat_tool spec;
        spec.name = tool.name;
        spec.description = tool.description;
        spec.parameters = tool.parameters_json.empty() ? "{}" : tool.parameters_json;
        inputs.tools.push_back(std::move(spec));
    }
    for (const ChatMessage& message : messages) {
        common_chat_msg out;
        out.role = chat_role_name(message.role);
        out.content = message.content;
        out.tool_call_id = message.tool_call_id;
        out.tool_name = message.name;
        for (const ToolCall& call : message.tool_calls) {
            common_chat_tool_call mapped;
            mapped.id = call.id;
            mapped.name = call.name;
            mapped.arguments = call.arguments_json.empty() ? "{}" : call.arguments_json;
            out.tool_calls.push_back(std::move(mapped));
        }
        inputs.messages.push_back(std::move(out));
    }

    try {
        applied.params = common_chat_templates_apply(templates.get(), inputs);
    } catch (const std::exception& ex) {
        applied.error = std::string("model chat template failed: ") + ex.what();
        return applied;
    }
    // A template that ignores `tools` (Phi-4-mini reads a system message field
    // instead) renders without error and the model never sees the schema. The
    // schema reached the prompt only if a tool name appears there more often
    // than the transcript itself mentions it.
    if (!template_renders_tools(applied.params.prompt, offered, inputs.messages)) {
        applied.error = "model chat template does not render tools";
        return applied;
    }
    applied.ok = true;
    return applied;
}

std::string parsed_content(const common_chat_msg& message) {
    if (!message.content.empty())
        return message.content;
    std::string text;
    for (const common_chat_msg_content_part& part : message.content_parts) {
        if (part.type == "text")
            text += part.text;
    }
    return text;
}

} // namespace

ToolTemplateResult render_tool_prompt(const void* llama_model, const std::string& jinja_override,
                                      const std::vector<ChatMessage>& messages,
                                      const std::vector<ToolSpec>& tools, ToolChoice choice,
                                      const std::string& forced_name) {
    ToolTemplateResult out;
    const Applied applied =
        apply_template(llama_model, jinja_override, messages, tools, choice, forced_name);
    if (!applied.ok) {
        out.error = applied.error;
        return out;
    }
    out.ok = true;
    out.prompt = applied.params.prompt;
    out.stop_sequences = applied.params.additional_stops;
    out.preserved_tokens = applied.params.preserved_tokens;
    out.grammar = grammar_spec(applied.params);
    return out;
}

ToolParseResult parse_tool_generation(const void* llama_model, const std::string& jinja_override,
                                      const std::vector<ChatMessage>& messages,
                                      const std::vector<ToolSpec>& tools, ToolChoice choice,
                                      const std::string& forced_name, const std::string& raw_text,
                                      bool hit_cap) {
    ToolParseResult out;
    const Applied applied =
        apply_template(llama_model, jinja_override, messages, tools, choice, forced_name);
    if (!applied.ok) {
        log_output(std::string("[xllama] api: tool template unavailable after generation: ") +
                   applied.error + "\n");
        out.parser_threw = true;
        return out;
    }
    try {
        common_chat_parser_params parser_params(applied.params);
        parser_params.parse_tool_calls = true;
        if (!applied.params.parser.empty())
            parser_params.parser.load(applied.params.parser);
        const common_chat_msg message = common_chat_parse(raw_text, hit_cap, parser_params);
        out.content = parsed_content(message);
        for (const common_chat_tool_call& call : message.tool_calls) {
            IncomingCall incoming;
            incoming.id = call.id;
            incoming.name = call.name;
            incoming.arguments = call.arguments;
            incoming.complete = !hit_cap || arguments_complete(call.arguments);
            out.calls.push_back(std::move(incoming));
        }
    } catch (const std::exception&) {
        log_output("[xllama] api: tool output did not parse; returning text\n");
        out.parser_threw = true;
        out.calls.clear();
        out.content.clear();
    }
    return out;
}

} // namespace xllama

#else

namespace xllama {

ToolTemplateResult render_tool_prompt(const void*, const std::string&,
                                      const std::vector<ChatMessage>&, const std::vector<ToolSpec>&,
                                      ToolChoice, const std::string&) {
    ToolTemplateResult out;
    out.error = "tools require a GGUF chat template";
    return out;
}

ToolParseResult parse_tool_generation(const void*, const std::string&,
                                      const std::vector<ChatMessage>&, const std::vector<ToolSpec>&,
                                      ToolChoice, const std::string&, const std::string&, bool) {
    ToolParseResult out;
    out.parser_threw = true;
    return out;
}

} // namespace xllama

#endif
