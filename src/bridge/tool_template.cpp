// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// The only translation unit that includes llama.cpp common/chat.h. Tool
// proposals render and parse through the GGUF Jinja template. This file does
// not execute a tool, start a process, or open a socket.

#include "xllama/tool_template.h"

#include "xllama/chat_prompt.h"
#include "xllama/platform.h"

#include <string>
#include <utility>
#include <vector>

#ifdef XLLAMA_USE_LLAMA

    #include "chat.h"
    #include "common.h" // regex_escape

    #include <nlohmann/json.hpp>

    #include "gguf.h"
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

// Some LFM2/LFM2.5 chat templates print an assistant message's `content` and
// never read its `tool_calls` (llama.cpp models/templates LFM2.5-Instruct.jinja
// and LFM2-8B-A1B.jinja; the lfm2-2.6b GGUF). A replayed tool turn then shows
// the tool result with no call before it (#664). The family is the one
// common/chat.cpp routes to its LFM2 parser: "List of tools: [" (LFM2.5) or
// the <|tool_list_start|> / <|tool_list_end|> pair (LFM2). A template that
// names `tool_calls` renders the calls itself and is left alone.
bool lfm_template_drops_tool_calls(const std::string& source) {
    const bool lfm = source.find("List of tools: [") != std::string::npos ||
                     (source.find("<|tool_list_start|>") != std::string::npos &&
                      source.find("<|tool_list_end|>") != std::string::npos);
    return lfm && source.find("tool_calls") == std::string::npos;
}

// One argument value in the model's own call syntax, as the LFM2.5 GGUF
// template that does render calls writes it: a string in single quotes with
// \\, \', \n and \r escaped; True / False / None; numbers, objects and arrays
// as JSON. The LFM2 parser reads each form back to the same JSON value
// (normalize_quotes_to_json in common/chat-peg-parser.cpp).
std::string lfm_arg_value(const nlohmann::ordered_json& value) {
    if (value.is_string()) {
        std::string out = "'";
        for (const char c : value.get_ref<const std::string&>()) {
            switch (c) {
            case '\\':
                out += "\\\\";
                break;
            case '\'':
                out += "\\'";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            default:
                out += c;
            }
        }
        return out + "'";
    }
    if (value.is_boolean())
        return value.get<bool>() ? "True" : "False";
    if (value.is_null())
        return "None";
    return value.dump();
}

// Appends <|tool_call_start|>[name(arg=value, ...), ...]<|tool_call_end|> to an
// assistant message's content. False when a call's arguments are not a JSON
// object: the syntax has no form for them.
bool append_lfm_calls(std::string& content, const std::vector<ToolCall>& calls) {
    if (calls.empty() || content.find("<|tool_call_start|>") != std::string::npos)
        return true; // nothing to add, or the client replayed the raw call text
    std::string text = "<|tool_call_start|>[";
    for (size_t i = 0; i < calls.size(); ++i) {
        const std::string& raw = calls[i].arguments_json;
        const auto args = nlohmann::ordered_json::parse(raw.empty() ? "{}" : raw, nullptr, false);
        if (!args.is_object())
            return false;
        text += (i ? ", " : "") + calls[i].name + "(";
        bool first = true;
        for (const auto& [name, value] : args.items()) {
            text += (first ? "" : ", ") + name + "=" + lfm_arg_value(value);
            first = false;
        }
        text += ")";
    }
    content += text + "]<|tool_call_end|>";
    return true;
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
    std::optional<std::string> prompt_without_tools;
};

size_t count_occurrences(const std::string& text, const std::string& needle) {
    size_t n = 0;
    for (size_t at = text.find(needle); at != std::string::npos;
         at = text.find(needle, at + needle.size()))
        ++n;
    return n;
}

// A template that ignores `tools` (Phi-4-mini reads a system message field
// instead) renders without error and the model never sees the schema. The schema
// reached the prompt only if a tool name occurs more often with `tools` than
// without: counting the transcript cannot say which of its fields a template prints.
bool template_renders_tools(const common_chat_templates* templates,
                            const common_chat_templates_inputs& inputs, const std::string& prompt,
                            const std::vector<ToolSpec>& offered,
                            std::optional<std::string>& baseline) {
    common_chat_templates_inputs without_tools = inputs;
    without_tools.tools.clear();
    without_tools.tool_choice = COMMON_CHAT_TOOL_CHOICE_AUTO;
    try {
        baseline = common_chat_templates_apply(templates, without_tools).prompt;
    } catch (const std::exception&) {
        return true; // it needs `tools` to render this transcript, so it reads them
    }
    for (const ToolSpec& tool : offered) {
        if (count_occurrences(prompt, tool.name) > count_occurrences(*baseline, tool.name))
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

Applied apply_template(const void* llama_model, bool enable_thinking,
                       const std::string& jinja_override, const std::vector<ChatMessage>& messages,
                       const std::vector<ToolSpec>& tools, ToolChoice choice,
                       const std::string& forced_name) {
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

    // The variant common_chat_templates_apply renders: tool_use when tools are
    // offered and the model has one, else the default.
    std::string source =
        offered.empty() ? std::string() : common_chat_templates_source(templates.get(), "tool_use");
    if (source.empty())
        source = common_chat_templates_source(templates.get());
    const bool replay_lfm_calls = lfm_template_drops_tool_calls(source);

    common_chat_templates_inputs inputs;
    inputs.add_generation_prompt = true;
    inputs.use_jinja = true;
    inputs.parallel_tool_calls = true;
    inputs.reasoning_format = COMMON_REASONING_FORMAT_AUTO;
    // The decision plain chat renders (#655, #648): with it off a Qwen3,
    // MiniCPM5 or SmolLM3 template renders its no-think prefill instead of
    // reasoning through max_tokens on every tool turn.
    inputs.enable_thinking = enable_thinking;
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
        if (replay_lfm_calls && !append_lfm_calls(out.content, message.tool_calls)) {
            applied.error = "tool call arguments must be a JSON object for this model";
            return applied;
        }
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
    if (!offered.empty() && !template_renders_tools(templates.get(), inputs, applied.params.prompt,
                                                    offered, applied.prompt_without_tools)) {
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

ToolTemplateResult render_tool_prompt(const void* llama_model, bool enable_thinking,
                                      const std::string& jinja_override,
                                      const std::vector<ChatMessage>& messages,
                                      const std::vector<ToolSpec>& tools, ToolChoice choice,
                                      const std::string& forced_name) {
    ToolTemplateResult out;
    Applied applied = apply_template(llama_model, enable_thinking, jinja_override, messages, tools,
                                     choice, forced_name);
    if (!applied.ok) {
        out.error = applied.error;
        return out;
    }
    out.ok = true;
    out.prompt = std::move(applied.params.prompt);
    out.prompt_without_tools = std::move(applied.prompt_without_tools);
    out.stop_sequences = applied.params.additional_stops;
    out.preserved_tokens = applied.params.preserved_tokens;
    out.grammar = grammar_spec(applied.params);
    return out;
}

ToolParseResult parse_tool_generation(const void* llama_model, bool enable_thinking,
                                      const std::string& jinja_override,
                                      const std::vector<ChatMessage>& messages,
                                      const std::vector<ToolSpec>& tools, ToolChoice choice,
                                      const std::string& forced_name, const std::string& raw_text,
                                      bool hit_cap) {
    ToolParseResult out;
    const Applied applied = apply_template(llama_model, enable_thinking, jinja_override, messages,
                                           tools, choice, forced_name);
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
        out.reasoning = message.reasoning_content;
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
        out.reasoning.clear();
    }
    return out;
}

bool chat_template_renders_tools(const std::string& jinja) {
    if (jinja.empty())
        return false;
    ChatMessage user;
    user.role = ChatRole::User;
    user.content = "Hello.";
    ToolSpec probe;
    probe.name = "xllama_probe";
    probe.description = "Probe tool.";
    probe.parameters_json = R"({"type":"object","properties":{}})";
    return render_tool_prompt(nullptr, "", jinja, {user}, {probe}, ToolChoice::Auto, "").ok;
}

std::string gguf_chat_template(const std::string& gguf_path) {
    gguf_init_params params = {};
    params.no_alloc = true;
    params.ctx = nullptr;
    gguf_context* g = gguf_init_from_file(gguf_path.c_str(), params);
    if (!g)
        return {};
    std::string jinja;
    const int64_t key = gguf_find_key(g, "tokenizer.chat_template");
    if (key >= 0 && gguf_get_kv_type(g, key) == GGUF_TYPE_STRING)
        jinja = gguf_get_val_str(g, key);
    gguf_free(g);
    return jinja;
}

} // namespace xllama

#else

namespace xllama {

ToolTemplateResult render_tool_prompt(const void*, const std::string&, const std::string&,
                                      const std::vector<ChatMessage>&, const std::vector<ToolSpec>&,
                                      ToolChoice, const std::string&) {
    ToolTemplateResult out;
    out.error = "tools require a GGUF chat template";
    return out;
}

ToolParseResult parse_tool_generation(const void*, const std::string&, const std::string&,
                                      const std::vector<ChatMessage>&, const std::vector<ToolSpec>&,
                                      ToolChoice, const std::string&, const std::string&, bool) {
    ToolParseResult out;
    out.parser_threw = true;
    return out;
}

bool chat_template_renders_tools(const std::string&) {
    return false;
}

std::string gguf_chat_template(const std::string&) {
    return {};
}

} // namespace xllama

#endif
