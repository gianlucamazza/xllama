# Changelog

All notable changes to xllama are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

- **Fixed:** Replaying a tool call through `/v1/chat/completions` no longer
  returns a false `400` `model chat template does not render tools`. The check
  counted the tool's name in transcript fields the template may not print: `name`
  on a `tool` message, or the assistant's calls on templates without a `tool_calls`
  branch (LFM2-2.6B, the catalogue's LFM2.5-1.2B-Thinking file). It now compares
  the prompt rendered with and without `tools`. Phi-4-mini is still refused.
