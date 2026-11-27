# LFM2 tool-call parser backports, before and after (2026-11-27)

A before/after measurement on a project Xbox Series S in Developer Mode for #610 (#625): the
backports of upstream llama.cpp [#31937](https://github.com/ggml-org/llama.cpp/pull/31937)
(`patches/0006`, control characters in tool-call string arguments) and
[#32164](https://github.com/ggml-org/llama.cpp/pull/32164) (`patches/0007`, undeclared tool
calls, adapted to `tool_choice` `auto` only).

## Binding

- **before:** `xllama_1.10.0.2000_x64.msix`, the v1.10.0 release package. `/health`
  `runtime.patches` lists 0001–0005.
- **after:** `xllama_1.10.0.2009_x64.msix`, #625's CI package (build-uwp run #2009), installed in
  place over 2000. `/health` `runtime.patches` lists 0001–0007.
- Models: `lfm25-1.2b-instruct` (QAD Q4_0, d3d12 GPU default) and `lfm25-8b-a1b` (UD-IQ3_S,
  CPU), both on the catalogue's sampling as of 1.10.0, through the LAN API only. Both phases used
  the same seeds and order.

## Method

1. **Probes** (`*-probes.json`): direct `/v1/chat/completions` requests, 8 seeds each, temperature
   0.7, `max_tokens` 1024, one user message:
   - `newline`: `write_file(path, content)` offered with `auto`, and a request for a file with two
     lines. Exercises 0006.
   - `undecl`: only `get_weather` offered, `auto`, and a request to "call the tool
     reseat_sim_card". Exercises 0007.
   - `named`: `tool_choice` names `get_weather` on an unrelated request. It must stay on
     `get_weather`; this checks the 0007 adaptation.

   Each row records the first call's name, whether its arguments are a JSON object
   (`args_ok`), whether `content` holds a real newline, and whether raw call markers were left
   in the text (`raw_marker_in_text`).

2. **Eval** (`*-eval.json`): a client's agent eval suite (v1), the tool categories only, one block
   per model. `lfm25-1.2b-instruct` ran single, multi-turn, missing parameter, retrieval, web and
   irrelevance (119 cases × 2). `lfm25-8b-a1b` ran single, multi-turn and irrelevance
   (64 cases × 1). The JSON field `variant` is the suite's harness variant (`base` = default
   configuration) in both phases; the phase is in the file name.

The suite and its harness belong to the client and are not in this repository. Paths under the
client's temporary directory read `/tmp/eval-<dir>`, and the console's address reads `console`.
On the first before-phase run, the 8B probes stopped on a harness error: a call whose arguments
were valid JSON but not an object, now counted as `args_ok: false`. Only those probes were
rerun on 2000; `console-log.txt` shows both runs.

## Results

Probes, 8 seeds each:

| Probe     | Model                 | before                                                                                       | after                                                                             |
| --------- | --------------------- | -------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| `undecl`  | `lfm25-8b-a1b`        | 6 calls, **all forced onto `get_weather`**; 2 text replies                                   | 6 calls, **all `reseat_sim_card`**; 2 text replies                                |
| `undecl`  | `lfm25-1.2b-instruct` | 0 calls (text)                                                                               | 0 calls (text)                                                                    |
| `newline` | `lfm25-8b-a1b`        | 6 calls, **1 with arguments that are not valid JSON**; 1 raw-marker text; 1 JSON-shaped text | 6 calls, **all valid**, with real newlines; same raw-marker and JSON-shaped texts |
| `newline` | `lfm25-1.2b-instruct` | 8 valid calls                                                                                | 8 valid calls                                                                     |
| `named`   | both                  | 8/8 `get_weather`                                                                            | 8/8 `get_weather`                                                                 |

- The 1.2B escapes `\n` itself, so its `newline` probe does not reach the path 0006 changes.
- The 8B's remaining raw-marker reply (seed 1) is the same text in both phases. The model wrote
  that call malformed, so no parser change recovers it.

Eval, paired by case:

| Model                 | Cases × trials | before pass^1 [95% CI] | after pass^1 [95% CI] | Paired after − before [95% CI] | Better / worse | Sign test | text instead of a call |
| --------------------- | -------------: | ---------------------- | --------------------- | ------------------------------ | -------------: | --------: | ---------------------- |
| `lfm25-1.2b-instruct` |        119 × 2 | 0.710 [0.626, 0.786]   | 0.697 [0.613, 0.773]  | −0.013 [−0.034, +0.004]        |          1 / 4 | p = 0.375 | 23/162 → 23/162        |
| `lfm25-8b-a1b`        |         64 × 1 | 0.953 [0.891, 1.000]   | 0.984 [0.953, 1.000]  | +0.031 [0.000, +0.078]         |          2 / 0 |   p = 0.5 | 1/34 → 0/34            |

- The four 1.2B cases that got worse are choices among declared tools, for example
  `get_current_time` where no call was expected. Neither phase has a call to an undeclared tool in
  the eval, so 0007 is not involved; the change is within the interval.

**Decision:** keep both backports. Each fixes its bug where the bug occurs (the 8B), the
`required`/named adaptation holds on device, and the eval shows no regression outside noise.
