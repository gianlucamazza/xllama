# LFM2.5 model-card sampling vs the global defaults (2026-11-27)

A paired evaluation on a project Xbox Series S in Developer Mode. It decides whether the LFM2.5
model cards' sampling values go into the catalogue (`uwp/models/manifest.json`, an entry's
`sampling`; `include/xllama/model_sampling.h`).

## Binding

- Package: `xllama_1.10.0.2000_x64.msix`, the v1.10.0 release package
  (`../2026-11-27-v1100-series-s-release/`). Requests went through the LAN API only.
- Models: `lfm25-1.2b-instruct` (QAD Q4_0, d3d12 GPU default) and `lfm25-8b-a1b` (UD-IQ3_S, CPU).
- Card values, from Liquid AI's model cards. The 8B's are also in its `generation_config.json`.
  Neither card sets `top_p`, so it stayed at the global 0.9 in both arms.

| Model                 | Card values                                        | Source                                                                                            |
| --------------------- | -------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| `lfm25-1.2b-instruct` | temperature 0.1, top_k 50, repetition_penalty 1.05 | [LiquidAI/LFM2.5-1.2B-Instruct](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct)             |
| `lfm25-8b-a1b`        | temperature 0.2, top_k 80, repetition_penalty 1.05 | [LiquidAI/LFM2.5-8B-A1B](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B), `generation_config.json` |

## Method

A client's agent eval suite (v1) ran each case through `/v1/chat/completions`. The cases are a
tool proposal, irrelevant tools, a multi-turn tool chain, abstention (declining to act without the
tools for it), and plain chat, in English and Italian. Each case is scored by checks on the tool,
arguments, content, abstention and reply language. Seeds are 0..n-1, one block per variant.

- **base:** temperature 0.1 from the request, with `top_p` 0.9, `top_k` 40 and
  `repetition_penalty` 1.1 from `include/xllama/sampling.h`.
- **card:** the card values in the request.

On the 1.2B both arms ran at temperature 0.1, so that arm tests `top_k` and
`repetition_penalty` only. The 8B ran a 95-case subset (tool proposal, irrelevance, multi-turn,
abstention and chat) with 2 trials.

The suite and its harness belong to the client and are not in this repository. Each receipt
records every row: the case, the reply's first 300 characters, the checks, usage and
`finish_reason`. Paths under the client's temporary directory read `/tmp/eval-<dir>`, and the
console's address reads `console`.

## Results

| Model                 | Cases × trials | base pass^1 [95% CI] | card pass^1 [95% CI] | Paired card − base [95% CI] | Better / worse | Sign test |
| --------------------- | -------------: | -------------------- | -------------------- | --------------------------- | -------------: | --------: |
| `lfm25-1.2b-instruct` |        160 × 3 | 0.746 [0.681, 0.808] | 0.748 [0.681, 0.815] | +0.002 [−0.008, +0.013]     |          4 / 3 |   p = 1.0 |
| `lfm25-8b-a1b`        |         95 × 2 | 0.842 [0.774, 0.900] | 0.858 [0.795, 0.916] | +0.016 [−0.042, +0.074]     |         10 / 9 |   p = 1.0 |

- Empty replies: 0 in each arm, except one on the 8B card arm, which stopped at the token limit
  (`finish_reason: length`).
- Median latency: 1.4 s on the 1.2B and 17 s on the 8B, the same in both arms.
- On the 8B, the cases that changed are abstention cases in both directions.

**Decision:** the card values go into the catalogue for these two models. The eval does not
contradict them and no regression was measured, but **no gain was measured either**. They are the
publisher's recommendation, and now one place holds them, for the app and every client. The other
LFM2.5 entries keep the global defaults until they are measured.

The eval compares the card values with the global defaults at the client's temperature (0.1). It
does not compare them with the app's previous temperature, 0.8: a console user who never moved
the Temperature slider now gets 0.1 (or 0.2 on the 8B), as the cards recommend.

## Finding beyond sampling: the 8B in plain chat

`lfm25-8b-a1b` passes 21/42 abstention cases on base and 19/42 on card. A rough count of the
failures that write a tool call as text, such as `[ls_files()]`, `[git_status(repo_path=…)]` or a
JSON `tool_calls` object, finds about a third of them in both arms. It also often answers an
Italian request in English (language match 92/118). Both are traits of the model, not of the
sampling, and they matter to anyone relying on its behaviour without tools.

## Files

- `lfm25-1.2b-instruct-base.json`, `lfm25-1.2b-instruct-card.json`, `lfm25-8b-a1b-base.json`,
  `lfm25-8b-a1b-card.json`: per-row results and summaries.
- `console-log.txt`: the run log, with each arm's summary and the paired comparison.
- `SHA256SUMS`: hashes of the files above as committed.
