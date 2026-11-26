# Long-prompt GGUF prefill knobs on Series S (#582)

Manifest, frozen before the first cell runs. Results are added below it and
the design is not edited afterwards.

## Question

Prefill throughput on Series S falls as the prompt grows
(`../2026-11-25-nctx-train-bound/`: `lfm25-350m` 354 → 147 tok/s from 1.7k to
30k tokens, `llama32-3b` 29 tok/s at 7.5k). Does one llama.cpp context knob
change that curve? The campaign is an estimate: it reports each knob's paired
ratio with its interval, and has no pass margin.

## Design

- Package: CI `build-uwp` of `feat/prefill-fa-knob` (1.8.1.1908), installed in
  place. Series S, operator-declared (`--hardware xbox-series-s`), CPU backend.
- Driver: `scripts/bench-paired-knob.sh`, one campaign per cell and knob. Arm A
  is the shipped configuration and arm B the same with one knob changed. Pairs
  alternate AB/BA. Each arm is `bench-xbox-ort.sh --runs 2`: run 1 is a dropped
  warmup and run 2 the recorded row. `--retries 2` reruns an arm whose harness
  call failed before writing a row.
- Decode is fixed at 16 tokens (`--n-predict 16 --ignore-eog`); the metric is
  prefill.
- Summary: `scripts/paired-bench-summary.py <dir> --report prefill,ttft,peak`.
  It gives the median paired ratio (B/A tok/s, B/A TTFT) or delta (peak working
  set) over complete pairs, with a paired percentile bootstrap 95 % interval.

Cells (prompts in `prompts/`, token counts from the LFM tokenizer on the host):

| Cell       | Model        | Prompt        | Tokens (LFM) | `--ctx` | Pairs per knob |
| ---------- | ------------ | ------------- | ------------ | ------- | -------------- |
| `lfm-8k`   | `lfm25-350m` | `prefill-8k`  | 7209         | 8192    | 3              |
| `lfm-16k`  | `lfm25-350m` | `prefill-16k` | 14387        | 16384   | 3              |
| `llama-4k` | `llama32-3b` | `prefill-4k`  | 3626         | 8192    | 3              |

Knobs (arm B options), each with the app-log lines both arms must show. A
missing expected line or a forbidden one stops the campaign:

| Knob    | Arm B option     | B must log                                            | A must log, and not                                                                               |
| ------- | ---------------- | ----------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| `fa0`   | `--flash-attn 0` | `flash attention: off`, not `Flash Attention enabled` | `Flash Attention enabled`, not `flash attention: off`, `prefill batch override`, `KV cache: q8_0` |
| `u256`  | `--ubatch 256`   | `n_ubatch=256`                                        | same as above                                                                                     |
| `u1024` | `--ubatch 1024`  | `n_ubatch=1024`                                       | same as above                                                                                     |
| `kvq8`  | `--kv-q8`        | `KV cache: q8_0`                                      | same as above                                                                                     |

Thread count is not a knob here: #168 already set prefill threads equal to
decode threads, and `bench_threads.txt` is a separate, older sweep.

## Commands

```bash
source ~/.config/xllama/xbox-env
C=bench/results/2026-11-26-prefill-knobs
run() { # cell model prompt ctx knob b-args...
  local cell=$1 model=$2 prompt=$3 ctx=$4 knob=$5; shift 5
  local bargs=(); for a in "$@"; do bargs+=(--b-arg "$a"); done
  scripts/bench-paired-knob.sh --model "$model" --prompt "$C/prompts/$prompt.txt" --pairs 3 \
    "${bargs[@]}" --out-dir "$C/$cell/$knob" --retries 2 --save-arm-logs \
    --expect-a 'Flash Attention enabled' \
    --forbid-a 'flash attention: off' --forbid-a 'prefill batch override' --forbid-a 'KV cache: q8_0' \
    "${EXPECT_B[@]}" \
    --hardware xbox-series-s --ctx "$ctx" --n-predict 16 --ignore-eog --timeout 1500
}
for cell in "lfm-8k lfm25-350m prefill-8k 8192" "lfm-16k lfm25-350m prefill-16k 16384" \
            "llama-4k llama32-3b prefill-4k 8192"; do
  set -- $cell
  EXPECT_B=(--expect-b 'flash attention: off' --forbid-b 'Flash Attention enabled')
  run "$@" fa0 --flash-attn 0
  EXPECT_B=(--expect-b 'n_ubatch=256');  run "$@" u256 --ubatch 256
  EXPECT_B=(--expect-b 'n_ubatch=1024'); run "$@" u1024 --ubatch 1024
  EXPECT_B=(--expect-b 'KV cache: q8_0'); run "$@" kvq8 --kv-q8
done
for d in "$C"/*/*/; do echo "$d"; scripts/paired-bench-summary.py "$d" --report prefill,ttft,peak \
  --json-out "$d/summary.json"; done
```

A pilot of a different, unpaired design ran on the same package before this
manifest. It was stopped and is not used here.

## Results

Package 1.8.1.1908, 2026-11-26, all 36 pairs complete. No arm needed a retry,
and every arm's app log matched its expected lines. Per cell, `<knob>/A.csv`
and `B.csv` hold the recorded rows, `ledger.jsonl` the arms, `pair*-app.log` the
app log of each arm, and `summary.json` the `--report` output. Arm A is the
shipped configuration: llama.cpp's auto flash attention (resolved to on,
`Flash Attention enabled` in every A log), `n_ubatch` 512, f16 KV.

Prefill is the median paired ratio B/A of `prompt_tok_s`, with its 95 %
bootstrap interval; peak is the median paired delta of `peak_ws_mb`.

| Cell       | Prompt tokens | A prefill (tok/s, s) | `fa0`                          | `kvq8`                         | `u256`                        | `u1024`                       |
| ---------- | ------------- | -------------------- | ------------------------------ | ------------------------------ | ----------------------------- | ----------------------------- |
| `lfm-8k`   | 7250          | 281, 25.8 s          | 0.573 [0.572, 0.574], +318 MiB | 0.364 [0.364, 0.365], −44 MiB  | 0.969 [0.968, 0.969], −21 MiB | 1.012 [1.012, 1.013], +43 MiB |
| `lfm-16k`  | 14428         | 220, 65.5 s          | 0.458 [0.457, 0.458], +658 MiB | 0.238 [0.236, 0.239], −93 MiB  | 0.950 [0.949, 0.951], −26 MiB | 1.029 [1.026, 1.031], +51 MiB |
| `llama-4k` | 3555          | 34.4, 103.3 s        | 0.694 [0.693, 0.695], +196 MiB | 0.488 [0.487, 0.489], −420 MiB | 0.973 [0.973, 0.977], −36 MiB | 1.005 [1.005, 1.006], +73 MiB |

TTFT ratios mirror the prefill ratios (`summary.json`).

## Reading

- **No knob changes the curve.** The shipped configuration is within 3 % of
  the fastest arm in every cell. `n_ubatch` 1024 gains 0.5–2.9 % for
  43–73 MiB more peak; 256 loses 2.7–5.0 %. The 512 default (#172) stays.
- **Flash attention is already on, and it matters more as the prompt grows.**
  Turning it off costs 31 % at 3.6k tokens on the 3B and 54 % at 14.4k on
  `lfm25-350m`, with more memory, not less.
- **q8_0 KV is a memory trade, not a speed one.** On this CPU it slows prefill
  2.0–4.2× (worse with length) for 44–420 MiB less peak (#171). It is only
  worth it where the KV, not time, is the limit.
- **So the outcome of #582 is a documented ceiling, not a tuned default.**
  Series S CPU prefill with the shipped knobs: `lfm25-350m` 281 tok/s at 7.2k
  tokens and 220 tok/s at 14.4k; `llama32-3b` 34 tok/s at 3.6k (22 tok/s at
  15.6k in `../2026-11-25-nctx-train-bound/`). A catalogue `n_ctx` raise for a
  3B dense model means minutes of prefill before the first token at 8k and
  beyond; a hybrid 350M model stays near one minute at 14k.
