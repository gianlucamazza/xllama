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
