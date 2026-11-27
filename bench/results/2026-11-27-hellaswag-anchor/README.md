# HellaSwag external anchor (#608)

Informational, like the [standard campaign](../2026-11-26-standard/README.md) it
anchors. H9 stays the only promotion gate.

## Why

`llama.cpp/tools/perplexity/README.md` at pin `d81235049` publishes no
HellaSwag figure, and there is no llama.cpp reference for the catalogue GGUFs.
The first campaign could only be checked for internal consistency. This run
checks the scorer itself. It compares `llama-perplexity` with an independent
implementation of the same rule on the same weights, item by item, first on
identical text, then on lm-evaluation-harness's text.

## Result

On identical inputs, `llama-perplexity` and the independent implementation
pick the same ending on 1995 of 2000 items (42.10 % against 42.25 %, McNemar
p = 0.37). The 0.9–1.05 points by which lm-eval's prompts score higher come
from the text, not from the scorer. The catalogue QAD Q4_0 quant scores like
its BF16 weights.

## Setup

- **Items.** The first 2000 tasks of the pinned HellaSwag validation file.
  - The spec is [`hellaswag-2000.json`](hellaswag-2000.json): `"lines": 12000`
    of `hellaswag_val_full.txt` (sha256 `d5725393…`), cut by `--fetch` and
    pinned as `413d24e6…`.
  - These are the first 2000 documents of the HF `Rowan/hellaswag`
    validation split, in the same order: gold labels and endings compared
    2000/2000.
  - The whole split would take about 6 h per arm on this host.
- **Weights.**
  - `LiquidAI/LFM2.5-350M` at revision `9921cd4d…`: BF16 safetensors.
  - `LiquidAI/LFM2.5-350M-GGUF` at revision `657e078c…`:
    `LFM2.5-350M-BF16.gguf` (sha256 `8da57615…`, as Hugging Face publishes it)
    and the catalogue's `LFM2.5-350M-QAD-Q4_0.gguf` (sha256 `3d10b6ab…`).
- **`llama-perplexity`.** Release build at the pin `d81235049` with the repo's
  patches: `scripts/eval-host-standard.py --spec hellaswag-2000.json`, 8
  threads, `n_ctx` 2048, `-np` 4. Rows:
  [`host-loglikelihood.jsonl`](host-loglikelihood.jsonl).
- **The independent implementation** is
  [`scripts/hellaswag_token_mean.py`](../../../scripts/hellaswag_token_mean.py):
  transformers 5.24.1, torch 2.15.0 (CPU, float32).
- **lm-evaluation-harness** 0.4.13: `--model hf --tasks hellaswag --limit
2000`, zero-shot, `--log_samples`. Summary:
  [`lm-eval-hellaswag.json`](lm-eval-hellaswag.json).
- **Host:** shared Linux machine (`linux-host-declared`). Times are not a
  performance figure.

## The rule

For each of the four choices, `llama-perplexity` tokenises context, a space
and the ending with BOS. It skips the token prefix all four share and picks
the choice with the highest **mean log-probability per token** over the rest
(`perplexity.cpp`, `hellaswag_score`; the same per-token mean scores
WinoGrande and multiple choice). `hellaswag_token_mean.py` applies that rule
with transformers.

The token-level boundary matters. In 180 of the 2000 items the endings share
leading tokens, and dividing each whole continuation by its token count
instead can flip the choice, as a unit test shows. lm-eval's own `acc`
(summed log-probability) and `acc_norm` (per character) are other rules.

## Arms (2000 tasks, accuracy %, ± binomial stderr)

| Arm | Scorer                     | Text                 | Weights              | Accuracy      |
| --- | -------------------------- | -------------------- | -------------------- | ------------- |
| A   | `llama-perplexity`         | klosax (llama.cpp's) | QAD Q4_0 (catalogue) | 41.80 ± 1.10  |
| B   | `llama-perplexity`         | klosax (llama.cpp's) | BF16 GGUF            | 42.10 ± 1.10  |
| D   | `hellaswag_token_mean.py`  | klosax (llama.cpp's) | BF16 safetensors     | 42.25 ± 1.10  |
| C   | `hellaswag_token_mean.py`  | lm-eval's prompts    | BF16 safetensors     | 43.15 ± 1.11  |
| —   | lm-eval `acc` / `acc_norm` | lm-eval's prompts    | BF16 safetensors     | 37.25 / 44.30 |

Per-item outcomes (`doc_id`, `correct`) are in `per-item-{A,B,C,D}.jsonl`.

## Paired comparison

All arms score the same 2000 items, so they are compared item by item. The
test is McNemar's, with continuity correction, on the discordant items (one
arm right, the other wrong). The paired stderr of the difference is
√(b + c − (b − c)²/n) / n.

| Pair  | What differs            | Discordant (first only / second only) | Difference | Paired stderr | p     |
| ----- | ----------------------- | ------------------------------------- | ---------- | ------------- | ----- |
| B – D | implementation          | 5 (1 / 4)                             | −0.15      | 0.11          | 0.371 |
| D – C | text                    | 68 (25 / 43)                          | −0.90      | 0.41          | 0.039 |
| B – C | implementation and text | 69 (24 / 45)                          | −1.05      | 0.41          | 0.016 |
| A – B | quantization            | 168 (81 / 87)                         | −0.30      | 0.65          | 0.700 |

## Reading

- **The anchor (B against D).** Same text, same rule, same weights, two
  implementations. They agree on 1995 of 2000 items, and the 0.15-point
  difference is not significant. That is the external reference #608 asked
  for: `llama-perplexity`'s HellaSwag score is the score an independent
  implementation of its rule gives.
- **The text (D against C).** Same implementation, same rule. lm-eval's
  prompts score 0.9 points higher than llama.cpp's klosax text, a small but
  significant difference over 68 items. The two texts differ visibly, for
  example "a roof. he" in the klosax file against "a roof. He" after lm-eval's
  `preprocess`. This measurement shows that the text accounts for the
  difference. It does not say which edit does.
- **Quantization (A against B).** QAD Q4_0 against BF16: −0.30 points over 168
  discordant items, p = 0.70. The catalogue quant costs nothing measurable
  here.
- **Metric choice.** lm-eval's `acc` and `acc_norm` bracket the per-token mean
  (37.25 < 43.15 < 44.30 on lm-eval's text). So a model-card figure that
  prints `acc_norm` sits a few points above what `llama-perplexity` reports
  for the same model, and part of the difference is the text.

## How the per-item outcomes were obtained

- **C and D:** `hellaswag_token_mean.py --per-item`.
- **A and B.** `llama-perplexity` prints only a running accuracy per task, in
  the order it draws them. It draws without replacement, with `std::mt19937
rng(1)` and `std::uniform_int_distribution<size_t>(0, remaining − 1)`
  over the tasks still in the file, then erases the drawn task. Replaying
  that draw with the same libstdc++ gives each row's task. The difference
  between consecutive running counts gives its outcome:

```cpp
// g++ -O2 draw.cpp && ./a.out 2000 2000 > order.txt  (one file index per row)
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
int main(int argc, char** argv) {
    const size_t n = std::strtoull(argv[1], nullptr, 10), k = std::strtoull(argv[2], nullptr, 10);
    std::vector<size_t> left(n);
    for (size_t i = 0; i < n; i++) left[i] = i;
    std::mt19937 rng(1);
    for (size_t i = 0; i < k; i++) {
        std::uniform_int_distribution<size_t> dist(0, left.size() - 1);
        const size_t idx = dist(rng);
        std::printf("%zu\n", left[idx]);
        left.erase(left.begin() + idx);
    }
}
```

The replay checks out two ways:

- The outcomes sum to the published accuracies (836 and 842 of 2000).
- Aligned this way, they agree with lm-eval's per-item `acc_norm` on 83–85 %
  of items, against 51.5 % when the alignment is shifted by seven: chance at
  these accuracies.

## Reproducing

```bash
scripts/eval-host-standard.py --fetch --spec bench/results/2026-11-27-hellaswag-anchor/hellaswag-2000.json
scripts/eval-host-standard.py --model <gguf> --hardware <label> --tasks hellaswag \
  --spec bench/results/2026-11-27-hellaswag-anchor/hellaswag-2000.json --out arms.jsonl --log-dir logs
lm_eval --model hf --model_args pretrained=<LFM2.5-350M at 9921cd4d>,dtype=float32 \
  --tasks hellaswag --limit 2000 --device cpu --batch_size 16 --log_samples --output_path out
scripts/hellaswag_token_mean.py --samples out/<model>/samples_hellaswag_*.jsonl \
  --model <LFM2.5-350M at 9921cd4d> --per-item per-item-C.jsonl
scripts/hellaswag_token_mean.py --klosax ~/.cache/xllama/eval-datasets/hellaswag_val_first2000.txt \
  --model <LFM2.5-350M at 9921cd4d> --per-item per-item-D.jsonl
```
