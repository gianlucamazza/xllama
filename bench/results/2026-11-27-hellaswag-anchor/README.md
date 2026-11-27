# HellaSwag external anchor (#608)

Informational, like the [standard campaign](../2026-11-26-standard/README.md) it
anchors. H9 stays the only promotion gate.

## Why

`llama.cpp/tools/perplexity/README.md` at pin `d81235049` publishes no
HellaSwag figure, and there is no llama.cpp reference for the catalogue GGUFs.
The first campaign could only be checked for internal consistency. This run
checks the scorer itself: the same weights and the same items, scored by
`llama-perplexity` and by an independent implementation
(lm-evaluation-harness on transformers), with the same metric.

## Setup

- **Items.** The first 2000 tasks of the pinned HellaSwag validation file:
  - `head -n 12000` of `hellaswag_val_full.txt`, sha256 `d5725393…`. That
    gives `hellaswag_val_first2000.txt`, sha256 `413d24e6…`, pinned in
    [`hellaswag-2000.json`](hellaswag-2000.json).
  - These are the first 2000 documents of the HF `Rowan/hellaswag`
    validation split, in the same order: gold labels and endings compared
    2000/2000.
  - So `llama-perplexity`, which scores every task of a file that holds
    exactly the tasks asked for, and `lm_eval --limit 2000` score the same
    items.
  - The whole split (10042 tasks) would take about 6 h per arm on this host
    for the same conclusion.
- **Weights.**
  - `LiquidAI/LFM2.5-350M` at revision `9921cd4d…`: BF16 safetensors for
    lm-eval.
  - `LiquidAI/LFM2.5-350M-GGUF` at revision `657e078c…`:
    `LFM2.5-350M-BF16.gguf` (sha256 `8da57615…`, as Hugging Face publishes it)
    and the catalogue's `LFM2.5-350M-QAD-Q4_0.gguf` (sha256 `3d10b6ab…`).
- **`llama-perplexity`.** Release build at the pin `d81235049` with the repo's
  patches, `scripts/eval-host-standard.py --spec hellaswag-2000.json`, 8
  threads, `n_ctx` 2048, `-np` 4. Rows:
  [`host-loglikelihood.jsonl`](host-loglikelihood.jsonl).
- **lm-evaluation-harness** 0.4.13, transformers 5.24.1, torch 2.15.0 (CPU),
  `--model hf --model_args dtype=float32 --tasks hellaswag --limit 2000`,
  zero-shot, `--log_samples`. Summary:
  [`lm-eval-hellaswag.json`](lm-eval-hellaswag.json).
- **Host:** shared Linux machine (`linux-host-declared`). Times are not a
  performance figure.

## Metric

`llama-perplexity` tokenises context plus ending for each of the four choices,
skips the token prefix all four share, and picks the choice with the highest
**mean log-probability per token** over the rest (`perplexity.cpp`,
`hellaswag_score`). lm-eval prints two other metrics: `acc` (summed
log-probability of the continuation) and `acc_norm` (per character).

The like-for-like number for the second implementation is
[`scripts/hellaswag_token_mean.py`](../../../scripts/hellaswag_token_mean.py). It applies
`llama-perplexity`'s rule with transformers on the same weights, to the items
the lm-eval run scored: lm-eval's context and continuations, from
`--log_samples`. It needs per-token log-probabilities, because in 180 of the
2000 items the endings share leading tokens. There, dividing each whole
continuation's log-probability by its token count is not the same rule, and
the unit test shows that it can flip the choice.

```bash
lm_eval --model hf --model_args pretrained=<LFM2.5-350M at 9921cd4d>,dtype=float32 \
  --tasks hellaswag --limit 2000 --device cpu --batch_size 16 --log_samples --output_path out
scripts/hellaswag_token_mean.py --samples out/<model>/samples_hellaswag_*.jsonl \
  --model <LFM2.5-350M at 9921cd4d> --threads 8
```

## Results (2000 tasks, accuracy %, ± binomial stderr)

| Arm | Scorer                                                    | Weights              | Token-mean     | `acc`      | `acc_norm` |
| --- | --------------------------------------------------------- | -------------------- | -------------- | ---------- | ---------- |
| A   | `llama-perplexity`                                        | QAD Q4_0 (catalogue) | **41.8 ± 1.1** |            |            |
| B   | `llama-perplexity`                                        | BF16 GGUF            | **42.1 ± 1.1** |            |            |
| C   | transformers, `llama-perplexity`'s rule (lm-eval's items) | BF16 safetensors     | **43.2 ± 1.1** | 37.3 ± 1.1 | 44.3 ± 1.1 |

Row C's `acc` and `acc_norm` are lm-eval's own figures for the same run; its
token-mean is `scripts/hellaswag_token_mean.py` on that run's items.

## Reading

- **The anchor (B against C).** On the same weights, items and metric, the two
  implementations differ by 1.05 points (42.10 against 43.15). That is 0.67
  of their combined stderr (1.56), so `llama-perplexity`'s HellaSwag agrees with an independent
  scorer within stderr, which is what #608 asked for. The gap that remains is in
  what the two read, not in how they score. lm-eval's `preprocess` of the
  WikiHow and ActivityNet markup produces slightly different text from the
  klosax file, and the GGUF and Hugging Face tokenizers need not split it the
  same way.
- **Quantization (A against B).** QAD Q4_0 scores 0.3 points below BF16,
  well inside the stderr: the catalogue quant costs nothing measurable here.
- **Metric choice.** `acc` and `acc_norm` bracket the token-mean (37.3 < 43.2
  < 44.3), so a figure from a model card that prints `acc_norm` is a few points
  above what `llama-perplexity` reports for the same model. The standard
  campaign's HellaSwag numbers compare across xllama runs at this pin, as its
  README says, and now with a known relation to lm-eval's metrics.
