#!/usr/bin/env python3
"""llama-perplexity's HellaSwag rule on a Hugging Face model (#608).

`llama-perplexity` (perplexity.cpp, hellaswag_score) tokenises context plus
ending for each of the four choices with BOS, skips the token prefix all four
share, and picks the choice with the highest mean log-probability over the
rest. This applies the same rule with transformers to the items an
lm-evaluation-harness `--log_samples` run scored, so the two implementations
can be compared on the same weights, items and metric.

  hellaswag_token_mean.py --samples samples_hellaswag_*.jsonl --model <hf dir>

Prints one JSON object: acc and its binomial stderr in percent, the item
count, and how many items had endings that share leading tokens (where
dividing a whole continuation's log-probability would differ). The pure part
(shared prefix, mean, choice) needs neither torch nor transformers.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def shared_prefix(sequences: list[list[int]]) -> int:
    """Tokens all sequences share at the start, leaving each at least one."""
    limit = min(len(s) for s in sequences) - 1
    n = 0
    while n < limit and all(s[n] == sequences[0][n] for s in sequences):
        n += 1
    return n


def choose(token_logprobs: list[list[float]], start: int) -> int:
    """Index of the choice with the highest mean log-probability past |start|.

    token_logprobs[c][i] is the log-probability of token i of choice c given
    the tokens before it (entry 0, the first token, is never scored).
    """
    start = max(start, 1)  # token 0 has no prediction to score
    means = [sum(lp[start:]) / len(lp[start:]) for lp in token_logprobs]
    return max(range(len(means)), key=means.__getitem__)


def binomial(hits: int, n: int) -> tuple[float, float]:
    p = hits / n
    return 100 * p, 100 * math.sqrt(p * (1 - p) / (n - 1))


def items(samples: Path):
    """(context, texts, gold) per sample: texts are the context plus each
    continuation, as lm-eval scored them."""
    for line in samples.read_text().splitlines():
        r = json.loads(line)
        args = [r["arguments"][f"gen_args_{i}"] for i in range(len(r["arguments"]))]
        yield args[0]["arg_0"], [a["arg_0"] + a["arg_1"] for a in args], int(r["target"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--samples", type=Path, required=True)
    ap.add_argument("--model", required=True, help="Hugging Face model directory or id")
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if a.threads:
        torch.set_num_threads(a.threads)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).eval()

    hits = n = shared_items = 0
    with torch.inference_mode():
        for context, texts, gold in items(a.samples):
            seqs = [tok(t, add_special_tokens=True)["input_ids"] for t in texts]
            start = shared_prefix(seqs)
            # The endings themselves share leading tokens: here dividing each
            # whole continuation would not be llama-perplexity's rule.
            shared_items += start > len(tok(context, add_special_tokens=True)["input_ids"])
            logprobs = []
            for s in seqs:
                ids = torch.tensor([s])
                out = torch.log_softmax(model(ids).logits[0].float(), dim=-1)
                # Token i is predicted by the logits at i - 1; token 0 has none.
                lp = [0.0] + [out[i - 1, s[i]].item() for i in range(1, len(s))]
                logprobs.append(lp)
            hits += choose(logprobs, start) == gold
            n += 1
    acc, stderr = binomial(hits, n)
    print(json.dumps({"acc": round(acc, 2), "stderr": round(stderr, 2), "n": n,
                      "rule": "llama-perplexity hellaswag_score", "shared_items": shared_items}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
