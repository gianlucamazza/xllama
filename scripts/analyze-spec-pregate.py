#!/usr/bin/env python3
"""Turn the H3 pre-gate measurements into a console prediction.

Usage:
    ./scripts/analyze-spec-pregate.py [bench/results/phase15-spec-pregate.csv]

Why this exists as code and not as arithmetic in a doc: the prediction quoted in
docs/phase7-hypotheses.md under H3 is derived, not measured, and a derived number
nobody can re-run is a number nobody can check. The measured inputs are the
acceptance rate and the drafting frequency (host, hardware-independent); the cost
model comes from the console figures in docs/model-matrix.md.

Cost model: a forward pass over n tokens costs T(n) = W + n*C, with W the weight
read paid once and C the marginal compute per token. Fitting the two published
console figures for a model (prefill = 1/C at large batch, decode = 1/(W+C) at
n=1) pins both. llama-batched-bench confirms T(n) is linear over n=1..8, i.e.
there is no bandwidth-bound plateau at small n that would make verification
cheaper than this model says.

The frequency-aware part matters: a variant that declines to draft pays nothing
for the rounds it skips. Computing this per-round instead — as if drafting always
happened — understates prompt-lookup precisely where its safety comes from.
"""

import csv
import sys
from pathlib import Path

# Console-measured, docs/model-matrix.md §A2 (MSIX 1.5.1.737, t6).
TARGETS = {
    "qwen25-coder-3b": {"prefill": 46.2, "decode": 14.0},
    "lfm25-1.2b-thinking": {"prefill": 130.4, "decode": 36.7},
}
# Draft cost per drafted token = that model's own decode step.
DRAFTS = {
    "qwen25-coder-0.5b": 62.4,
    "lfm25-350m": 94.9,
}
N_PREDICT = 128  # the -n the pre-gate ran at


def cost_model(prefill_tok_s: float, decode_tok_s: float) -> tuple[float, float]:
    c = 1000.0 / prefill_tok_s
    w = 1000.0 / decode_tok_s - c
    return w, c


def predict(alpha, k, draft_ms, n_drafted, w, c, n_predict=N_PREDICT):
    base = w + c
    rounds = n_drafted / k if k else 0.0
    expected = sum(alpha**i for i in range(k + 1))  # accepted + 1 per drafting round
    from_draft = rounds * expected
    plain = max(0.0, n_predict - from_draft)
    cost = rounds * (k * draft_ms + w + (k + 1) * c) + plain * base
    total = from_draft + plain
    if total <= 0:
        return None
    per_token = cost / total
    frequency = rounds / (rounds + plain) if (rounds + plain) else 0.0
    return expected, frequency, per_token, base / per_token


def main() -> int:
    path = Path(
        sys.argv[1] if len(sys.argv) > 1 else "bench/results/phase15-spec-pregate.csv"
    )
    if not path.exists():
        print(
            f"missing {path} — run scripts/bench-spec-pregate.sh first", file=sys.stderr
        )
        return 1

    print(f"cost model, from docs/model-matrix.md (console):")
    for name, f in TARGETS.items():
        w, c = cost_model(f["prefill"], f["decode"])
        print(
            f"  {name:<22} W={w:6.1f} ms  C={c:5.1f} ms  "
            f"baseline={w + c:6.1f} ms/tok  compute share={c / (w + c) * 100:4.1f}%"
        )
    print()
    hdr = f"{'variant':<12}{'target':<22}{'regime':<6}{'k':>2}{'drafted':>9}{'alpha':>8}{'draft%':>8}{'ms/tok':>9}{'speedup':>9}"
    print(hdr)
    print("-" * len(hdr))

    exit_code = 0
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            tgt = row["target"]
            if tgt not in TARGETS:
                print(f"  (no console cost model for {tgt} — skipped)", file=sys.stderr)
                exit_code = 1
                continue
            w, c = cost_model(TARGETS[tgt]["prefill"], TARGETS[tgt]["decode"])
            draft_ms = 0.0 if row["kind"] == "ngram" else 1000.0 / DRAFTS[row["draft"]]
            if not row["n_drafted"] or not row["accept_pct"]:
                print(
                    f"  ({row['kind']}/{row['regime']}: no data in row — skipped)",
                    file=sys.stderr,
                )
                exit_code = 1
                continue
            k = int(row["n_draft_max"])
            alpha = float(row["accept_pct"]) / 100.0
            n_drafted = int(row["n_drafted"])
            out = predict(alpha, k, draft_ms, n_drafted, w, c)
            if out is None:
                continue
            _, freq, per_token, speedup = out
            # Three-way, because "neutral" is the verdict that decides this gate:
            # a variant that cannot hurt is shippable at a gain no single number
            # would justify, and one that hurts is not shippable at any gain.
            if speedup >= 1.4:
                flag = "  <- clears the 1.4x gate"
            elif speedup >= 0.98:
                flag = "  <- neutral: costs nothing when it does not help"
            else:
                flag = f"  <- SLOWER than no speculation ({(1 - speedup) * 100:.0f}% worse)"
            print(
                f"{row['kind']:<12}{tgt:<22}{row['regime']:<6}{k:>2}{n_drafted:>9}"
                f"{alpha * 100:>7.1f}%{freq * 100:>7.0f}%{per_token:>9.1f}{speedup:>8.2f}x{flag}"
            )

    print()
    for label, tok_s in (
        ("with a draft model (coder-0.5b)", 62.4),
        ("draft-free (n-gram)", None),
    ):
        w, c = cost_model(
            TARGETS["qwen25-coder-3b"]["prefill"], TARGETS["qwen25-coder-3b"]["decode"]
        )
        marginal = c + (1000.0 / tok_s if tok_s else 0.0)
        print(
            f"ceiling on coder-3b, {label}: {(w + c) / marginal:.2f}x  (alpha=1, k->inf)"
        )
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
