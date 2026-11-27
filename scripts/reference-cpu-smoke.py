#!/usr/bin/env python3
# Copyright (c) 2024 Gianluca Mazza
# SPDX-License-Identifier: MIT
"""Offline oracle regeneration. Recompute each causal prefix in float64 without ggml."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "llama.cpp" / "gguf-py"))
from gguf import GGUFReader


def generate(model):
    weights = {t.name: t.data.astype(np.float64) for t in GGUFReader(model).tensors}

    def norm(x, name):
        return x * weights[name + ".weight"] / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-5)

    def linear(x, name):
        return x @ weights[name + ".weight"].T

    def rope(x):
        angles = np.outer(np.arange(len(x)), 10000 ** (-np.arange(0, 32, 2) / 32))
        a, b = x[:, ::2], x[:, 1::2]
        out = np.empty_like(x)
        out[:, ::2] = a * np.cos(angles) - b * np.sin(angles)
        out[:, 1::2] = a * np.sin(angles) + b * np.cos(angles)
        return out

    def logits(ids):
        x = weights["token_embd.weight"][ids]
        h = norm(x, "blk.0.attn_norm")
        q = rope(linear(h, "blk.0.attn_q"))
        k = rope(linear(h, "blk.0.attn_k"))
        v = linear(h, "blk.0.attn_v")
        # Only the final query is needed for a one-layer causal transformer.
        scores = k @ q[-1] / np.sqrt(32)
        probs = np.exp(scores - scores.max())
        x = x[-1] + linear(probs @ v / probs.sum(), "blk.0.attn_output")
        h = norm(x, "blk.0.ffn_norm")
        g = linear(h, "blk.0.ffn_gate")
        x += linear(g / (1 + np.exp(-g)) * linear(h, "blk.0.ffn_up"), "blk.0.ffn_down")
        return linear(norm(x, "output_norm"), "output")

    def greedy(ids, count):
        ids = list(ids)
        rows = [logits(ids)]
        tokens = []
        for _ in range(count):
            token = int(np.argmax(rows[-1]))
            tokens.append(token)
            ids.append(token)
            rows.append(logits(ids))
        return tokens, rows

    prompt = [1, 3, 4, 5, 6]  # BOS + "abcd", no prefix space
    tokens, rows = greedy(prompt, 4)
    delta = [15, 8]  # "mf", no BOS on continuation
    continuation, _ = greedy(prompt + tokens + delta, 4)
    reset, _ = greedy([1] + delta, 4)
    assert continuation != reset, "delta must discriminate retained and reset KV"
    return {
        "prompt_ids": prompt,
        "generated_ids": tokens,
        "continuation_ids": continuation,
        "reset_ids": reset,
        "positions": list(range(len(prompt) - 1, len(prompt) + 4)),
        # Discard sub-picounit BLAS variation in the human-reviewable JSON.
        "rows": [row.round(10).tolist() for row in rows],
    }


if __name__ == "__main__":
    print(json.dumps(generate(sys.argv[1]), indent=2))
