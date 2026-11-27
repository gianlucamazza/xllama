#!/usr/bin/env python3
# Copyright (c) 2024 Gianluca Mazza
# SPDX-License-Identifier: MIT
"""Materialize the pinned, first-party CPU CI model, checking bytes on cache hits too."""

import argparse
import hashlib
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "llama.cpp" / "gguf-py"))
from gguf import GGUFWriter

SHA256 = "ea5812a29a3260b9d489fadb272e1dddedadaf291d5fcf839b944ee440311881"
TOKENS = ["<unk>", "<s>", "</s>", *"abcdefghijklmnop"]


def write_model(path):
    writer = GGUFWriter(path, "llama")
    writer.add_name("xllama CPU smoke (synthetic weights)")
    writer.add_context_length(128)
    writer.add_embedding_length(32)
    writer.add_block_count(1)
    writer.add_feed_forward_length(64)
    writer.add_head_count(1)
    writer.add_rope_dimension_count(32)
    writer.add_layer_norm_rms_eps(1e-5)
    writer.add_tokenizer_model("llama")
    writer.add_token_list(TOKENS)
    writer.add_token_scores([0.0] * len(TOKENS))
    writer.add_token_types([2, 3, 3] + [1] * (len(TOKENS) - 3))
    writer.add_unk_token_id(0)
    writer.add_bos_token_id(1)
    writer.add_eos_token_id(2)
    writer.add_add_bos_token(True)
    writer.add_add_space_prefix(False)
    rng = np.random.Generator(np.random.PCG64(431))
    shapes = {
        "token_embd": (len(TOKENS), 32),
        "blk.0.attn_norm": (32,),
        "blk.0.attn_q": (32, 32),
        "blk.0.attn_k": (32, 32),
        "blk.0.attn_v": (32, 32),
        "blk.0.attn_output": (32, 32),
        "blk.0.ffn_norm": (32,),
        "blk.0.ffn_gate": (64, 32),
        "blk.0.ffn_up": (64, 32),
        "blk.0.ffn_down": (32, 64),
        "output_norm": (32,),
        "output": (len(TOKENS), 32),
    }
    for name, shape in shapes.items():
        weight = rng.integers(-16, 17, shape).astype(np.float32) / 64
        if len(shape) == 1:
            weight += 1
        if name == "output":
            weight[:3] = 0  # Keep special tokens out of the greedy transcript.
        writer.add_tensor(name + ".weight", weight)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def verify(path):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SHA256:
        raise ValueError(f"CPU fixture {path}: expected sha256 {SHA256}, got {digest}")
    print(f"CPU fixture {path} sha256={digest}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    args = parser.parse_args()
    path = args.model
    if path.exists():
        verify(path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent) as tmp:
        staged = Path(tmp) / "model.gguf"
        write_model(staged)
        verify(staged)
        staged.replace(path)


if __name__ == "__main__":
    main()
