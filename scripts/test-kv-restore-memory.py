#!/usr/bin/env python3
# Copyright (c) 2024 Gianluca Mazza
# SPDX-License-Identifier: MIT
"""Exercise hybrid state restore with the existing LFM2 shape writer or a supplied model."""

import argparse
from pathlib import Path
import runpy
import subprocess
import tempfile


def write_hybrid(path):
    directory = Path(__file__).resolve().parent
    shapes = runpy.run_path(str(directory / "make-shape-gguf.py"))
    tokens = runpy.run_path(str(directory / "cpu-smoke-fixture.py"))["TOKENS"]
    from gguf import GGUFValueType as G

    tokenizer = [
        ("tokenizer.ggml.model", "llama", G.STRING, None),
        ("tokenizer.ggml.add_space_prefix", False, G.BOOL, None),
        ("tokenizer.ggml.scores", [0.0] * len(tokens), G.ARRAY, G.FLOAT32),
    ]
    template = (tokenizer, tokens, [2, 3, 3] + [1] * (len(tokens) - 3), [], 10000.0, 1e-5)
    shape = shapes["Shape"]("kv-restore", 2, 64, 128, len(tokens), 1, [1])
    shapes["write_f16_gguf"](shape, template, path, 431)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, help="Optional local hybrid GGUF")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        model = args.model or Path(tmp) / "fixture.gguf"
        if args.model is None:
            write_hybrid(model)
        cmd = [args.build_dir / "tests/test-state-restore-memory", model, Path(tmp) / "snapshot.bin"]
        if args.model:
            cmd.append("--chat")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
