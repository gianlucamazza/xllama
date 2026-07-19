#!/usr/bin/env python3
"""Host-side ORT-GenAI prefill logit dump (#91 plan B, phase 2).

Runs a GenAI model directory through the onnxruntime_genai Python runtime
(same runtime family as the device bridge, CPU EP via the model's
genai_config.json), prefills the golden prompt and dumps the last-token
logits as float32 in the scripts/compare-logits.py format (.bin + .json
sidecar).

The Linux xllama-cli is llama.cpp-only (XLLAMA_USE_ORT is defined only in
the UWP project), so this driver is the host CPU-EP parity path documented
in docs/model-selection.md step 2 for machines without the Windows VM.

Usage:
    ort-prefill-logits.py -m <model_dir> -o <out.bin>
        [--golden tests/golden/logits-smol-short.bin]  # prompt source
        [--prompt "..."]                               # explicit override
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime_genai as og


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-m", "--model-dir", required=True, type=Path)
    ap.add_argument("-o", "--output", required=True, type=Path)
    ap.add_argument(
        "--golden",
        type=Path,
        default=Path("tests/golden/logits-smol-short.bin"),
        help="golden dump whose .json sidecar supplies the prompt",
    )
    ap.add_argument("--prompt", help="explicit prompt (overrides --golden sidecar)")
    args = ap.parse_args()

    if args.prompt is not None:
        prompt = args.prompt
    else:
        side = Path(str(args.golden) + ".json")
        if not side.exists():
            print(f"error: golden sidecar not found: {side}", file=sys.stderr)
            return 2
        prompt = json.loads(side.read_text())["prompt"]

    model = og.Model(str(args.model_dir))
    tokenizer = og.Tokenizer(model)
    ids = tokenizer.encode(prompt)

    params = og.GeneratorParams(model)
    params.set_search_options(max_length=len(ids) + 1, do_sample=False)
    generator = og.Generator(model, params)
    generator.append_tokens(ids)

    logits = np.asarray(generator.get_output("logits"))
    last = logits.reshape(-1, logits.shape[-1])[-1].astype(np.float32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    last.tofile(args.output)
    top1 = int(last.argmax())
    sidecar = {
        "model": str(args.model_dir),
        "prompt": prompt,
        "backend": "ort-genai-py",
        "vocab_size": int(last.shape[0]),
        "greedy": True,
        "top1_id": top1,
        "top1_piece": tokenizer.decode(np.array([top1], dtype=np.int32)),
        "input_ids": [int(i) for i in ids],
    }
    Path(str(args.output) + ".json").write_text(json.dumps(sidecar, indent=2) + "\n")
    print(
        f"dumped {last.shape[0]} logits for {len(ids)} prompt tokens -> "
        f"{args.output} (top1_id={top1}, top1_piece={sidecar['top1_piece']!r})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
