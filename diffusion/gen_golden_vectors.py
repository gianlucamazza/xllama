#!/usr/bin/env python3
# Copyright (c) 2024 Venere Labs
# SPDX-License-Identifier: MIT
#
# Generate golden vectors that pin the correctness-critical logic the C++
# diffusion pipeline (uwp/diffuse.cpp) must reproduce bit-for-bit:
#   1. CLIP tokenization (byte-level BPE) → token IDs, for a few prompts.
#   2. EulerDiscreteScheduler setup (sigmas, timesteps, init_noise_sigma) for the
#      SD-Turbo 1-step schedule, plus one deterministic scheduler step on a fixed
#      fake UNet output — testing the pure scheduler math without the real UNet.
#
# Run against the SAME reference that validated the pipeline (diffusers 0.31.0 in
# the pinned venv). The output JSON is committed as the C++ test fixture; the host
# unit test (uwp/diffuse_test.cpp) loads it and asserts the C++ tokenizer +
# scheduler match. This is the anti-theater gate: the correctness-critical logic is
# verified against the reference before it ships in un-runtime-testable console C++.
#
#   ~/.cache/xllama-diffusion/venv310/bin/python diffusion/gen_golden_vectors.py \
#       [sd_onnx_dir] [out.json]
import sys, json
import numpy as np
from transformers import CLIPTokenizer
from diffusers import EulerDiscreteScheduler

SD_DIR = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "/home/gianluca/.cache/xllama-diffusion/sd-turbo-onnx"
)
OUT = sys.argv[2] if len(sys.argv) > 2 else "diffusion/golden_vectors.json"

MAX_LEN = 77
PROMPTS = [
    "a red sports car on a mountain road at sunset",
    "a photo of an astronaut riding a horse",
    "",  # empty prompt → bos + eos + all pad, the unconditional path
    "Café déjà-vu 42%",  # non-ASCII + digits + punctuation, exercises byte-BPE
]

# ---- 1. CLIP tokenization --------------------------------------------------
tok = CLIPTokenizer.from_pretrained(f"{SD_DIR}/tokenizer")
tokens = []
for p in PROMPTS:
    ids = (
        tok(
            p,
            padding="max_length",
            max_length=MAX_LEN,
            truncation=True,
            return_tensors="np",
        )["input_ids"][0]
        .astype(int)
        .tolist()
    )
    tokens.append({"prompt": p, "input_ids": ids})
    print(f"[tok] {p[:32]!r:34} -> {ids[:8]}... (len {len(ids)})")

assert tok.bos_token_id == 49406 and tok.eos_token_id == 49407
pad_id = tok.convert_tokens_to_ids(tok.pad_token)  # "!" == 0

# ---- 2. EulerDiscreteScheduler (SD-Turbo, 1 step) --------------------------
sched = EulerDiscreteScheduler.from_pretrained(f"{SD_DIR}/scheduler")
sched.set_timesteps(1)  # SD-Turbo: single inference step
sigmas = np.asarray(sched.sigmas, dtype=np.float64).tolist()
timesteps = np.asarray(sched.timesteps, dtype=np.float64).tolist()
init_noise_sigma = float(sched.init_noise_sigma)
print(
    f"[sched] sigmas={sigmas} timesteps={timesteps} init_noise_sigma={init_noise_sigma}"
)

# One deterministic step on a FIXED fake UNet output. Small 2x2x2x2 tensor so the
# golden values are human-checkable; the math is channel/pixel-independent so the
# shape is irrelevant to correctness.
rng = np.random.RandomState(1234)
shape = (1, 4, 8, 8)
sample0 = rng.randn(*shape).astype(np.float32) * init_noise_sigma  # init latent
fake_eps = rng.randn(*shape).astype(np.float32)  # fake UNet noise

t = sched.timesteps[0]
scaled = sched.scale_model_input(
    __import__("torch").tensor(sample0), t
).numpy()  # sample / sqrt(sigma^2+1)
prev = sched.step(
    __import__("torch").tensor(fake_eps), t, __import__("torch").tensor(sample0)
).prev_sample.numpy()

golden = {
    "_comment": "Golden vectors for the C++ diffusion pipeline. See gen_golden_vectors.py.",
    "reference": {
        "diffusers": "0.31.0",
        "transformers": "4.46.3",
        "model": "stabilityai/sd-turbo",
    },
    "tokenizer": {
        "class": "CLIPTokenizer",
        "max_length": MAX_LEN,
        "bos_id": int(tok.bos_token_id),
        "eos_id": int(tok.eos_token_id),
        "pad_id": int(pad_id),
        "do_lower_case": True,
        "cases": tokens,
    },
    "scheduler": {
        "class": "EulerDiscreteScheduler",
        "prediction_type": "epsilon",
        "num_inference_steps": 1,
        "sigmas": sigmas,
        "timesteps": timesteps,
        "init_noise_sigma": init_noise_sigma,
        # For final_sigmas_type=zero + 1 step, step() reduces to x0 = sample - sigma*eps.
        "step_case": {
            "shape": list(shape),
            "sample": sample0.astype(float).ravel().tolist(),
            "model_output": fake_eps.astype(float).ravel().tolist(),
            "scaled_model_input": scaled.astype(float).ravel().tolist(),
            "prev_sample": prev.astype(float).ravel().tolist(),
        },
    },
    "vae": {
        "scaling_factor": 0.18215,
        "latent_channels": 4,
        "latent_hw": 64,
        "image_hw": 512,
    },
    "text_encoder": {"hidden_size": 1024, "seq_len": MAX_LEN},
}

with open(OUT, "w") as f:
    json.dump(golden, f, indent=2)
print(f"[golden] wrote {OUT} ({len(PROMPTS)} token cases, 1 scheduler step case)")
