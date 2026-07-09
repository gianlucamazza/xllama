#!/usr/bin/env python3
# Copyright (c) 2024 Venere Labs
# SPDX-License-Identifier: MIT
#
# Model-side validation for the console diffusion pipeline: replicate uwp/diffuse.cpp's
# exact orchestration (CLIP tokenize -> text_encoder -> Euler scale/step -> UNet ->
# VAE decode -> PNG) against a *deployable* ONNX model directory, through ONNX
# Runtime (CPU EP) — the same runtime that runs on Xbox via the DirectML EP. This
# proves the artifact + the pipeline recipe on the layer we own before any console
# run. Dtypes are fed adaptively from the session metadata, exactly like diffuse.cpp
# (covers the ORT-team fp16 export where timestep is float16).
#
#   ~/.cache/xllama-diffusion/venv310/bin/python diffusion/validate_pipeline.py \
#       [model_dir] [out.png] ["prompt"] [steps] [seed] [vae_dir]
#
# vae_dir (optional): directory holding an alternative vae_decoder/model.onnx
# (e.g. a TAESD export) — validates a decoder swap against the same contract.
import sys, time
import numpy as np
import onnxruntime as ort
from transformers import CLIPTokenizer
from diffusers import EulerDiscreteScheduler

model_dir = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "/home/gianluca/.cache/xllama-diffusion/sd-turbo-ort-fp16"
)
out_png = sys.argv[2] if len(sys.argv) > 2 else "sd_turbo_pipeline.png"
prompt = (
    sys.argv[3]
    if len(sys.argv) > 3
    else "a red sports car on a mountain road at sunset"
)
steps = int(sys.argv[4]) if len(sys.argv) > 4 else 1
seed = int(sys.argv[5]) if len(sys.argv) > 5 else 42
vae_dir = sys.argv[6] if len(sys.argv) > 6 else model_dir

VAE_SCALE = 0.18215
LATENT_HW = 64


def sess(comp, root=None):
    # EXTENDED, not ALL: ORT_ENABLE_ALL's layout transforms crash session init on
    # the fp16 SD graphs (graph_utils GetIndexFromName) — same cap as diffuse.cpp,
    # so this validates the exact configuration the console runs.
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    return ort.InferenceSession(
        f"{root or model_dir}/{comp}/model.onnx",
        sess_options=so,
        providers=["CPUExecutionProvider"],
    )


def in_type(s, name):
    for i in s.get_inputs():
        if i.name == name:
            return i.type  # e.g. 'tensor(float16)'
    return None


def feed(s, name, arr_f32):
    t = in_type(s, name)
    if t == "tensor(float16)":
        return arr_f32.astype(np.float16)
    return arr_f32.astype(np.float32)


t0 = time.time()
# ---- Tokenize (reference CLIPTokenizer — the C++ tokenizer matches it via golden)
tok = CLIPTokenizer.from_pretrained(f"{model_dir}/tokenizer")
ids = tok(
    prompt, padding="max_length", max_length=77, truncation=True, return_tensors="np"
)["input_ids"]

# ---- Text encoder
te = sess("text_encoder")
ids_t = in_type(te, "input_ids")
ids = ids.astype(np.int64 if ids_t == "tensor(int64)" else np.int32)
hidden = te.run(["last_hidden_state"], {"input_ids": ids})[0].astype(np.float32)
print(f"[val] text_encoder ok: hidden {hidden.shape} ({ids_t})")

# ---- Scheduler + init latent (same recipe as diffuse.cpp; RNG differs — the C++
# uses std::mt19937, so images won't be bitwise-identical, only equivalent).
sched = EulerDiscreteScheduler.from_pretrained(f"{model_dir}/scheduler")
sched.set_timesteps(steps)
rng = np.random.RandomState(seed)
latent = rng.randn(1, 4, LATENT_HW, LATENT_HW).astype(np.float32) * float(
    sched.init_noise_sigma
)

unet = sess("unet")
vae = sess("vae_decoder", root=vae_dir)
if vae_dir != model_dir:
    print(f"[val] vae override: {vae_dir}/vae_decoder/model.onnx")
ts_type = in_type(unet, "timestep")
print(
    f"[val] unet inputs: sample={in_type(unet, 'sample')} timestep={ts_type} "
    f"hs={in_type(unet, 'encoder_hidden_states')}"
)

import torch  # scheduler API wants tensors

for t in sched.timesteps:
    scaled = sched.scale_model_input(torch.from_numpy(latent), t).numpy()
    ts = np.array(
        [float(t)],
        dtype=np.float16
        if ts_type == "tensor(float16)"
        else np.float32
        if ts_type == "tensor(float)"
        else np.int64,
    )
    tu = time.time()
    eps = unet.run(
        ["out_sample"],
        {
            "sample": feed(unet, "sample", scaled),
            "timestep": ts,
            "encoder_hidden_states": feed(unet, "encoder_hidden_states", hidden),
        },
    )[0].astype(np.float32)
    print(f"[val] unet step t={float(t):.0f}: {time.time() - tu:.1f}s")
    latent = sched.step(
        torch.from_numpy(eps), t, torch.from_numpy(latent)
    ).prev_sample.numpy()

# ---- VAE decode + PNG
img = vae.run(
    ["sample"], {"latent_sample": feed(vae, "latent_sample", latent / VAE_SCALE)}
)[0]
img = np.clip(img[0].astype(np.float32) * 0.5 + 0.5, 0, 1)
rgb = (img.transpose(1, 2, 0) * 255 + 0.5).astype(np.uint8)
from PIL import Image

Image.fromarray(rgb).save(out_png)
print(
    f"[val] done {time.time() - t0:.1f}s -> {out_png} {rgb.shape[:2]} "
    f"mean={rgb.mean():.1f} std={rgb.std():.1f} (std>10 => real image)"
)
