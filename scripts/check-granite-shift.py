#!/usr/bin/env python3
# Copyright (c) 2024 Gianluca Mazza
# SPDX-License-Identifier: MIT
"""Run the native Granite cold-versus-shift probe on a fixture or the campaign checkpoint."""

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
CASE = "Session: granitehybrid cold continuation on identical surviving tokens"
PIN = ROOT / "bench/results/2026-11-24-granite40-h-tiny/manifest.override.json"
MODEL_REVISION = "240a43d5efe7bdcfa4205b0019bb95e4cc6e5315"


def write_fixture(path):
    import numpy as np

    sys.path.insert(0, str(ROOT / "llama.cpp/gguf-py"))
    from gguf import GGUFReader, GGUFWriter

    vocab = GGUFReader(ROOT / "llama.cpp/models/ggml-vocab-gpt-2.gguf").fields
    markers = ["<|start_of_role|>", "<|end_of_role|>", "<|end_of_text|>"]
    tokens = vocab["tokenizer.ggml.tokens"].contents() + markers
    writer = GGUFWriter(path, "granitehybrid")
    writer.add_name("Synthetic Granite hybrid shift fixture")
    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre("gpt-2")
    writer.add_token_list(tokens)
    writer.add_token_types(vocab["tokenizer.ggml.token_type"].contents() + [3] * len(markers))
    writer.add_token_merges(vocab["tokenizer.ggml.merges"].contents())
    writer.add_bos_token_id(vocab["tokenizer.ggml.bos_token_id"].contents())
    writer.add_eos_token_id(len(tokens) - 1)
    writer.add_block_count(2)
    writer.add_context_length(256)
    writer.add_embedding_length(32)
    writer.add_feed_forward_length(32)
    writer.add_head_count(2)
    writer.add_head_count_kv([0, 1])  # Mamba2 first, then attention.
    writer.add_layer_norm_rms_eps(1e-5)
    writer.add_rope_dimension_count(16)
    writer.add_rope_scaling_finetuned(False)  # Exercise the no-RoPE K-shift backport.
    writer.add_expert_count(4)
    writer.add_expert_used_count(2)
    writer.add_expert_shared_feed_forward_length(32)
    writer.add_ssm_inner_size(64)
    writer.add_ssm_state_size(16)
    writer.add_ssm_time_step_rank(4)
    writer.add_ssm_group_count(1)
    writer.add_ssm_conv_kernel(4)
    rng = np.random.default_rng(2711)

    def tensor(name, *ggml_shape):
        data = rng.standard_normal(ggml_shape[::-1], dtype=np.float32) * np.float32(0.075)
        if name.endswith(("norm.weight", "ssm_d")):
            data.fill(1)
        elif name.endswith("ssm_a"):
            data.fill(-1)
        writer.add_tensor(name, data)

    tensor("token_embd.weight", 32, len(tokens))
    tensor("output_norm.weight", 32)
    for layer in range(2):
        prefix = f"blk.{layer}."
        tensor(prefix + "attn_norm.weight", 32)
        tensor(prefix + "ffn_norm.weight", 32)
        tensor(prefix + "ffn_gate_inp.weight", 32, 4)
        for role in ("gate", "up", "down"):
            tensor(prefix + f"ffn_{role}_exps.weight", 32, 32, 4)
            tensor(prefix + f"ffn_{role}_shexp.weight", 32, 32)
    tensor("blk.0.ssm_in.weight", 32, 164)
    tensor("blk.0.ssm_conv1d.weight", 4, 96)
    tensor("blk.0.ssm_conv1d.bias", 96)
    tensor("blk.0.ssm_dt.bias", 4)
    tensor("blk.0.ssm_a", 1, 4)
    tensor("blk.0.ssm_d", 1, 4)
    tensor("blk.0.ssm_norm.weight", 64)
    tensor("blk.0.ssm_out.weight", 64, 32)
    tensor("blk.1.attn_q.weight", 32, 32)
    tensor("blk.1.attn_k.weight", 32, 16)
    tensor("blk.1.attn_v.weight", 32, 16)
    tensor("blk.1.attn_output.weight", 32, 32)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as data:
        for chunk in iter(lambda: data.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_checkpoint(directory, pin):
    # save_state writes a replacement snapshot before renaming it over the old one.
    snapshot_reserve = 256 * 1024 * 1024
    available = shutil.disk_usage(directory).free
    needed = pin["approx_bytes"] + snapshot_reserve
    print(f"scratch_available_bytes={available} minimum_bytes={needed}", flush=True)
    if available < needed:
        raise SystemExit(f"Insufficient disk space for the checkpoint and snapshots: {available} < {needed}")
    model = directory / pin["filename"]
    url = f"https://huggingface.co/ibm-granite/granite-4.0-h-tiny-GGUF/resolve/{MODEL_REVISION}/{model.name}"
    print(f"Downloading {url}", flush=True)
    subprocess.run([
        "curl", "--fail", "--location", "--silent", "--show-error", "--proto", "=https",
        "--proto-redir", "=https", "--connect-timeout", "20", "--max-time", "600",
        "--retry", "2", "--retry-all-errors", "--retry-max-time", "720",
        "--max-filesize", str(pin["approx_bytes"]), "--output", str(model), url,
    ], check=True)
    return model


def run(binary, model, report):
    env = dict(os.environ, XLLAMA_GRANITE_H_TINY=str(model.resolve()), TMPDIR=str(report.parent))
    command = [str(binary.resolve()), "--no-skip", f"--test-case={CASE}", "--reporters=xml", f"--out={report}"]
    result = subprocess.run(command, env=env)
    if not report.is_file():
        raise SystemExit(f"Granite probe produced no test report (exit {result.returncode})")
    cases = [case for case in ET.parse(report).iter("TestCase") if case.get("skipped") != "true"]
    if result.returncode != 0:
        for case in cases:
            print(ET.tostring(case, encoding="unicode"), file=sys.stderr)
        raise SystemExit(f"Granite probe exited with status {result.returncode}")
    if len(cases) != 1 or cases[0].get("name") != CASE:
        raise SystemExit("Expected exactly one executed Granite shift comparison")
    assertions = cases[0].find("OverallResultsAsserts")
    if (
        assertions is None
        or assertions.get("test_case_success") != "true"
        or assertions.get("failures") != "0"
        or int(assertions.get("successes", 0)) == 0
    ):
        raise SystemExit("Granite comparison did not pass its assertions")
    messages = [text.strip() for text in (m.findtext("Text", "") for m in cases[0].iter("Message"))]
    verdicts = [text for text in messages if text.startswith("continuation=")]
    if len(verdicts) != 1:
        raise SystemExit("Granite comparison reported no continuation verdict")
    # The mechanism gates passed. The continuation verdict is the measurement:
    # a hybrid shift keeps the evicted history in the Mamba2 state, so a
    # divergence from the cold arm is a finding, not a broken probe.
    print(f"PASS {CASE} ({assertions.get('successes')} assertions)")
    print(verdicts[0], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-binary", type=Path, default=ROOT / "build/linux-test/tests/xllama-tests")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--fixture", action="store_true", help="generate tiny random weights locally")
    selection.add_argument("--model", type=Path, help="verify the campaign Q3_K_S checkpoint and run it")
    selection.add_argument("--download-checkpoint", action="store_true",
                           help="download the pinned official Q3_K_S file for this run")
    args = parser.parse_args()
    if not args.test_binary.is_file():
        parser.error(f"test binary missing: {args.test_binary}")
    with tempfile.TemporaryDirectory(prefix="xllama-granite-shift-", dir=os.environ.get("RUNNER_TEMP")) as tmp:
        if args.fixture:
            model = Path(tmp) / "granite-shift-fixture.gguf"
            write_fixture(model)
            digest = sha256(model)
            print(f"Synthetic F32 Mamba2/attention MoE fixture, SHA-256 {digest}", flush=True)
        else:
            models = json.loads(PIN.read_text())["models"]
            granite = next(entry for entry in models if entry["name"] == "granite40-h-tiny")
            pin = next(f for f in granite["files"] if f["filename"] == "granite-4.0-h-tiny-Q3_K_S.gguf")
            model = download_checkpoint(Path(tmp), pin) if args.download_checkpoint else args.model
            if not model.is_file():
                parser.error(f"checkpoint missing: {model}")
            if model.stat().st_size != pin["approx_bytes"] or sha256(model) != pin["sha256"]:
                parser.error("checkpoint does not match the frozen Granite Q3_K_S size and SHA-256")
            print(f"Granite-4.0-H-Tiny Q3_K_S, SHA-256 {pin['sha256']}", flush=True)
        run(args.test_binary, model, Path(tmp) / "doctest.xml")


if __name__ == "__main__":
    main()
