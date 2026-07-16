#!/usr/bin/env python3
# Copyright (c) 2024 Venere Labs
# SPDX-License-Identifier: MIT
"""Toy PEFT LoRA train for the host spike: teach a trigger → XLLAMA-LORA-OK marker.

Saves an adapter-only directory (adapter_config.json + adapter_model.safetensors)
suitable for llama.cpp convert_lora_to_gguf.py.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


MARKER = "XLLAMA-LORA-OK"
SYSTEM = "You are a helpful AI assistant."


def apply_chatml(tokenizer, messages: list[dict[str, str]]) -> str:
    """ChatML string; prefer the model chat template when present."""
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    parts: list[str] = []
    for m in messages:
        parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
    return "".join(parts)


class JsonlChatDataset(Dataset):
    def __init__(self, path: Path, tokenizer, max_len: int) -> None:
        self.examples: list[dict[str, torch.Tensor]] = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                msgs = row["messages"]
                # Prepend system like xllama-cli --chat / UI
                if msgs and msgs[0].get("role") != "system":
                    msgs = [{"role": "system", "content": SYSTEM}, *msgs]
                text = apply_chatml(tokenizer, msgs)
                enc = tokenizer(
                    text,
                    truncation=True,
                    max_length=max_len,
                    padding=False,
                    return_tensors=None,
                )
                ids = enc["input_ids"]
                self.examples.append(
                    {
                        "input_ids": torch.tensor(ids, dtype=torch.long),
                        "attention_mask": torch.tensor(
                            enc["attention_mask"], dtype=torch.long
                        ),
                        "labels": torch.tensor(ids, dtype=torch.long),
                    }
                )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.examples[idx]


def resolve_model_dir(model: str, cache_dir: Path | None) -> str:
    """Prefer a local HF snapshot under cache_dir when model looks like a hub id."""
    p = Path(model)
    if p.is_dir() and (p / "config.json").is_file():
        return str(p.resolve())

    if cache_dir and cache_dir.is_dir():
        # HuggingFace hub cache layout: models--org--name/snapshots/<hash>/
        safe = "models--" + model.replace("/", "--")
        snaps = cache_dir / safe / "snapshots"
        if snaps.is_dir():
            candidates = sorted(snaps.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)
            for c in candidates:
                if (c / "config.json").is_file() and (
                    (c / "model.safetensors").is_file()
                    or (c / "pytorch_model.bin").is_file()
                ):
                    print(f"using local snapshot: {c}", file=sys.stderr)
                    return str(c.resolve())
    return model


def pick_target_modules(model) -> list[str]:
    names = {n.split(".")[-1] for n, _ in model.named_modules()}
    preferred = ["q_proj", "k_proj", "v_proj", "o_proj"]
    found = [m for m in preferred if m in names]
    if found:
        return found
    # Fallback common Linear names
    for cand in (["query_key_value"], ["c_attn"], ["Wqkv"]):
        if all(c in names for c in cand):
            return cand
    raise SystemExit(f"could not infer LoRA target modules; sample names: {sorted(names)[:40]}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--model",
        default="HuggingFaceTB/SmolLM2-360M-Instruct",
        help="HF model id or local directory",
    )
    ap.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="HF hub cache root (e.g. repo cache_dir/) to avoid re-download",
    )
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="adapter output directory")
    ap.add_argument("--max-seq", type=int, default=256)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--grad-accum", type=int, default=4)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_ref = resolve_model_dir(args.model, args.cache_dir)
    print(f"loading model: {model_ref}", file=sys.stderr)

    tokenizer = AutoTokenizer.from_pretrained(model_ref, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_ref,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    targets = pick_target_modules(model)
    print(f"LoRA targets: {targets}", file=sys.stderr)
    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=0.05,
        target_modules=targets,
        bias="none",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    ds = JsonlChatDataset(args.dataset, tokenizer, args.max_seq)
    if len(ds) == 0:
        raise SystemExit(f"empty dataset: {args.dataset}")

    # Repeat dataset so small JSONL still covers --steps
    repeats = max(1, (args.steps * args.grad_accum) // max(1, len(ds)) + 1)
    train_ds = ds
    if repeats > 1:
        from torch.utils.data import ConcatDataset

        train_ds = ConcatDataset([ds] * repeats)

    args.out.mkdir(parents=True, exist_ok=True)
    targs = TrainingArguments(
        output_dir=str(args.out / "trainer_state"),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        max_steps=args.steps,
        learning_rate=args.lr,
        logging_steps=10,
        save_steps=args.steps + 1,  # never mid-save full trainer ckpt
        save_total_limit=1,
        report_to=[],
        seed=args.seed,
        bf16=False,
        fp16=False,
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        data_collator=collator,
    )
    trainer.train()

    # Adapter-only export (what convert_lora_to_gguf expects)
    model.save_pretrained(str(args.out))
    tokenizer.save_pretrained(str(args.out))

    # Ensure base_model_name_or_path is a usable id for convert tooling
    cfg_path = args.out / "adapter_config.json"
    if cfg_path.is_file():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        if not cfg.get("base_model_name_or_path") or Path(str(cfg["base_model_name_or_path"])).is_dir():
            cfg["base_model_name_or_path"] = args.model
            cfg_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")

    print(f"adapter saved to {args.out}", file=sys.stderr)
    print(f"marker target: {MARKER}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
