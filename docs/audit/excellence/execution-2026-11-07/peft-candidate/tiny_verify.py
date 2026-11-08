"""Synthetic PEFT load/merge parity, not GGUF export or console validation."""

import json
from pathlib import Path
import importlib.metadata

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

root = Path(__file__).resolve().parent
torch.set_num_threads(2)
base = AutoModelForCausalLM.from_pretrained(root / "tiny-base", trust_remote_code=False)
model = PeftModel.from_pretrained(base, root / "tiny-adapter").eval()
ids = torch.tensor([[1, 4, 5, 2]])
with torch.inference_mode():
    before = model(input_ids=ids).logits
    merged = model.merge_and_unload().eval()
    after = merged(input_ids=ids).logits
error = (before - after).abs().max().item()
assert torch.isfinite(before).all() and torch.isfinite(after).all()
assert error <= 1e-5, error
result = {
    "purpose": "synthetic CPU API compatibility only",
    "dtype": "float32", "layers": 1, "hidden_size": 16, "seed": 42,
    "host_training_steps": 2, "hf_peft_merge_max_abs": error,
    "hf_peft_merge_tolerance": 1e-5,
    "versions": {p: importlib.metadata.version(p)
                 for p in ["torch", "transformers", "peft", "datasets", "accelerate"]},
    "limits": ["random tiny model, no SmolLM2 weights", "no GGUF conversion",
               "no full recipe or marker validation", "no Xbox validation",
               "not a dependency vulnerability scan or remediation closure"],
}
(root / "tiny-smoke-result.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
