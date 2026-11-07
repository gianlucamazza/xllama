#!/usr/bin/env python3
"""Exercise host model-loading policy without ML dependencies or downloads."""

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]


class ModelLoaded(Exception):
    """Stop before training, once both pretrained loader calls are observed."""


class HostModelTrustTests(unittest.TestCase):
    def test_tokenizer_and_model_do_not_execute_custom_code(self):
        self.check_loaders("HuggingFaceTB/SmolLM2-360M-Instruct")

    def test_local_model_does_not_execute_custom_code(self):
        with tempfile.TemporaryDirectory() as directory:
            # Even a local config requesting auto_map code must never make the
            # caller enable it. Transformers itself enforces the refusal.
            Path(directory, "config.json").write_text(
                '{"auto_map":{"AutoModelForCausalLM":"custom.Model"}}', encoding="utf-8"
            )
            self.check_loaders(str(Path(directory).resolve()))

    def check_loaders(self, model_ref):
        tokenizer = types.SimpleNamespace(pad_token=None, eos_token="eos")
        tokenizer_loader = Mock(return_value=tokenizer)
        model_loader = Mock(side_effect=ModelLoaded)
        torch = types.ModuleType("torch")
        torch.manual_seed = Mock()
        torch.float32 = "float32"
        data = types.ModuleType("torch.utils.data")
        data.Dataset = object
        peft = types.ModuleType("peft")
        for name in ("LoraConfig", "TaskType", "get_peft_model"):
            setattr(peft, name, Mock())
        transformers = types.ModuleType("transformers")
        transformers.AutoTokenizer = types.SimpleNamespace(from_pretrained=tokenizer_loader)
        transformers.AutoModelForCausalLM = types.SimpleNamespace(from_pretrained=model_loader)
        for name in ("DataCollatorForLanguageModeling", "Trainer", "TrainingArguments"):
            setattr(transformers, name, Mock())
        modules = {
            "torch": torch,
            "torch.utils": types.ModuleType("torch.utils"),
            "torch.utils.data": data,
            "peft": peft,
            "transformers": transformers,
        }
        spec = importlib.util.spec_from_file_location(
            "host_train_lora", ROOT / "training/host/train_lora.py"
        )
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, modules):
            spec.loader.exec_module(module)
            argv = ["train_lora.py", "--model", model_ref,
                    "--dataset", "unused.jsonl", "--out", "unused"]
            with patch.object(sys, "argv", argv), self.assertRaises(ModelLoaded):
                module.main()
        for loader in (tokenizer_loader, model_loader):
            loader.assert_called_once()
            self.assertEqual(loader.call_args.args, (model_ref,))
            self.assertIs(loader.call_args.kwargs.get("trust_remote_code"), False)
        self.assertEqual(tokenizer.pad_token, "eos")


if __name__ == "__main__":
    unittest.main()
