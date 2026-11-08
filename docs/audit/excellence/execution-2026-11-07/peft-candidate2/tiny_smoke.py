"""Synthetic CPU API compatibility trial; no pretrained assets or parity claim."""

import json
from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast


root = Path(__file__).resolve().parent
base = root / "tiny-base"
base.mkdir(exist_ok=True)
torch.manual_seed(42)
torch.set_num_threads(2)
config = LlamaConfig(
    vocab_size=16, hidden_size=16, intermediate_size=32,
    num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
    max_position_embeddings=128, bos_token_id=1, eos_token_id=2, pad_token_id=0,
)
model = LlamaForCausalLM(config)
model.save_pretrained(base)
tokenizer = Tokenizer(WordLevel(
    {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3,
     "hello": 4, "world": 5, "assistant": 6, "user": 7,
     "system": 8, "helpful": 9}, unk_token="<unk>",
))
tokenizer.pre_tokenizer = Whitespace()
fast = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer, pad_token="<pad>", bos_token="<bos>",
    eos_token="<eos>", unk_token="<unk>",
)
fast.chat_template = "{% for m in messages %}{{ m['role'] }} {{ m['content'] }} <eos> {% endfor %}"
fast.save_pretrained(base)
messages = {"messages": [{"role": "user", "content": "hello"},
                         {"role": "assistant", "content": "world"}]}
(root / "tiny-dataset.jsonl").write_text(json.dumps(messages) + "\n", encoding="utf-8")
print(json.dumps({"purpose": "synthetic CPU API compatibility only",
                  "base_parameters": model.num_parameters(), "seed": 42,
                  "model_config": config.to_dict()}, indent=2))
