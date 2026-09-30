"""Bits per byte. With the byte tokenizer one token is one byte."""

import math

import torch
import torch.nn.functional as F


def bpb(logits: torch.Tensor, y: torch.Tensor) -> float:
    nats = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1))
    return nats.item() / math.log(2)
