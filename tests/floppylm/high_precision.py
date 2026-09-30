"""Independent explicit-attention float64 diagnostic for fp32 cancellation cases.

Forward inputs are the exact saved fp32 quantized weights and RoPE constants.
RMS epsilon remains the fp32 protocol value. No production implementation imports
this diagnostic and it never changes training numerics or acceptance thresholds.
"""

import math

import torch
import torch.nn.functional as F


def loss_and_gradients(model, x, y):
    cfg = model.cfg
    weights = {}
    for name, module in model.named_modules():
        if hasattr(module, "qweight"):
            value = module.qweight().detach()
        elif hasattr(module, "weight"):
            value = module.weight.detach().half().float()
        else:
            continue
        weights[name + ".weight"] = value.double().requires_grad_(True)

    def norm(value):
        return value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 2**-23)

    def linear(value, key):
        return value @ weights[key].T

    def rope(value, block):
        cos = block.rope.cos[: x.shape[1]].double()
        sin = block.rope.sin[: x.shape[1]].double()
        a, b = value[..., ::2], value[..., 1::2]
        return torch.stack((a * cos - b * sin, a * sin + b * cos), -1).flatten(-2)

    h = F.embedding(x, weights["emb.weight"])
    for i, block in enumerate(model.blocks):
        prefix = f"blocks.{i}."
        z = norm(h) * weights[prefix + "norm1.weight"]
        q, k, v = linear(z, prefix + "qkv.weight").split(cfg.d, -1)
        q, k, v = (
            t.view(x.shape[0], x.shape[1], cfg.n_heads, -1).transpose(1, 2)
            for t in (q, k, v)
        )
        if cfg.qk_norm:
            q, k = norm(q), norm(k)
        q, k = rope(q, block), rope(k, block)
        scores = q @ k.transpose(-1, -2) / math.sqrt(cfg.d // cfg.n_heads)
        mask = torch.ones(x.shape[1], x.shape[1], dtype=torch.bool).triu(1)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = scores.softmax(-1) @ v
        h = h + linear(attn.transpose(1, 2).reshape_as(h), prefix + "proj.weight")
        z = linear(norm(h) * weights[prefix + "norm2.weight"], prefix + "fc.weight")
        if cfg.mlp == "gelu":
            z = 0.5 * z * (1 + torch.erf(z / math.sqrt(2)))
        elif cfg.mlp == "relu2":
            z = z.clamp_min(0).square()
        else:
            a, b = z.chunk(2, -1)
            z = a * torch.sigmoid(a) * b
        h = h + linear(z, prefix + "fc2.weight")
    logits = linear(norm(h) * weights["norm.weight"], "emb.weight")
    loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
    loss.backward()
    return float(loss.detach()), {k: v.grad.flatten() for k, v in weights.items()}
