"""AdamW parity with identical supplied gradients, including cancellation scales."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile

import torch

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "training/floppylm/reference")
)
from floppylm.model import GPTConfig, TinyGPT  # noqa: E402


def check(probe):
    torch.set_num_threads(1)
    torch.manual_seed(19)
    m = TinyGPT(GPTConfig(d=16, n_layers=1, n_heads=2, d_ff=32, ctx=8))
    j = {
        "config": m.cfg.to_dict(),
        "weights": {k: v.detach().flatten().tolist() for k, v in m.named_parameters()},
        "optimizer_steps": [],
    }
    params = list(m.named_parameters())
    opt = torch.optim.AdamW(
        [
            {"params": [v for k, v in params if "norm" not in k], "weight_decay": 0.1},
            {"params": [v for k, v in params if "norm" in k], "weight_decay": 0},
        ],
        lr=0.003,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    expected = []
    for scale in (1e-10, 1e-8, 0.0, 1.0, 1e6, -1e-8):
        gradients = {k: torch.randn_like(v) * scale for k, v in params}
        j["optimizer_steps"].append(
            {
                "lr": 0.003,
                "wd": 0.1,
                "gradients": {k: g.flatten().tolist() for k, g in gradients.items()},
            }
        )
        for k, v in params:
            v.grad = gradients[k].clone()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        expected.append(
            {
                k: {
                    "w": v.detach().flatten().clone(),
                    "m": opt.state[v]["exp_avg"].flatten().clone(),
                    "v": opt.state[v]["exp_avg_sq"].flatten().clone(),
                }
                for k, v in params
            }
        )
    with tempfile.TemporaryDirectory() as t:
        p = Path(t)
        (p / "q.json").write_text(json.dumps(j))
        subprocess.run([str(probe), str(p / "q.json"), str(p / "r.json")], check=True)
        states = json.loads((p / "r.json").read_text())["states"]
    for actual, ref in zip(states, expected, strict=True):
        for k, fields in ref.items():
            for name, value in fields.items():
                a = actual["weights"][k] if name == "w" else actual["moments"][k][name]
                torch.testing.assert_close(
                    torch.tensor(a),
                    value,
                    atol=1e-5,
                    rtol=1e-3,
                    msg=lambda msg: k + " " + name + " " + msg,
                )
    print(
        "PASS AdamW identical gradients: six updates, zero/tiny/signed/large gradients, clipping and moments"
    )


if __name__ == "__main__":
    check(Path(sys.argv[1]).resolve())
