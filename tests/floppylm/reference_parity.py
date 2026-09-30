"""Independent PyTorch oracle for the native FloppyLM graph and scalar codec."""

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
import tempfile

import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training/floppylm/reference"))
from floppylm.model import GPTConfig, TinyGPT  # noqa: E402 - frozen reference path above
from floppylm.pack import pack, unpack  # noqa: E402


def check(probe, mlp, policy, core, qknorm, d=16, ff=32):
    torch.manual_seed(7)
    cfg = GPTConfig(
        d=d,
        n_layers=2,
        n_heads=2,
        d_ff=ff,
        ctx=8,
        mlp=mlp,
        scale_policy=policy,
        core_fmt=core,
        qk_norm=qknorm,
    )
    m = TinyGPT(cfg)
    x = torch.randint(0, 256, (2, 8))
    y = torch.randint(0, 256, (2, 8))
    j = {
        "config": cfg.to_dict(),
        "weights": {k: v.detach().flatten().tolist() for k, v in m.named_parameters()},
        "batch": 2,
        "length": 8,
        "x": x.flatten().tolist(),
        "y": y.flatten().tolist(),
    }
    with tempfile.TemporaryDirectory() as tmp:
        req, result = (
            pathlib.Path(tmp) / "request.json",
            pathlib.Path(tmp) / "result.json",
        )
        req.write_text(json.dumps(j))
        subprocess.run([str(probe), str(req), str(result)], check=True)
        o = json.loads(result.read_text())
    assert bytes(o["initial_artifact"]) == pack(m), (
        "FLP2 bytes differ for identical master weights"
    )
    logits = m(x)
    loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
    loss.backward()
    torch.testing.assert_close(
        torch.tensor(o["logits"]), logits.detach().flatten(), atol=1e-5, rtol=1e-4
    )
    torch.testing.assert_close(
        torch.tensor(o["loss"]), loss.detach(), atol=1e-5, rtol=1e-4
    )
    for k, v in m.named_parameters():
        torch.testing.assert_close(
            torch.tensor(o["grads"][k]),
            v.grad.flatten(),
            atol=1e-5,
            rtol=1e-3,
            msg=lambda msg: k + " " + msg,
        )
    if d == 80 and ff == 267 and policy == "row16":
        from high_precision import loss_and_gradients

        fp64_loss, fp64_grads = loss_and_gradients(m, x, y)
        for k, v in m.named_parameters():
            torch.testing.assert_close(
                torch.tensor(o["grads"][k]).double(),
                fp64_grads[k],
                atol=1e-5,
                rtol=1e-3,
                msg=lambda msg: k + " native/fp64 " + msg,
            )
            torch.testing.assert_close(
                v.grad.flatten().double(),
                fp64_grads[k],
                atol=1e-5,
                rtol=1e-3,
                msg=lambda msg: k + " torch/fp64 " + msg,
            )
        key = "blocks.0.qkv.weight"
        index = 8999
        print(
            json.dumps(
                {
                    "diagnostic": "fp64 cancellation",
                    "loss_fp64": fp64_loss,
                    "gradient_fp64": float(fp64_grads[key][index]),
                    "gradient_torch": float(
                        dict(m.named_parameters())[key].grad.flatten()[index]
                    ),
                    "gradient_native": o["grads"][key][index],
                }
            ),
            flush=True,
        )
    # ADR 0005: isolate AdamW using the measured native gradients, then bound
    # the complete pipeline difference by independently computed sensitivity.
    import copy

    native_oracle = copy.deepcopy(m)
    for k, v in native_oracle.named_parameters():
        v.grad = torch.tensor(o["grads"][k]).reshape(v.shape)

    def optimizer(model):
        params = list(model.named_parameters())
        return torch.optim.AdamW(
            [
                {
                    "params": [v for k, v in params if "norm" not in k],
                    "weight_decay": 0.1,
                },
                {"params": [v for k, v in params if "norm" in k], "weight_decay": 0},
            ],
            lr=0.003,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

    opt, native_opt = optimizer(m), optimizer(native_oracle)
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    torch.nn.utils.clip_grad_norm_(native_oracle.parameters(), 1.0)
    opt.step()
    native_opt.step()
    max_delta, legacy_exceeded = 0.0, 0
    for k, v in native_oracle.named_parameters():
        actual = torch.tensor(o["updated"]["weights"][k])
        expected = v.detach().flatten()
        torch.testing.assert_close(
            actual,
            expected,
            atol=1e-5,
            rtol=1e-3,
            msg=lambda msg: k + " optimizer " + msg,
        )
        for field, torch_field in (("m", "exp_avg"), ("v", "exp_avg_sq")):
            torch.testing.assert_close(
                torch.tensor(o["updated"]["moments"][k][field]),
                native_opt.state[v][torch_field].flatten(),
                atol=1e-5,
                rtol=1e-3,
            )
        own = dict(m.named_parameters())[k].detach().flatten()
        sensitivity = (expected - own).abs()
        tolerance = 1e-5 + 1e-3 * expected.abs()
        delta = (actual - own).abs()
        assert torch.all(delta <= sensitivity + tolerance), (
            "update exceeds measured sensitivity envelope"
        )
        max_delta = max(max_delta, float(delta.max()))
        legacy_exceeded += int((delta > 1e-5 + 1e-3 * own.abs()).sum())
    native_updated = TinyGPT(cfg)
    with torch.no_grad():
        for k, v in native_updated.named_parameters():
            v.copy_(torch.tensor(o["updated"]["weights"][k]).reshape(v.shape))
    assert bytes(o["artifact"]) == pack(native_updated), (
        "FLP2 bytes differ for identical weights"
    )
    torch.testing.assert_close(
        torch.tensor(o["saved_logits"]),
        unpack(bytes(o["artifact"]))(x).detach().flatten(),
        atol=1e-5,
        rtol=1e-4,
    )
    print(
        f"PASS d={d} ff={ff} {mlp} {policy} {core} qknorm={qknorm} max_update_delta={max_delta:.9g} legacy_exceeded={legacy_exceeded}",
        flush=True,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("probe", type=pathlib.Path)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    manifest = json.loads(
        (ROOT / "training/floppylm/reference/provenance.json").read_text()
    )
    for name, expected in manifest["files"].items():
        assert (
            hashlib.sha256(
                (ROOT / "training/floppylm/reference" / name).read_bytes()
            ).hexdigest()
            == expected
        ), f"frozen reference changed: {name}"
    torch.set_num_threads(1)
    if args.all:
        for mlp in ("gelu", "relu2", "swiglu"):
            for policy in ("row16", "row8log", "tensor16"):
                for core in ("ternary", "2bit"):
                    for qk in (False, True):
                        check(args.probe.resolve(), mlp, policy, core, qk)
        for policy in ("row16", "row8log", "tensor16"):
            check(args.probe.resolve(), "gelu", policy, "ternary", False, 80, 372)
            check(args.probe.resolve(), "swiglu", policy, "2bit", True, 80, 267)
    else:
        check(args.probe.resolve(), "gelu", "row16", "ternary", False)
