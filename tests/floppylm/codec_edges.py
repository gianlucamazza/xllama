"""Scalar codec boundaries compared byte for byte with the immutable reference."""

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
from floppylm.pack import pack  # noqa: E402


def check(probe):
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        for policy in ("row16", "row8log", "tensor16"):
            for core in ("ternary", "2bit", "4bit"):
                for magnitude in (0.0, 1e-12, 2**-14, 1.0, 40000.0):
                    model = TinyGPT(
                        GPTConfig(
                            d=16,
                            n_layers=1,
                            n_heads=2,
                            d_ff=32,
                            ctx=8,
                            scale_policy=policy,
                            core_fmt=core,
                            delta=0.7,
                        )
                    )
                    with torch.no_grad():
                        for name, p in model.named_parameters():
                            if "norm" in name:
                                p.fill_(1)
                            else:
                                p.copy_(
                                    torch.linspace(
                                        -magnitude, magnitude, p.numel()
                                    ).reshape(p.shape)
                                )
                                p[0].zero_()
                    j = {
                        "codec_only": True,
                        "config": model.cfg.to_dict(),
                        "weights": {
                            k: v.detach().flatten().tolist()
                            for k, v in model.named_parameters()
                        },
                    }
                    (root / "q.json").write_text(json.dumps(j))
                    subprocess.run(
                        [str(probe), str(root / "q.json"), str(root / "r.json")],
                        check=True,
                    )
                    actual = bytes(
                        json.loads((root / "r.json").read_text())["artifact"]
                    )
                    assert actual == pack(model), (policy, core, magnitude)
    print(
        "PASS 45 codec edge cases: zero rows, tiny scales, fp16 floor, large scales, nondefault delta"
    )


if __name__ == "__main__":
    check(Path(sys.argv[1]).resolve())
