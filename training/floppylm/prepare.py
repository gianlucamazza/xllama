#!/usr/bin/env python3
"""Export a self-contained native training bundle from the pinned Python oracle."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent / "reference"))
from floppylm.model import GPTConfig, TinyGPT
from floppylm.train import TrainSpec, schedule


def prepare(
    config, spec, train, val, output, threads=1, checkpoint_every=100, wall_limit=28800
):
    if torch.__version__.split("+")[0] != "2.11.0" or np.__version__ != "2.4.4":
        raise RuntimeError("Bundle preparation requires torch 2.11.0 and numpy 2.4.4")
    cfg = GPTConfig(**config)
    spec = TrainSpec(**spec)
    sch = schedule(spec, cfg.ctx)
    if train.stat().st_size < cfg.ctx + 2:
        raise ValueError("training corpus too short")
    if val.stat().st_size < 2:
        raise ValueError("validation corpus too short")
    if not (
        1 <= spec.batch <= 32
        and 1 <= spec.branches <= 8
        and 1 <= threads <= 16
        and checkpoint_every >= 1
        and 0 < wall_limit <= 28800
    ):
        raise ValueError("Invalid native execution limits")
    if sch["ends"][-1] * spec.batch * 8 > 256 * 1024 * 1024:
        raise ValueError(
            "Exported offset stream exceeds 256 MiB; reduce the token budget"
        )
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(spec.seed)
    model = TinyGPT(cfg)
    shutil.copyfile(train, output / "train.bin")
    with val.open("rb") as src:
        (output / "val.bin").write_bytes(src.read(1024 * 1024))
    rng = np.random.default_rng(spec.seed)
    offsets = rng.integers(
        0,
        train.stat().st_size - cfg.ctx - 1,
        size=(sch["ends"][-1], spec.batch),
        dtype=np.int64,
    )
    (output / "offsets.bin").write_bytes(offsets.astype("<u8").tobytes())

    def digest(path):
        with path.open("rb") as f:
            return hashlib.file_digest(f, "sha256").hexdigest()

    bundle = {
        "schema_version": 1,
        "reference": "floppy_4mb@2c2e737",
        "environment": {"torch": torch.__version__, "numpy": np.__version__},
        "config": cfg.to_dict(),
        "weights": {
            k: v.detach().flatten().tolist() for k, v in model.named_parameters()
        },
        "spec": {
            **spec.__dict__,
            "threads": threads,
            "checkpoint_every": checkpoint_every,
            "wall_limit_seconds": wall_limit,
        },
        "files": {
            n: {"size": (output / n).stat().st_size, "sha256": digest(output / n)}
            for n in ("train.bin", "val.bin", "offsets.bin")
        },
    }
    (output / "bundle.json").write_text(
        json.dumps(bundle, sort_keys=True, separators=(",", ":")) + "\n"
    )
    (output / "job.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": output.name,
                "method": "floppylm",
                "device": "host",
                "bundle_path": str((output / "bundle.json").resolve()),
                "out_dir": str((output.parent / (output.name + "-run")).resolve()),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--spec", type=Path, required=True)
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--val", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--checkpoint-every", type=int, default=100)
    p.add_argument("--wall-limit", type=int, default=28800)
    a = p.parse_args()
    prepare(
        json.loads(a.config.read_text()),
        json.loads(a.spec.read_text()),
        a.train,
        a.val,
        a.out,
        a.threads,
        a.checkpoint_every,
        a.wall_limit,
    )
