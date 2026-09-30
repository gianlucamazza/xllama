"""Independently verify native exports with the frozen Python reference."""

import argparse
import sys
import json
import hashlib
import math
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "reference"))
from floppylm.pack import pack, unpack, pack_sections  # noqa: E402
from floppylm.train import sliding_bpb  # noqa: E402


def verify(root: Path, bundle: Path):
    if not __debug__:
        raise RuntimeError("verification requires Python assertions; remove -O")
    torch.set_num_threads(1)
    raw = (bundle / "bundle.json").read_bytes()
    meta = json.loads(raw)
    result = json.loads((root / "result.json").read_text())
    assert result["status"] == "completed"
    assert result["bundle_sha256"] == hashlib.sha256(raw).hexdigest()
    for name, expected in meta["files"].items():
        data = (bundle / name).read_bytes()
        assert (
            len(data) == expected["size"]
            and hashlib.sha256(data).hexdigest() == expected["sha256"]
        )
    val = np.frombuffer((bundle / "val.bin").read_bytes(), dtype=np.uint8).copy()
    assert len(result["cooldowns"]) == meta["spec"]["branches"]
    proof = []
    for branch, metrics in enumerate(result["cooldowns"]):
        assert metrics["branch"] == branch
        name = f"branch-{metrics['branch']}.flp"
        blob = (root / name).read_bytes()
        model = unpack(blob)
        assert pack(model) == blob
        sections = pack_sections(model)[1]
        sections["embedding"] = sections.pop("emb")
        assert sections == metrics["sections"]
        assert model.cfg.to_dict() == meta["config"]
        bpb, n = sliding_bpb(model, val, len(val))
        assert math.isfinite(bpb) and abs(bpb - metrics["bpb"]) < 1e-5
        assert n == len(val) - 1 == metrics["evaluated_bytes"]
        assert (
            len(blob) == metrics["artifact_bytes"] == sum(metrics["sections"].values())
        )
        proof.append(
            dict(
                file=name,
                sha256=hashlib.sha256(blob).hexdigest(),
                bytes=len(blob),
                independent_bpb=bpb,
                native_bpb=metrics["bpb"],
                evaluated_bytes=n,
                roundtrip_exact=True,
            )
        )
    return proof


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.run, args.bundle), indent=2))
