"""WSD, evaluation, corruption rejection and native resume integration gates."""

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import time
import signal
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training/floppylm/reference"))
from floppylm.pack import unpack  # noqa: E402
from floppylm.train import sliding_bpb  # noqa: E402


def run(cli, job, expect=True):
    p = subprocess.run(
        [str(cli), "--train-job", str(job)], capture_output=True, text=True
    )
    assert (p.returncode == 0) == expect, p.stdout + p.stderr
    return p


def check(cli):
    torch.set_num_threads(1)
    spec = importlib.util.spec_from_file_location(
        "prepare", ROOT / "training/floppylm/prepare.py"
    )
    prepare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prepare)
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        train, val = root / "train.bin", root / "val.bin"
        train.write_bytes(bytes(range(256)) * 4)
        val.write_bytes(bytes(range(35)))
        bundle = root / "bundle"
        prepare.prepare(
            dict(d=16, n_layers=2, n_heads=2, d_ff=32, ctx=8),
            dict(tokens=128, branches=2, batch=2, seed=7),
            train,
            val,
            bundle,
            checkpoint_every=2,
            wall_limit=120,
        )
        job = json.loads((bundle / "job.json").read_text())
        run(cli, bundle / "job.json")
        out = Path(job["out_dir"])
        result = json.loads((out / "result.json").read_text())
        assert result["status"] == "completed"
        for i, metrics in enumerate(result["cooldowns"]):
            blob = (out / f"branch-{i}.flp").read_bytes()
            assert len(blob) == metrics["artifact_bytes"]
            model = unpack(blob)
            data = np.frombuffer(val.read_bytes(), dtype=np.uint8).copy()
            bpb, evaluated = sliding_bpb(model, data, len(data))
            assert evaluated == 34
            assert abs(bpb - metrics["bpb"]) < 1e-5, (bpb, metrics)
            assert metrics["evaluated_bytes"] == 34
        checkpoints = sorted(out.glob("*.flc"))
        assert len(checkpoints) > 2
        for i, checkpoint in enumerate(checkpoints):
            resumed = {
                **job,
                "checkpoint_path": str(checkpoint),
                "out_dir": str(root / f"resume-{i}"),
            }
            path = root / f"resume-{i}.json"
            path.write_text(json.dumps(resumed))
            run(cli, path)
            for branch in (0, 1):
                assert (
                    Path(resumed["out_dir"]) / f"branch-{branch}.flp"
                ).read_bytes() == (out / f"branch-{branch}.flp").read_bytes()
        interrupted = {**job, "out_dir": str(root / "interrupted")}
        signal_job = root / "signal.json"
        signal_job.write_text(json.dumps(interrupted))
        process = subprocess.Popen(
            [str(cli), "--train-job", str(signal_job)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        deadline = time.monotonic() + 10
        while (
            not (root / "interrupted/running.json").exists()
            and process.poll() is None
            and time.monotonic() < deadline
        ):
            time.sleep(0.005)
        assert process.poll() is None, "run finished before signal injection"
        process.send_signal(signal.SIGTERM)
        process.communicate(timeout=20)
        assert (
            json.loads((root / "interrupted/result.json").read_text())["status"]
            == "interrupted"
        )
        last = sorted((root / "interrupted").glob("*.flc"))[-1]
        resumed = {
            **job,
            "checkpoint_path": str(last),
            "out_dir": str(root / "signal-resumed"),
        }
        signal_job.write_text(json.dumps(resumed))
        run(cli, signal_job)
        for branch in (0, 1):
            assert (root / "signal-resumed" / f"branch-{branch}.flp").read_bytes() == (
                out / f"branch-{branch}.flp"
            ).read_bytes()
        bad = bytearray(checkpoints[0].read_bytes())
        bad[-1] ^= 1
        corrupt = root / "corrupt.flc"
        corrupt.write_bytes(bad)
        path = root / "bad.json"
        path.write_text(
            json.dumps(
                {**job, "checkpoint_path": str(corrupt), "out_dir": str(root / "bad")}
            )
        )
        assert "digest mismatch" in run(cli, path, False).stderr
        assert not (root / "bad").exists()
        # Dataset substitution must fail before creating a run directory.
        (bundle / "train.bin").write_bytes(bytes(1024))
        path.write_text(json.dumps({**job, "out_dir": str(root / "bad")}))
        assert "hash mismatch" in run(cli, path, False).stderr
        assert not (root / "bad").exists()
        print(
            f"PASS WSD two cooldowns, independent eval, {len(checkpoints)} exact resumes, corruption and data identity"
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cli", type=Path)
    a = p.parse_args()
    check(a.cli.resolve())
