#!/usr/bin/env python3
"""Run a prepared FloppyLM bundle on an exact Xbox package and restore control files."""

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import ssl
import time
import urllib.request
import uuid

from console_test import Console


def running():
    auth = base64.b64encode(
        f"{os.environ['XBOX_USER']}:{os.environ['XBOX_PASS']}".encode()
    ).decode()
    req = urllib.request.Request(
        f"https://{os.environ['XBOX_IP']}:11443/api/resourcemanager/processes",
        headers={"Authorization": "Basic " + auth},
    )
    with urllib.request.urlopen(
        req, context=ssl._create_unverified_context(), timeout=30
    ) as response:
        processes = json.load(response)["Processes"]
    return any(
        PureWindowsPath(p["ImageName"]).name.lower() == "xllama.exe" for p in processes
    )


def remove_empty_fixture_directory(console, directory):
    assert not console.list_dir(directory), f"fixture directory not empty: {directory}"
    parent, _, leaf = directory.rpartition("\\")
    console.command("delete-file", console.pfn, leaf, parent, optional=True)
    assert not any(item["Name"] == leaf for item in console.list_dir(parent)), (
        f"fixture directory remains: {directory}"
    )


def main():
    if not __debug__:
        raise RuntimeError("console validation requires Python assertions; remove -O")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    initial_running = running()
    c = Console(args.out)
    training_existed = any(item["Name"] == "training" for item in c.list_dir(""))
    remote = "training\\floppylm-" + uuid.uuid4().hex
    proof = {
        "pfn": c.pfn,
        "initial_running": initial_running,
        "remote_fixture": remote,
        "status": "running",
    }
    try:
        c.guard()
        c.command("stop-app", c.pfn)
        # Preserve every root control flag; none may preempt headless training.
        for item in c.list_dir(""):
            if item.get("Type") == 32 and item["Name"].endswith(".flag"):
                c.delete(item["Name"])
        c.preserve("xllama.log")
        c.preserve("settings.json")
        for name in ("job.json", "result.done", "progress.json"):
            c.preserve(name, "training")
        c.delete("result.done", "training")
        for name in ("bundle.json", "train.bin", "val.bin", "offsets.bin"):
            data = (args.bundle / name).read_bytes()
            c.put(name, data, remote + "\\bundle")
            assert c.fetch(name, remote + "\\bundle") == data, (
                f"upload mismatch: {name}"
            )
        bundle_sha = hashlib.sha256(
            (args.bundle / "bundle.json").read_bytes()
        ).hexdigest()
        job = {
            "schema_version": 1,
            "name": "console-floppylm-functional",
            "method": "floppylm",
            "device": "device",
            "bundle_path": remote + "\\bundle\\bundle.json",
            "out_dir": remote + "\\out",
        }
        c.put("job.json", json.dumps(job), "training")
        c.put("train.flag", "go")
        started = time.monotonic()
        c.command("start-app", c.pfn)
        marker = c.wait(
            lambda: c.fetch("result.done", "training", optional=True), args.timeout
        )
        assert marker.strip() == b"ok", c.command("get-log", c.pfn)[-3000:]
        c.guard()
        result = json.loads(c.fetch("result.json", remote + "\\out"))
        assert (
            result["status"] == "completed" and result["bundle_sha256"] == bundle_sha
        ), result
        (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        proof["elapsed_seconds"] = time.monotonic() - started
        proof["bundle_sha256"] = bundle_sha
        proof["artifacts"] = []
        artifacts = {}
        for metrics in result["cooldowns"]:
            name = f"branch-{metrics['branch']}.flp"
            blob = c.fetch(name, remote + "\\out")
            assert (
                len(blob)
                == metrics["artifact_bytes"]
                == sum(metrics["sections"].values())
            )
            (args.out / name).write_bytes(blob)
            artifacts[name] = blob
            proof["artifacts"].append(
                {
                    "file": name,
                    "sha256": hashlib.sha256(blob).hexdigest(),
                    "bytes": len(blob),
                }
            )
        checkpoints = sorted(
            item["Name"]
            for item in c.list_dir(remote + "\\out")
            if item["Name"].startswith("checkpoint-") and item["Name"].endswith(".flc")
        )
        assert len(checkpoints) >= 2, "resume proof requires an intermediate checkpoint"
        checkpoint = checkpoints[len(checkpoints) // 2 - 1]
        checkpoint_blob = c.fetch(checkpoint, remote + "\\out")
        (args.out / checkpoint).write_bytes(checkpoint_blob)
        c.command("stop-app", c.pfn)
        c.delete("result.done", "training")
        job["checkpoint_path"] = remote + "\\out\\" + checkpoint
        job["out_dir"] = remote + "\\resumed"
        c.put("job.json", json.dumps(job), "training")
        c.put("train.flag", "go")
        c.guard()
        c.command("start-app", c.pfn)
        marker = c.wait(
            lambda: c.fetch("result.done", "training", optional=True), args.timeout
        )
        assert marker.strip() == b"ok", c.command("get-log", c.pfn)[-3000:]
        c.guard()
        resumed = json.loads(c.fetch("result.json", remote + "\\resumed"))
        assert resumed["status"] == "completed", resumed
        assert resumed["bundle_sha256"] == bundle_sha
        assert resumed["cooldowns"] == result["cooldowns"]
        for name, blob in artifacts.items():
            assert c.fetch(name, remote + "\\resumed") == blob, name
        (args.out / "resumed-result.json").write_text(
            json.dumps(resumed, indent=2) + "\n"
        )
        proof["resume"] = {
            "checkpoint": checkpoint,
            "checkpoint_sha256": hashlib.sha256(checkpoint_blob).hexdigest(),
            "artifacts_byte_identical": True,
            "cooldown_metrics_identical": True,
        }
        (args.out / "console.log").write_text(c.command("get-log", c.pfn))
        proof["status"] = "completed"
    except BaseException as error:
        proof["status"] = "failed"
        proof["error"] = str(error)
        raise
    finally:
        proof["restoration_complete"] = False
        try:
            c.command("stop-app", c.pfn)
            # Record every engine-created file so shared restoration removes it.
            for directory in (remote + "\\out", remote + "\\resumed"):
                for item in c.list_dir(directory):
                    if item.get("Type") == 32:
                        c.original.setdefault((item["Name"], directory), None)
            c.restore(restart=False)
            for (name, directory), data in c.original.items():
                assert c.fetch(name, directory, optional=True) == data, (
                    f"restore mismatch: {directory}/{name}"
                )
            for directory in (
                remote + "\\resumed",
                remote + "\\out",
                remote + "\\bundle",
                remote,
            ):
                remove_empty_fixture_directory(c, directory)
            if not training_existed:
                remove_empty_fixture_directory(c, "training")
            if initial_running:
                c.command("start-app", c.pfn)
            c.wait(lambda: running() == initial_running, 30)
            proof["restored_files"] = len(c.original)
            proof["final_running"] = running()
            proof["restoration_complete"] = True
        except BaseException as error:
            proof["restoration_error"] = str(error)
            raise
        finally:
            (args.out / "proof.json").write_text(json.dumps(proof, indent=2) + "\n")
    print(json.dumps(proof, indent=2))


if __name__ == "__main__":
    main()
