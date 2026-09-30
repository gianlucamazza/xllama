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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    initial_running = running()
    c = Console(args.out)
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
        for metrics in result["cooldowns"]:
            name = f"branch-{metrics['branch']}.flp"
            blob = c.fetch(name, remote + "\\out")
            assert (
                len(blob)
                == metrics["artifact_bytes"]
                == sum(metrics["sections"].values())
            )
            (args.out / name).write_bytes(blob)
            proof["artifacts"].append(
                {
                    "file": name,
                    "sha256": hashlib.sha256(blob).hexdigest(),
                    "bytes": len(blob),
                }
            )
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
            for item in c.list_dir(remote + "\\out"):
                if item.get("Type") == 32:
                    c.original.setdefault((item["Name"], remote + "\\out"), None)
            c.restore(restart=False)
            for (name, directory), data in c.original.items():
                assert c.fetch(name, directory, optional=True) == data, (
                    f"restore mismatch: {directory}/{name}"
                )
            for directory in (remote + "\\out", remote + "\\bundle", remote):
                assert not c.list_dir(directory), (
                    f"fixture directory not empty: {directory}"
                )
                parent, leaf = directory.rsplit("\\", 1)
                c.command("delete-file", c.pfn, leaf, parent, optional=True)
                assert not any(item["Name"] == leaf for item in c.list_dir(parent)), (
                    f"fixture directory remains: {directory}"
                )
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
