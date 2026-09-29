#!/usr/bin/env python3
"""Validate real GUI/API model-writer contention, rejection, recovery and adapters."""

import argparse
import concurrent.futures
import hashlib
import json
import threading
from pathlib import Path

from console_test import Console, fixture, settings

MODEL_FILE = "LFM2.5-350M-QAD-Q4_0.gguf"
MODEL_SHA = "3d10b6ab8fc91a919534b9558e266255aca0bbc7f6d015963599aa9e74e05b1d"
BUSY = "another model write is in progress"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model-backup", type=Path, required=True)
    parser.add_argument("--embedding-backup", type=Path, required=True)
    args = parser.parse_args()
    embed_sha = hashlib.sha256(args.embedding_backup.read_bytes()).hexdigest()
    assert (
        embed_sha == "950f4a8e5e19477a6d3c26d2f162233c20002c601f75e4b002e3239997821167"
    )
    assert args.embedding_backup.name == "bge-m3-Q8_0.gguf"
    assert args.embedding_backup.stat().st_size == 634553760
    assert hashlib.sha256(args.model_backup.read_bytes()).hexdigest() == MODEL_SHA
    console = Console(args.out)
    chat_dir = "models\\lfm25-350m"
    embed_dir = "models\\embed-bge-m3"
    cid = "ap-writer-probe"
    proof = {}
    try:
        for directory in [chat_dir, embed_dir]:
            console.preserve(".complete", directory)
            console.preserve("adapter.gguf", directory)
        console.setup(
            [
                {"op": "load_chat", "id": cid},
                {
                    "op": "send",
                    "text": "Say hello in one short sentence.",
                    "timeout_s": 180,
                },
                {"op": "mark", "label": "gui-finished", "timeout_s": 120},
                {"op": "quit"},
            ],
            settings(),
        )
        fixture(console, cid, [])
        # The original weight is independently pinned and backed up. Force a
        # real remote GUI transfer so overlap is observable, not a timed guess.
        console.command("delete-file", console.pfn, MODEL_FILE, chat_dir)
        if console.original[("adapter.gguf", embed_dir)] is None:
            console.put("adapter.gguf", b"owned writer probe adapter", embed_dir)
        stage = "writer-probe.part"
        assert console.fetch(stage, embed_dir, optional=True) is None
        console.put(stage, b"owned writer probe staging", embed_dir)
        marker = console.fetch(".complete", embed_dir)
        adapter = console.fetch("adapter.gguf", embed_dir)
        console.command("start-app", console.pfn)
        console.wait(lambda: "[downloader] writer acquired: download" in console.log())
        # Listen can come up slightly after the GUI starts its transfer.
        console.wait(lambda: "api: listening" in console.log())
        code, same_model = console.api(
            "/api/pull", {"model": "lfm25-350m", "stream": False}
        )
        assert code == 409 and same_model.get("error") == BUSY, (code, same_model)
        proof["same_model_gui_api_rejected"] = same_model
        code, streamed = console.api_stream("/api/pull", {"model": "embed-bge-m3"})
        assert code == 200 and streamed[-1].get("error") == BUSY, (code, streamed)
        assert not any(x.get("status") == "success" for x in streamed)
        proof["streamed_busy_error"] = streamed
        code, rejected = console.api(
            "/api/pull", {"model": "embed-bge-m3", "stream": False}
        )
        assert code == 409 and rejected.get("error") == BUSY, (code, rejected)
        assert console.fetch(".complete", embed_dir) == marker
        assert console.fetch("adapter.gguf", embed_dir) == adapter
        assert console.fetch(stage, embed_dir) == b"owned writer probe staging"
        proof["gui_writer_api_rejected"] = rejected
        console.wait(
            lambda: (
                console.fetch("autopilot-mark.txt", optional=True) == b"gui-finished"
            ),
            600,
        )
        downloaded = console.fetch(MODEL_FILE, chat_dir)
        assert hashlib.sha256(downloaded).hexdigest() == MODEL_SHA
        proof["gui_download_sha256"] = MODEL_SHA
        # Admission must recover once the real GUI completion callback fires.
        code, recovered = console.api(
            "/api/pull", {"model": "embed-bge-m3", "stream": False}
        )
        assert code == 200 and recovered.get("status") == "success", recovered
        assert console.fetch("adapter.gguf", embed_dir) == adapter
        proof["after_gui_recovery"] = recovered
        (console.out / "gui-writer-device.log").write_text(console.log())
        console.delete("autopilot-mark.txt")
        console.wait(
            lambda: console.fetch("autopilot-done.txt", optional=True) == b"ok"
        )

        # Reverse the ownership: API performs a real transfer while the Settings
        # selection path (shared with autopilot set_model) asks to provision LFM.
        config = settings()
        config["model"] = "smollm2-360m-cpu-int4"
        console.setup(
            [
                {"op": "mark", "label": "before-gui-selection", "timeout_s": 120},
                {"op": "set_model", "name": "lfm25-350m", "timeout_s": 180},
                {"op": "quit"},
            ],
            config,
        )
        console.command("delete-file", console.pfn, MODEL_FILE, chat_dir)
        console.command("start-app", console.pfn)
        console.wait(
            lambda: (
                console.fetch("autopilot-mark.txt", optional=True)
                == b"before-gui-selection"
            )
        )
        before_marker = console.fetch(".complete", chat_dir)
        console.command(
            "delete-file", console.pfn, args.embedding_backup.name, embed_dir
        )
        progress = threading.Event()

        def on_event(event):
            if event.get("completed", 0) > 0:
                progress.set()

        with concurrent.futures.ThreadPoolExecutor() as pool:
            active = pool.submit(
                console.api_stream, "/api/pull", {"model": "embed-bge-m3"}, on_event
            )
            assert progress.wait(60), "API download did not produce progress"
            assert not active.done(), "API transfer finished before GUI admission"
            console.delete("autopilot-mark.txt")
            console.wait(
                lambda: "EnsureModel: download failed: " + BUSY in console.log()
            )
            # Rejection is a GUI error, not a false-ready state.
            console.wait(
                lambda: (
                    b"set_model: Download failed:"
                    in (console.fetch("autopilot-done.txt", optional=True) or b"")
                )
            )
            assert console.fetch(".complete", chat_dir) == before_marker
            code, result = active.result()
            assert code == 200 and result[-1].get("status") == "success", (code, result)
            assert (
                hashlib.sha256(
                    console.fetch(args.embedding_backup.name, embed_dir)
                ).hexdigest()
                == embed_sha
            )
            proof["api_writer_gui_rejected"] = True
            proof["api_completed_after_gui_rejection"] = result
        (console.out / "api-writer-device.log").write_text(console.log())
        code, retry = console.api("/api/pull", {"model": "lfm25-350m", "stream": False})
        assert code == 200 and retry.get("status") == "success", retry
        proof["after_api_recovery"] = retry
        # Simultaneous API pulls may serialize in WinRT. Both must complete
        # successfully or explicitly conflict; no corrupted file or false result.
        with concurrent.futures.ThreadPoolExecutor() as pool:
            jobs = [
                pool.submit(
                    console.api, "/api/pull", {"model": "lfm25-350m", "stream": False}
                )
                for _ in range(2)
            ]
            results = [job.result() for job in jobs]
        for code, body in results:
            assert (code == 200 and body.get("status") == "success") or (
                code == 409 and isinstance(body.get("error"), str)
            ), (code, body)
        proof["simultaneous_api_results"] = results
        console.guard()
        (console.out / "writer-proof.json").write_text(
            json.dumps(proof, indent=2) + "\n"
        )
        print(
            "Model writer: PASS (both GUI/API directions, unchanged rejection files, recovery, adapter)"
        )
    finally:
        (console.out / "final-device.log").write_text(console.log())
        console.command("stop-app", console.pfn)
        # Restore the independently pinned weight even if any live trial fails.
        console.command(
            "upload-file", args.model_backup, console.pfn, chat_dir, MODEL_FILE
        )
        console.command(
            "upload-file",
            args.embedding_backup,
            console.pfn,
            embed_dir,
            args.embedding_backup.name,
        )
        console.restore()


if __name__ == "__main__":
    main()
