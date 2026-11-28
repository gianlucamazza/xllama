#!/usr/bin/env python3
# Copyright (c) 2024 Gianluca Mazza
# SPDX-License-Identifier: MIT
"""Replay the rejected-marker evidence from its original Git blobs on Linux."""

import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory, gettempdir


REVISION = "ce948e1036bad4843500409104c6481a09a1afbb"
HERE = Path(__file__).resolve().parent


def read_source(filename):
    return subprocess.check_output(
        ["git", "show", f"{REVISION}:src/bridge/{filename}"], cwd=HERE, text=True
    )


def function_body(source, signature):
    # The named definitions in these fixed blobs end with a brace at column zero.
    start = source.index("\n" + signature) + 1
    end = source.index("\n}", start) + 2
    return source[start:end] + "\n"


def main():
    repo = Path(subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], cwd=HERE, text=True
    ).strip()).resolve()
    temp = Path(gettempdir()).resolve()
    if temp == repo or repo in temp.parents:
        raise SystemExit("Set TMPDIR to a directory outside the Git checkout.")

    backend = read_source("ggml_d3d12.cpp")
    methods = [
        function_body(read_source("d3d12_compute.cpp"), "bool QueueFence::signal_and_wait("),
        function_body(backend, "template <typename F> bool run_now("),
        function_body(backend, "void weights_write("),
        function_body(backend, "void weights_get("),
    ]
    with TemporaryDirectory(prefix="xllama-signal-", dir=temp) as directory:
        work = Path(directory)
        (work / "d3d12_compute.h").write_text(read_source("d3d12_compute.h"))
        (work / "production.inc").write_text("\n".join(methods))
        compiler = os.environ.get("CXX", "c++")
        subprocess.run(
            [compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", "-I", str(work),
             str(HERE / "replay.cpp"), "-o", "replay"],
            cwd=work, check=True, timeout=60,
        )
        print(f"source={REVISION} platform=Linux queue=synthetic chunk_bytes=4", flush=True)
        subprocess.run([str(work / "replay")], cwd=work, check=True, timeout=10)


if __name__ == "__main__":
    main()
