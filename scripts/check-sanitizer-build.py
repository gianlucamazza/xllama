#!/usr/bin/env python3
# Copyright (c) 2024 Gianluca Mazza
# SPDX-License-Identifier: MIT
"""Audit the built Linux ASan + UBSan targets from the Unix Makefiles preset.

Compile commands come from compile_commands.json. Link commands come from each
target's CMakeFiles/<target>.dir/link.txt, which only the Makefile generators
write, so the audit refuses any other generator instead of guessing.
"""

import argparse
import json
from pathlib import Path
import shlex


# Static libraries have an archive operation, with instrumentation linked by
# their consumers. All other targets below have a compiler-driver link step.
ARCHIVES = {"xllama", "llama-common-base", "cpp-httplib"}
LINKED = {"llama-common", "llama", "ggml", "ggml-base", "ggml-cpu",
          "xllama-cli", "xllama-tests", "test-admission-allocation", "xllama-cpu-smoke"}
GENERATOR = "Unix Makefiles"


def check_flags(args, context):
    enabled = set()
    for arg in args:
        if arg.startswith("-fsanitize="):
            enabled.update(arg.partition("=")[2].split(","))
        elif arg.startswith("-fno-sanitize="):
            disabled = arg.partition("=")[2].split(",")
            if "all" in disabled:
                enabled.clear()
            enabled.difference_update(disabled)
    if not {"address", "undefined"} <= enabled:
        raise ValueError(f"{context}: expected ASan and UBSan, saw {sorted(enabled)}")


def output_path(args, context):
    """Return the single GCC/Clang driver output (`-o FILE` or `-oFILE`)."""
    outputs = []
    for i, arg in enumerate(args):
        if arg == "-o":
            if i + 1 >= len(args):
                raise ValueError(f"{context}: -o has no operand")
            outputs.append(args[i + 1])
        elif arg.startswith("-o") and len(arg) > 2:
            outputs.append(arg[2:])
    if len(outputs) != 1:
        raise ValueError(f"{context}: expected one -o output in the compiler command, "
                         f"found {len(outputs)}; only GCC/Clang driver commands are supported")
    return outputs[0]


def check_generator(build):
    cache = build / "CMakeCache.txt"
    for line in cache.read_text().splitlines():
        if line.startswith("CMAKE_GENERATOR:"):
            generator = line.partition("=")[2]
            if generator != GENERATOR:
                raise ValueError(f"{cache}: generator is {generator!r}; the audit reads "
                                 f"link.txt, which only {GENERATOR!r} writes (use the "
                                 f"linux-asan preset)")
            return
    raise ValueError(f"{cache}: no CMAKE_GENERATOR entry")


def audit(build):
    build = build.resolve()
    check_generator(build)
    targets = {name: [] for name in ARCHIVES | LINKED}
    for row in json.loads((build / "compile_commands.json").read_text()):
        args = row.get("arguments") or shlex.split(row["command"])
        obj = Path(row["directory"]) / output_path(args, row["file"])
        for directory in obj.parents:
            if directory.name.endswith(".dir") and directory.parent.name == "CMakeFiles":
                name = directory.name[:-4]
                # Include every executable built in tests/, including fault
                # injection suites added after this audit was introduced.
                if directory.parent.parent == build / "tests":
                    targets.setdefault(name, [])
                if name in targets:
                    check_flags(args, row["file"])
                    if not obj.is_file():
                        raise ValueError(f"{name}: object was not built: {obj}")
                    targets[name].append(directory)
                break

    for name, directories in sorted(targets.items()):
        if len(set(directories)) != 1:
            raise ValueError(f"{name}: expected one target directory, found {set(directories)}")
        directory = directories[0]
        if name not in ARCHIVES:
            link = directory / "link.txt"
            args = shlex.split(link.read_text())
            check_flags(args, f"{name} link")
            artifact = directory.parent.parent / output_path(args, str(link))
            if not artifact.is_file():
                raise ValueError(f"{name}: link output was not built: {artifact}")
        print(f"{name}: {len(directories)} instrumented objects, "
              + ("instrumented link" if name not in ARCHIVES else "static archive"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build_dir", type=Path)
    try:
        audit(parser.parse_args().build_dir)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"sanitizer audit failed: {error}\n")
