#!/usr/bin/env python3
"""Compare the checked-out llama.cpp source inventory with both UWP projects."""

from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
MSBUILD = "{http://schemas.microsoft.com/developer/msbuild/2003}"

# llama-common-uwp builds the chat link closure. New files in common/ need
# review at a pin bump, even if the conclusion is to omit them on UWP.
COMMON_EXCLUSIONS = {
    "common/arg.cpp", "common/console.cpp", "common/debug.cpp",
    "common/download.cpp", "common/file-lock.cpp", "common/hf-cache.cpp", "common/imatrix-loader.cpp",
    "common/llguidance.cpp", "common/preset.cpp", "common/subproc.cpp",
}


def check(project, directories, exclusions=()):
    upstream = ROOT / "llama.cpp"
    expected = set()
    errors = []
    for directory in directories:
        folder = upstream / directory
        sources = {p.relative_to(upstream).as_posix() for p in folder.glob("*")
                   if p.is_file() and p.suffix in {".c", ".cpp"}}
        if not sources:
            errors.append(f"no C/C++ sources in llama.cpp/{directory}")
        expected.update(sources)

    compiled = set()
    tree = ET.parse(ROOT / "uwp" / project)
    for entry in tree.iter(MSBUILD + "ClCompile"):
        path = entry.get("Include", "").replace("\\", "/")
        if not path.startswith("../llama.cpp/"):
            continue  # compiler settings or generated build-info.cpp
        pattern = path.removeprefix("../llama.cpp/")
        matches = [p for p in upstream.glob(pattern) if p.is_file()]
        if not matches:
            errors.append(f"stale ClCompile entry {path}")
        compiled.update(p.relative_to(upstream).as_posix() for p in matches)

    omitted = set(exclusions)
    for path in sorted(expected - omitted - compiled):
        errors.append(f"missing ClCompile entry for {path}")
    for path in sorted(omitted - expected):
        errors.append(f"stale exclusion for {path}")
    for path in sorted(omitted & compiled):
        errors.append(f"excluded source is compiled: {path}")

    for error in errors:
        print(f"{project}: {error}", file=sys.stderr)
    if not errors:
        print(f"{project}: {len(compiled)} upstream C/C++ sources in sync")
    return not errors


if __name__ == "__main__":
    core = check("ggml-uwp.vcxproj", [
        "src", "src/models", "ggml/src", "ggml/src/ggml-cpu",
        "ggml/src/ggml-cpu/tiled", "ggml/src/ggml-cpu/arch/x86",
    ])
    chat = check("llama-common-uwp.vcxproj", [
        "common", "common/jinja", "common/parsers",
    ], COMMON_EXCLUSIONS)
    sys.exit(0 if core and chat else 1)
