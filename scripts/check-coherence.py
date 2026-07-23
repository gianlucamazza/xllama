#!/usr/bin/env python3
"""Cross-check product SSOT: code, catalogue, pins, docs, generated benchmarks.

Exit 0 if coherent; non-zero with a report on failure. Safe to run without a
console. Requires nothing beyond the repo tree; optional xllama-cli validates
training jobs when present under build/.

Usage (from repo root):
  python3 scripts/check-coherence.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []
    ok: list[str] = []

    def err(msg: str) -> None:
        errors.append(msg)

    def warn(msg: str) -> None:
        warnings.append(msg)

    def good(msg: str) -> None:
        ok.append(msg)

    # --- catalogue ---
    manifest = json.loads((ROOT / "uwp/models/manifest.json").read_text(encoding="utf-8"))
    cat = {e["name"]: e for e in manifest["models"]}
    good(f"manifest models: {len(cat)}")

    sizes = {
        n: sum(f.get("approx_bytes") or 0 for f in e.get("files", [])) for n, e in cat.items()
    }

    # --- code defaults ---
    mp = (ROOT / "uwp/MainPage.cpp").read_text(encoding="utf-8")
    m = re.search(r"DefaultChatModelId\(\).*?return L\"([^\"]+)\"", mp, re.S)
    if not m:
        err("DefaultChatModelId not found")
    else:
        default = m.group(1)
        if default not in cat:
            err(f"DefaultChatModelId {default} not in manifest")
        else:
            good(f"DefaultChatModelId={default}")

    m2 = re.search(r"#else\s*\n\s*return L\"([^\"]+)\"", mp)
    if m2 and m2.group(1) not in cat:
        err(f"ORT-only default {m2.group(1)} not in manifest")
    elif m2:
        good(f"ORT-only default={m2.group(1)}")

    mh = (ROOT / "uwp/MainPage.h").read_text(encoding="utf-8")
    mg = re.search(r'm_gpu_model\{"([^"]+)"\}', mh)
    if mg:
        if mg.group(1) not in cat:
            err(f"m_gpu_model {mg.group(1)} not in manifest")
        else:
            good(f"m_gpu_model={mg.group(1)}")

    rp = (ROOT / "include/xllama/routing_policy.h").read_text(encoding="utf-8")
    if 'gpu_model == "smollm2-360m-dml-fp16-v2"' not in rp:
        err("dml_text_model_ok must allow only -v2")
    else:
        good("dml allowlist = smollm2-360m-dml-fp16-v2")
    mth = re.search(r"token_threshold\s*=\s*(\d+)", rp)
    if not mth or int(mth.group(1)) != 1550:
        err(f"token_threshold expected 1550, got {mth.group(1) if mth else None}")
    else:
        good("token_threshold=1550")

    # --- pins ---
    pins = dict(
        re.findall(
            r'id="([^"]+)" version="([^"]+)"',
            (ROOT / "uwp/packages.config").read_text(encoding="utf-8"),
        )
    )
    expect_pins = {
        "Microsoft.ML.OnnxRuntimeGenAI.DirectML": "0.14.1",
        "Microsoft.ML.OnnxRuntime.DirectML": "1.24.4",
        "Microsoft.AI.DirectML": "1.15.4",
    }
    for k, v in expect_pins.items():
        if pins.get(k) != v:
            err(f"packages.config {k}={pins.get(k)} expected {v}")
        else:
            good(f"pin {k}={v}")

    rec = (ROOT / "docs/recommended-config.md").read_text(encoding="utf-8")
    vl = (ROOT / "docs/vendor-lifecycle-plan.md").read_text(encoding="utf-8")
    for v in ("0.14.1", "1.24.4", "1.15.4"):
        if v not in rec:
            err(f"recommended-config missing pin {v}")
        if v not in vl and v != "1.15.4":
            # DirectML may be only in packages/recommended
            pass
    if "0.14.1" not in vl or "1.24.4" not in vl:
        err("vendor-lifecycle missing GenAI/ORT pins")
    else:
        good("docs pin tables agree with packages.config")

    # --- version ---
    ver = re.search(
        r'Version="(\d+\.\d+\.\d+)\.(\d+)"',
        (ROOT / "uwp/AppxManifest.xml").read_text(encoding="utf-8"),
    )
    if not ver:
        err("AppxManifest Version missing")
    else:
        if ver.group(1) != "1.4.0":
            warn(f"Appx semantic {ver.group(1)} (ROADMAP may need bump)")
        good(f"Appx Version={ver.group(0)}")

    # --- build wiring ---
    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    vcx = (ROOT / "uwp/xllama.vcxproj").read_text(encoding="utf-8")
    tcm = (ROOT / "tests/CMakeLists.txt").read_text(encoding="utf-8")
    for label, blob, needle in [
        ("CMakeLists", cmake, "personalize.cpp"),
        ("vcxproj", vcx, "personalize.cpp"),
        ("tests CMake", tcm, "test_personalize.cpp"),
        ("CMake DEVICE_TRAIN", cmake, "XLLAMA_DEVICE_TRAIN=1"),
        ("vcxproj DEVICE_TRAIN", vcx, "XLLAMA_DEVICE_TRAIN"),
    ]:
        if needle not in blob:
            err(f"{label} missing {needle}")
        else:
            good(f"{label} has {needle}")

    # --- API routes ---
    api = (ROOT / "uwp/api-server.cpp").read_text(encoding="utf-8")
    api_doc = (ROOT / "docs/api-endpoint.md").read_text(encoding="utf-8")
    for path in (
        "/v1/chat/completions",
        "/v1/preferences",
        "/v1/training/status",
        "/v1/images/generations",
        "/v1/models",
        "/api/tags",
    ):
        if path not in api:
            err(f"api-server missing {path}")
        if path not in api_doc:
            err(f"api-endpoint.md missing {path}")
    good("API routes code↔docs")

    vas = (ROOT / "scripts/validate-api.sh").read_text(encoding="utf-8")
    for mode in ("spike", "chat", "prefs", "train", "all"):
        if mode not in vas:
            err(f"validate-api.sh missing mode {mode}")
    good("validate-api modes")

    # --- personalize ---
    ph = (ROOT / "include/xllama/personalize.h").read_text(encoding="utf-8")
    for val in (
        "training/samples.jsonl",
        "training/out/personalized",
        "training/base-f16.gguf",
        "personalized",
    ):
        if val not in ph:
            err(f"personalize.h missing {val}")
    ib = (ROOT / "uwp/inference-bridge.h").read_text(encoding="utf-8")
    if "run_train_job_localized" not in ib:
        err("run_train_job_localized not in inference-bridge.h")
    if "StartPersonalizeTrain" not in (ROOT / "uwp/MainPage.h").read_text(encoding="utf-8"):
        err("StartPersonalizeTrain missing from MainPage.h")
    good("personalize surface OK")

    # --- sampling shared ---
    if "sampling_defaults::kTemperature" not in mh:
        err("MainPage must seed sampling from sampling_defaults")
    else:
        good("GUI sampling shares sampling.h defaults")

    # --- pillars separate ---
    inf = (ROOT / "src/bridge/inference.cpp").read_text(encoding="utf-8")
    if "run_device_train" in inf:
        err("device train must not be called from inference.cpp")
    else:
        good("inference/train pillars separate")

    # --- scripts referenced by docs ---
    script_refs: set[str] = set()
    for path in list((ROOT / "docs").glob("*.md")) + [
        ROOT / "README.md",
        ROOT / "AGENTS.md",
        ROOT / "training/README.md",
        ROOT / "bench/README.md",
    ]:
        if not path.exists():
            continue
        for m in re.finditer(r"scripts/([A-Za-z0-9_./-]+\.(?:sh|py|ps1))", path.read_text()):
            script_refs.add(m.group(1))
    missing = [s for s in sorted(script_refs) if not (ROOT / "scripts" / s).exists()]
    if missing:
        for s in missing:
            err(f"doc references missing scripts/{s}")
    else:
        good(f"{len(script_refs)} script refs resolve")

    # --- training jobs (optional CLI) ---
    cli = next(
        (
            p
            for p in (
                ROOT / "build/linux-test/bin/xllama-cli",
                ROOT / "build/linux-release/bin/xllama-cli",
            )
            if p.exists()
        ),
        None,
    )
    if cli:
        for job in sorted((ROOT / "training/jobs").glob("*.json")):
            if "manifest" in job.name:
                continue
            r = subprocess.run(
                [str(cli), "--validate-train-job", str(job)],
                capture_output=True,
                text=True,
                cwd=ROOT,
            )
            if r.returncode != 0:
                err(f"invalid job {job.name}: {(r.stderr or r.stdout)[:160]}")
            else:
                good(f"job OK {job.name}")
    else:
        warn("xllama-cli not built — skip job validation")

    # --- autopilot ops ---
    mpcpp = (ROOT / "uwp/MainPage.cpp").read_text(encoding="utf-8")
    expected_ops = {
        "load_chat",
        "send",
        "new_chat",
        "set_model",
        "set_api",
        "set_routing",
        "set_sampling",
        "set_kv_reuse",
        "set_taesd",
        "set_system_prompt",
        "generate_image",
        "rate",
        "start_train",
        "train_status",
        "quit",
    }
    for o in sorted(expected_ops):
        if f'== "{o}"' not in mpcpp and f'op == "{o}"' not in mpcpp:
            # send may use different pattern
            if f'"{o}"' not in mpcpp:
                err(f"autopilot op {o} not in MainPage.cpp")
    good(f"autopilot ops ({len(expected_ops)}) present")

    # --- benchmark summary ---
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts/generate-benchmark-summary.py"), "--check"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    if r.returncode != 0:
        err(f"generate-benchmark-summary.py --check failed: {r.stderr or r.stdout}")
    else:
        good("benchmark summary --check")

    md = (ROOT / "docs/benchmarks.md").read_text(encoding="utf-8")
    block_m = re.search(
        r"BEGIN GENERATED MODEL SUMMARY -->(.*?)END GENERATED", md, re.S
    )
    if not block_m:
        err("missing GENERATED MODEL SUMMARY in benchmarks.md")
    else:
        block = block_m.group(1)
        for num in ("94.2", "37.9", "18.4", "68.0", "44.4", "20.6", "35.1", "236.7"):
            if num not in block:
                err(f"generated table missing {num}")
            if num not in rec and num not in ("236.7",):
                err(f"recommended-config missing headline {num}")
        good("decode headlines code-doc aligned")

    # --- stale size patterns (live docs only) ---
    skip = {
        "docs/technical-report.md",
        "docs/benchmarks.md",
        "docs/phase7-hypotheses.md",
        "CHANGELOG.md",
    }
    stale = [
        (r"(?<![0-9.])219 MB|(?<![0-9.])218 MB", "LFM350 size understated"),
        (r"697 MB", "LFM1.2 size understated"),
        (r"1\.46 GB", "LFM2.6 size understated"),
    ]
    for path in list(ROOT.glob("*.md")) + list((ROOT / "docs").glob("*.md")):
        rel = path.relative_to(ROOT).as_posix()
        if rel in skip:
            continue
        text = path.read_text(encoding="utf-8")
        for pat, msg in stale:
            if re.search(pat, text):
                err(f"{rel}: stale {msg}")

    # corrected sizes present
    for path, needles in [
        ("docs/model-selection.md", ["~229 MB", "~731 MB", "~1.56 GB", "~533 MB", "~421 MB"]),
        ("docs/using-the-app.md", ["~229 MB", "~421 MB"]),
        ("README.md", ["~229 MB"]),
    ]:
        text = (ROOT / path).read_text(encoding="utf-8")
        for n in needles:
            if n not in text:
                err(f"{path} missing {n}")
    good("catalogue size headlines match manifest (±1 MB rounding)")

    # size vs manifest numerical closeness for LFM/Qwen
    for name, claimed_mb in [
        ("lfm25-350m", 229),
        ("lfm25-1.2b-instruct", 731),
        ("qwen35-0.8b", 533),
        ("smollm2-360m-cpu-int4", 421),
    ]:
        actual = sizes[name] / 1e6
        if abs(actual - claimed_mb) > 1.0:
            err(f"manifest {name}={actual:.1f} MB vs claimed ~{claimed_mb}")

    # --- evidence files ---
    p10 = json.loads(
        (ROOT / "bench/results/phase10-console-devtrain-result.json").read_text(encoding="utf-8")
    )
    if not p10.get("success") or p10.get("peak_ws_mb") != 1195:
        err(f"phase10 evidence unexpected: {p10}")
    else:
        good("phase10 peak_ws_mb=1195 success")

    h9: dict[str, list[bool]] = defaultdict(list)
    with (ROOT / "bench/results/phase7-h9.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            h9[row["model"]].append(bool(row["pass"]))
    expect_h9 = {
        "lfm25-1.2b-instruct": (6, 8),
        "lfm2-2.6b": (7, 8),
        "lfm25-350m": (4, 8),
        "llama32-3b": (5, 8),
        "gemma4-e2b": (6, 8),
    }
    for model, (a, b) in expect_h9.items():
        s, n = sum(h9[model]), len(h9[model])
        if (s, n) != (a, b):
            err(f"H9 {model}={s}/{n} expected {a}/{b}")
    good("H9 scores match docs")

    # --- links ---
    def check_links(md: Path, base: Path) -> list[str]:
        broken: list[str] = []
        for link in re.findall(r"\]\(([^)]+)\)", md.read_text(encoding="utf-8")):
            if link.startswith(("http", "#", "mailto:")):
                continue
            target = link.split("#")[0]
            if not target:
                continue
            p = (base / target)
            if not p.exists() and not (ROOT / target).exists():
                broken.append(link)
        return broken

    for label, path, base in [
        ("README", ROOT / "README.md", ROOT),
        ("docs/README", ROOT / "docs/README.md", ROOT / "docs"),
        ("architecture", ROOT / "docs/architecture.md", ROOT / "docs"),
    ]:
        broken = check_links(path, base)
        if broken:
            err(f"{label} broken links: {broken[:8]}")
        else:
            good(f"{label} links OK")

    # --- CI gates ---
    wf = (ROOT / ".github/workflows/build-linux.yml").read_text(encoding="utf-8")
    if "clang-format" not in wf:
        warn("build-linux.yml missing clang-format")
    if "generate-benchmark-summary.py --check" not in wf:
        err("build-linux.yml must run generate-benchmark-summary.py --check")
    else:
        good("CI enforces benchmark summary")

    # --- report ---
    print("=" * 60)
    print(f"OK {len(ok)}  WARN {len(warnings)}  ERR {len(errors)}")
    print("=" * 60)
    for msg in ok:
        print("  ✓", msg)
    for msg in warnings:
        print("  !", msg)
    for msg in errors:
        print("  ✗", msg)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
