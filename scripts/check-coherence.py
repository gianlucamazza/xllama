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
    manifest = json.loads(
        (ROOT / "uwp/models/manifest.json").read_text(encoding="utf-8")
    )
    cat = {e["name"]: e for e in manifest["models"]}
    good(f"manifest models: {len(cat)}")

    sizes = {
        n: sum(f.get("approx_bytes") or 0 for f in e.get("files", []))
        for n, e in cat.items()
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
        # Derive the expected version from the manifest instead of hardcoding
        # it: a pinned literal here goes stale at the next release and the
        # warning it emits is about itself, not about the docs.
        roadmap_head = (ROOT / "ROADMAP.md").read_text(encoding="utf-8")[:2000]
        if f"**{ver.group(1)}.0**" not in roadmap_head:
            warn(f"Appx semantic {ver.group(1)} not in the ROADMAP head (bump it)")
        else:
            good(f"ROADMAP states the shipping version {ver.group(1)}.0")

        # --- the published demo is what the capture actually produced ---
        #
        # The README linked a v1.2.0 demo while the product shipped 1.5.2, and
        # docs/launch-copy.md forbade citing that very file as current. Nothing
        # caught it because nothing was looking. Two rules, both cheap:
        #
        #   ERR  — the README cites a version the demo manifest does not claim.
        #          That is someone editing the link by hand, or a capture that
        #          never happened, and it is unambiguous either way.
        #   WARN — the demo is two or more minors behind the manifest. That is
        #          drift rather than error: a release can legitimately ship
        #          before someone re-records, but not indefinitely.
        #
        # demo-manifest.json is written by scripts/capture-demo-video.sh, so the
        # ERR case cannot be satisfied by editing prose — only by capturing.
        demo_manifest = ROOT / "docs/screenshots/demo-manifest.json"
        readme_text = (ROOT / "README.md").read_text(encoding="utf-8")
        # The cited version is read off the ARTEFACT the README references, not
        # off a prose label: `xllama-demo-v1.5.2.gif` / `.mp4`. The earlier form
        # parsed "**Demo:** ... (v1.5.2)", which broke the moment the link became
        # an embedded image — and a bold label is a weaker anchor anyway, since
        # it can agree with the manifest while pointing at another file.
        cited = re.search(r"xllama-demo-v(\d+\.\d+\.\d+)\.(?:gif|mp4)", readme_text)
        if not demo_manifest.exists():
            if cited:
                err(
                    f"README cites a v{cited.group(1)} demo but "
                    "docs/screenshots/demo-manifest.json does not exist — "
                    "regenerate it with scripts/capture-demo-video.sh"
                )
            else:
                good("no demo manifest and no demo claim in the README")
        else:
            dm = json.loads(demo_manifest.read_text(encoding="utf-8"))
            dm_ver = str(dm.get("version", ""))
            dm_short = ".".join(dm_ver.split(".")[:3])
            if not cited:
                err(
                    "docs/screenshots/demo-manifest.json exists but README cites no demo"
                )
            elif cited.group(1) != dm_short:
                err(
                    f"README cites demo v{cited.group(1)}, manifest recorded "
                    f"v{dm_short} — the link was edited by hand, or the capture "
                    "was never run"
                )
            else:
                # ...and the link must point at the file the capture produced,
                # not merely at a matching version. Nearly shipped: the README
                # was updated, the version agreed, and the URL pointed at an
                # asset that had never been uploaded — a 404 on the front page.
                # Whether the asset EXISTS is a network fact this check cannot
                # know by design (it runs offline); that the filename agrees is
                # the half that can be checked here.
                dm_file = str(dm.get("file", ""))
                if dm_file and dm_file not in readme_text:
                    err(
                        f"README demo link does not reference {dm_file}, the file "
                        "the capture produced"
                    )
                else:
                    good(f"README demo link matches the captured demo (v{dm_short})")
                # The GIF is the half that CAN be fully verified offline, and it
                # is the one readers actually see: an .mp4 linked from a release
                # renders as a plain link on GitHub, so the GIF is what makes the
                # demo play on the landing page. It is committed rather than
                # uploaded, so unlike the video its existence is a fact about
                # this tree — check the file is there AND that the README embeds
                # it, because either half alone still ships a broken image.
                #
                # A missing "gif" key is an error, not a skip. The manifest is
                # machine-written and always carries one, so its absence means
                # the manifest predates the GIF or was hand-edited — and a guard
                # that quietly does nothing in that case is no guard at all.
                dm_gif = str(dm.get("gif", ""))
                if not dm_gif:
                    err(
                        "demo-manifest.json has no 'gif' key — re-run "
                        "scripts/capture-demo-video.sh rather than editing it"
                    )
                else:
                    gif_path = ROOT / "docs/screenshots" / dm_gif
                    if not gif_path.exists():
                        err(
                            f"demo-manifest.json records {dm_gif} but "
                            f"docs/screenshots/{dm_gif} is not in the tree"
                        )
                    # Require the image SYNTAX, not the filename anywhere in the
                    # file. A fault-injection run passed this check with the
                    # embed commented out, because the path was still present as
                    # text — the string test proved the name was mentioned, not
                    # that anything renders.
                    elif not re.search(
                        r"!\[[^\]]*\]\(docs/screenshots/" + re.escape(dm_gif) + r"\)",
                        re.sub(r"<!--.*?-->", "", readme_text, flags=re.S),
                    ):
                        err(
                            f"docs/screenshots/{dm_gif} exists but the README "
                            "does not embed it as an image — the demo does not "
                            "play on the landing page"
                        )
                    else:
                        kb = gif_path.stat().st_size // 1000
                        good(f"README embeds the demo GIF ({dm_gif}, {kb} kB)")
                cur = [int(x) for x in ver.group(1).split(".")]
                got = [int(x) for x in dm_short.split(".")]
                # Distance in minors, treating a major bump as far behind.
                behind = (cur[0] - got[0]) * 100 + (cur[1] - got[1])
                if behind >= 2:
                    warn(
                        f"the demo is v{dm_short} against a shipping "
                        f"{ver.group(1)} — re-record before it is cited as current"
                    )
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
    if "StartPersonalizeTrain" not in (ROOT / "uwp/MainPage.h").read_text(
        encoding="utf-8"
    ):
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
        for m in re.finditer(
            r"scripts/([A-Za-z0-9_./-]+\.(?:sh|py|ps1))", path.read_text()
        ):
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
        # training/jobs/ holds only TrainingJob JSON (manifest overrides live
        # under training/manifest-overrides/ — not validated as jobs).
        for job in sorted((ROOT / "training/jobs").glob("*.json")):
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

    # --- the two build systems list the same bridge sources ---
    #
    # CMakeLists.txt builds the host library and the tests; uwp/xllama.vcxproj
    # builds the console app. They share src/bridge/ but keep separate lists, so
    # adding a source to one and not the other compiles and then fails at LINK —
    # on CI, on Windows, twenty minutes later. That is exactly what happened when
    # autopilot.cpp was added; the local host build was green throughout.
    #
    # Divergence is allowed only where it is deliberate, and the reason belongs
    # here rather than in someone's memory.
    host_only = {
        "cli.cpp",  # xllama-cli argument parsing; the app has no command line
    }
    cmake_srcs = set(
        re.findall(
            r"src/bridge/([a-z_0-9]+\.cpp)", (ROOT / "CMakeLists.txt").read_text()
        )
    )
    vcx_srcs = set(
        re.findall(
            r"src\\bridge\\([a-z_0-9]+\.cpp)",
            (ROOT / "uwp/xllama.vcxproj").read_text(),
        )
    )
    missing_uwp = sorted(cmake_srcs - vcx_srcs - host_only)
    missing_host = sorted(vcx_srcs - cmake_srcs)
    for s in missing_uwp:
        err(
            f"src/bridge/{s} is in CMakeLists.txt but not uwp/xllama.vcxproj (link error on CI)"
        )
    for s in missing_host:
        err(
            f"src/bridge/{s} is in uwp/xllama.vcxproj but not CMakeLists.txt (untested on host)"
        )
    stale_exempt = sorted(host_only - cmake_srcs)
    for s in stale_exempt:
        err(f"host_only lists {s}, which CMakeLists.txt no longer builds")
    if not missing_uwp and not missing_host and not stale_exempt:
        good(
            f"bridge sources agree across both build systems ({len(cmake_srcs & vcx_srcs)} shared)"
        )

    # --- autopilot ops: the validator's table and the driver's branches ---
    #
    # An op has to exist in two places that cannot see each other: kOps in
    # autopilot.cpp, which decides what a script may say, and the dispatch chain
    # in ApRun, which decides what happens. One without the other is a real
    # failure with a confusing shape — a script that validates and then dies at
    # run time, or a documented op no script may use.
    #
    # The list is READ from the validator rather than repeated here. Hardcoding
    # it made this check a third copy, and a third copy drifts like the other
    # two: `mark` had to be added to it by hand.
    apcpp = (ROOT / "src/bridge/autopilot.cpp").read_text(encoding="utf-8")
    table = re.search(r"kOps\s*=\s*\{(.*?)\};", apcpp, re.S)
    if not table:
        err("autopilot.cpp: kOps table not found")
    else:
        ops = set(re.findall(r'"([a-z_]+)"', table.group(1)))
        if not ops:
            err("autopilot.cpp: kOps table is empty")
        mpcpp = (ROOT / "uwp/MainPage.cpp").read_text(encoding="utf-8")
        missing = [o for o in sorted(ops) if f'a.op == "{o}"' not in mpcpp]
        for o in missing:
            err(
                f"autopilot op {o} is accepted by the validator but ApRun has no branch"
            )
        # ...and the other direction: a branch for an op the validator rejects
        # is unreachable code that looks supported.
        branches = set(re.findall(r'a\.op == "([a-z_]+)"', mpcpp))
        for o in sorted(branches - ops):
            err(
                f"ApRun has a branch for '{o}', which validate_autopilot_script rejects"
            )
        if not missing and not (branches - ops):
            good(
                f"autopilot ops ({len(ops)}) — validator table and ApRun branches agree"
            )

    # --- console gates: what runs vs what the runbook describes ---
    #
    # Same shape as the autopilot-ops check above, one axis over. The gates are
    # the release contract (cite N/N PASS with N = len(gates) in the script) —
    # but the runbook is the only place that says what each one ASSERTS, and it
    # had drifted to describing four of nine. A gate that fails without a written
    # contract sends the operator to read shell.
    #
    # Read from the script, not repeated here, for the same reason as kOps.
    vc = (ROOT / "scripts/validate-console.sh").read_text(encoding="utf-8")
    gates = set(re.findall(r"^(\w+)\) run_gate ", vc, re.M))
    runbook = (ROOT / "docs/console-validation-runbook.md").read_text(encoding="utf-8")
    # Scope the parse to the gate list itself — from the sentence that introduces
    # it to the next heading. The file has other bolded bullet lists (the
    # failure-class breakdown below it, whose leads are "autopilot" / "marker" /
    # "timeout"), and reading the whole document takes those for gate names.
    section = re.search(r"hardware gates pass:\n(.*?)\n#", runbook, re.S)
    if not section:
        err(
            "console-validation-runbook.md: the sentence introducing the gate list moved"
        )
    section_text = section.group(1) if section else ""
    # `gguf` is written "GGUF chat" and `taesd` "TAESD", so match case-insensitively
    # on the bolded lead word rather than requiring the shell identifier verbatim.
    documented = {
        m.lower() for m in re.findall(r"^- \*\*([A-Za-z]+)", section_text, re.M)
    }
    if not gates:
        err("validate-console.sh: no `<name>) run_gate` arms found")
    else:
        undocumented = sorted(g for g in gates if g.lower() not in documented)
        for g in undocumented:
            err(
                f"console gate '{g}' runs but docs/console-validation-runbook.md does not describe it"
            )
        # The other direction: a gate described in the runbook that no longer
        # exists sends an operator to run something that will just print usage.
        # `api` is documented on purpose and lives in validate-api.sh.
        doc_only = sorted(
            d
            for d in documented
            if d not in {g.lower() for g in gates} and d not in {"api"}
        )
        for d in doc_only:
            err(
                f"the runbook describes a '{d}' gate that validate-console.sh does not run"
            )
        if not undocumented and not doc_only:
            good(
                f"console gates ({len(gates)}) — runbook describes every gate that runs"
            )

    # --- every tracked top-level directory appears in the AGENTS.md map ---
    #
    # AGENTS.md calls itself the file-level map for agents and contributors, so a
    # directory missing from it is invisible to exactly the readers it is for.
    # `demo/` arrived with the capture pipeline and was absent from both trees.
    #
    # The exemptions are deliberate omissions, and they carry their reason here
    # rather than in someone's memory (same rule as `cli.cpp` above).
    # (`llama.cpp` needs no exemption: it is a submodule, so `git ls-files`
    # reports it as one gitlink entry with no path separator and it never reaches
    # this set. Adding it anyway is what the stale-exemption check below caught.)
    tree_exempt = {
        "vendor",  # patched-DLL overlay, owned by vendor-lifecycle-plan.md
        ".github",  # CI, described in the workflows section instead of the tree
    }
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    r = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    top_dirs = {
        p.split("/", 1)[0] for p in r.stdout.splitlines() if "/" in p
    } - tree_exempt
    missing_dirs = sorted(d for d in top_dirs if f"{d}/" not in agents)
    for d in missing_dirs:
        err(f"tracked directory {d}/ does not appear in the AGENTS.md map")
    stale_tree_exempt = sorted(
        d
        for d in tree_exempt
        if d not in {p.split("/", 1)[0] for p in r.stdout.splitlines() if "/" in p}
    )
    for d in stale_tree_exempt:
        err(f"tree_exempt lists {d}/, which is no longer a tracked directory")
    if not missing_dirs and not stale_tree_exempt:
        good(f"AGENTS.md maps every tracked top-level directory ({len(top_dirs)})")

    # --- benchmark summary ---
    r = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/generate-benchmark-summary.py"),
            "--check",
        ],
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
        # 2026-07-25 baseline: LFM decode 94.2 -> 93.0 (post-repack multi-run
        # median, phase13-repack-after.csv) and ORT CPU 68.0 -> 74.8 (shipped
        # t6 asset, t6-shipped-confirm.csv).
        # 2026-07-26: LFM 93.0 -> 94.9 (post-#168 n_threads_batch median,
        # phase13b-threadsbatch-after.csv).
        for num in ("94.9", "37.9", "18.4", "74.8", "44.4", "20.6", "35.1", "236.7"):
            if num not in block:
                err(f"generated table missing {num}")
            if num not in rec and num not in ("236.7",):
                err(f"recommended-config missing headline {num}")
        good("decode headlines code-doc aligned")

    # --- model-matrix numbers vs the CSV they claim to come from ---
    # docs/benchmarks.md is the numbers SSOT and model-matrix.md is the status
    # SSOT, but the phase14 inventory table restates prefill/decode/peak. A second
    # copy of a number is only debt when nothing checks the two agree — the same
    # class of drift that left "console pending" in manifest comments after the
    # console runs had landed. So: check it.
    mm_path = ROOT / "docs/model-matrix.md"
    csv_path = ROOT / "bench/results/phase14-console.csv"
    if mm_path.exists() and csv_path.exists():
        rows: dict[str, list[dict[str, str]]] = {}
        lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
        head = lines[0].split(",")
        for line in lines[1:]:
            cells = dict(zip(head, line.split(",")))
            rows.setdefault(cells["model"], []).append(cells)

        def median(vals: list[float]) -> float:
            vals = sorted(vals)
            mid = len(vals) // 2
            return vals[mid] if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2

        # A metrics row carries at least three numbers; the role/status tables in
        # the same file carry none and are not this check's business.
        covered: set[str] = set()
        for line in mm_path.read_text(encoding="utf-8").splitlines():
            m = re.match(r"\|[^|]+\|\s*`([a-z0-9.-]+)`\s*\|", line)
            if not m or m.group(1) not in rows:
                continue
            model = m.group(1)
            cells = [c.strip().strip("*") for c in line.strip("|").split("|")]
            nums = [float(c) for c in cells if re.fullmatch(r"[0-9]+(\.[0-9]+)?", c)]
            if len(nums) < 3:
                continue  # not a metrics row
            runs = rows[model]
            want_prefill = median([float(r["prompt_tok_s"]) for r in runs])
            want_decode = median([float(r["decode_tok_s"]) for r in runs])
            want_peak = float(max(int(r["peak_ws_mb"]) for r in runs))
            missing = [
                f"{label} {want:.1f}"
                for label, want in (
                    ("prefill", want_prefill),
                    ("decode", want_decode),
                    ("peak", want_peak),
                )
                if not any(abs(c - want) <= 0.15 for c in nums)
            ]
            if missing:
                err(
                    f"model-matrix {model}: row does not match phase14-console.csv "
                    f"({', '.join(missing)} absent from {nums})"
                )
            else:
                covered.add(model)
        absent = sorted(set(rows) - covered)
        if absent:
            err(f"model-matrix: no verified metrics row for {', '.join(absent)}")
        else:
            good(
                f"model-matrix numbers match phase14-console.csv ({len(covered)} models)"
            )

    # --- every catalogue model is documented, and no doc invents one ---
    # model-matrix.md is the status SSOT, and it grew one table per campaign (A1
    # with Role/n_ctx/Template, A2 with Status). A reader has to union them, and a
    # new catalogue entry can simply be missing — which is how the phase14 models
    # would have shipped undocumented. Coverage in both directions, plus the two
    # policy fields that change behaviour (role, n_ctx).
    if mm_path.exists():
        mm_text = mm_path.read_text(encoding="utf-8")
        text_models = {n: e for n, e in cat.items() if e.get("kind") != "diffusion"}
        # Rows mentioning each id, so the field checks look only where the id is.
        rows_for: dict[str, list[str]] = {n: [] for n in text_models}
        for line in mm_text.splitlines():
            if not line.startswith("|"):
                continue
            for n in text_models:
                if f"`{n}`" in line:
                    rows_for[n].append(line)

        undocumented = sorted(n for n, r in rows_for.items() if not r)
        if undocumented:
            err(
                "model-matrix: catalogue models absent from the inventory: "
                + ", ".join(undocumented)
            )
        else:
            good(f"model-matrix documents all {len(text_models)} catalogue text models")

        # The reverse direction: every OTHER backticked token in the inventory is
        # either a documented status label or a pointer to evidence — and a pointer
        # that no longer resolves is drift with a straight face. (An earlier
        # version of this check just warned about "unknown ids" and flagged twelve
        # legitimate ones; a check that cries wolf gets ignored.)
        status_vocab = (
            set(
                re.findall(
                    r"`([a-z-]+)`", re.search(r"\*\*Status\*\*.*", mm_text).group(0)
                )
            )
            if re.search(r"\*\*Status\*\*.*", mm_text)
            else set()
        )
        claimed = set(re.findall(r"`([a-z][a-z0-9.]*(?:-[a-z0-9.]+)+)`", mm_text))
        dangling, evidence_ok = [], 0
        for tok in sorted(claimed - set(cat) - status_vocab):
            if tok.endswith(
                (".md", ".csv", ".json", ".jsonl", ".txt", ".py", ".sh", ".h", ".cpp")
            ):
                continue
            if "_" in tok:  # code identifiers (strip_thinking_content, ...)
                continue
            if (ROOT / f"bench/results/{tok}.csv").exists() or (
                ROOT / f"bench/prompts/{tok}.txt"
            ).exists():
                evidence_ok += 1
                continue
            dangling.append(tok)
        if dangling:
            err(
                "model-matrix: backticked tokens that are neither a catalogue id, a "
                "status label, nor resolvable evidence: " + ", ".join(dangling)
            )
        else:
            good(f"model-matrix evidence pointers resolve ({evidence_ok} files)")

        for n, e in text_models.items():
            rows = " ".join(rows_for.get(n, []))
            if not rows:
                continue
            if e.get("role") == "coding" and "coding" not in rows:
                err(
                    f"model-matrix {n}: catalogue role is coding, the row does not say so"
                )
            want_ctx = e.get("n_ctx") or 0
            if want_ctx and str(want_ctx) not in rows:
                err(
                    f"model-matrix {n}: catalogue n_ctx {want_ctx} missing from the row"
                )

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
        (
            "docs/model-selection.md",
            ["~229 MB", "~731 MB", "~1.56 GB", "~533 MB", "~421 MB"],
        ),
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
        (ROOT / "bench/results/phase10-console-devtrain-result.json").read_text(
            encoding="utf-8"
        )
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
            p = base / target
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
