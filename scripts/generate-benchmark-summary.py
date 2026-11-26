#!/usr/bin/env python3
"""Generate benchmark summaries from committed raw result files."""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "bench" / "results"
POLICY = ROOT / "bench" / "benchmark-summary.json"
MARKDOWN = ROOT / "docs" / "benchmarks.md"
CHART = ROOT / "docs" / "benchmarks-charts.html"

MD_BEGIN = "<!-- BEGIN GENERATED MODEL SUMMARY -->"
MD_END = "<!-- END GENERATED MODEL SUMMARY -->"
JS_BEGIN = "  // BEGIN GENERATED BENCHMARK DATA"
JS_END = "  // END GENERATED BENCHMARK DATA"
STD_BEGIN = "<!-- BEGIN GENERATED STANDARD BENCHMARKS -->"
STD_END = "<!-- END GENERATED STANDARD BENCHMARKS -->"
STD_TASK_LABELS = {
    "hellaswag": "HellaSwag",
    "winogrande": "WinoGrande",
    "arc-challenge": "ARC-Challenge",
    "arc-easy": "ARC-Easy",
    "mmlu": "MMLU",
}


FAILED_HOST_MARKERS = ("-timeout", "-stall", "-hung")


def _is_failed_host(row: dict[str, str]) -> bool:
    host = row.get("host") or ""
    return any(marker in host for marker in FAILED_HOST_MARKERS)


def load_csv(name: str) -> list[dict[str, str]]:
    # `name` is relative to bench/results/ and may include a campaign subdirectory.
    path = RESULTS / name
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            row
            for row in csv.DictReader(handle)
            if row.get("model", "").strip() and not _is_failed_host(row)
        ]


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        raise ValueError("median of empty sequence")
    mid = n // 2
    return ordered[mid] if n % 2 else (ordered[mid - 1] + ordered[mid]) / 2


def _is_repeat(row: dict) -> bool:
    # W1.1: a run_index column present with a value other than 0 marks a recorded
    # repetition of one configuration. Legacy rows (written before the column
    # existed) have no run_index and are single measurements.
    return (row.get("run_index") or "0").strip() not in ("", "0")


def resolve_rows(policy: dict) -> list[dict]:
    resolved = []
    for spec in policy["rows"]:
        matches = [
            row
            for row in load_csv(spec["source"])
            if all(row.get(key) == value for key, value in spec["selector"].items())
        ]
        if not matches:
            raise ValueError(f"no row matches {spec['source']}: {spec['selector']}")

        # W1.1: if the selector resolves to two or more recorded repeats, report the
        # median and min-max spread across them instead of a single pre-averaged
        # point. A newer repeated measurement supersedes any legacy rows for the
        # same selector (reps carry a run_index; legacy rows do not), so mixing
        # epochs is not possible. Rows without run_index keep the historical
        # single-row selection below and are marked single-run.
        reps = [row for row in matches if _is_repeat(row)]
        if len(reps) >= 2:
            decodes = [float(row["decode_tok_s"]) for row in reps]
            resolved.append(
                {
                    **spec,
                    "quant": spec.get("quant_label", reps[0]["quant"]),
                    "prefill": _median([float(row["prompt_tok_s"]) for row in reps]),
                    "decode": _median(decodes),
                    "ram": round(_median([float(row["peak_ws_mb"]) for row in reps])),
                    "load_ms": round(_median([float(row["load_ms"]) for row in reps])),
                    "date": max(row["date"] for row in reps),
                    "spread": (min(decodes), max(decodes), len(reps)),
                    "single_run": False,
                }
            )
            continue

        select = spec.get("select", "only")
        if select == "only" and len(matches) != 1:
            raise ValueError(
                f"expected one row in {spec['source']} for {spec['selector']}, got {len(matches)}"
            )
        if select == "max_decode":
            row = max(matches, key=lambda item: float(item["decode_tok_s"]))
        elif select == "only":
            row = matches[0]
        else:
            raise ValueError(f"unknown selection policy: {select}")
        resolved.append(
            {
                **spec,
                "quant": spec.get("quant_label", row["quant"]),
                "prefill": float(row["prompt_tok_s"]),
                "decode": float(row["decode_tok_s"]),
                "ram": int(row["peak_ws_mb"]),
                "load_ms": int(row["load_ms"]),
                "date": row["date"],
                "spread": None,
                "single_run": True,
            }
        )
    return resolved


def resolve_quality(policy: dict) -> list[dict]:
    quality = policy["quality"]
    scores: dict[str, list[bool]] = defaultdict(list)
    tasks: dict[str, set[str]] = defaultdict(set)
    with (RESULTS / quality["source"]).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if row["task_id"] in tasks[row["model"]]:
                    raise ValueError(
                        f"duplicate quality result: {row['model']} / {row['task_id']}"
                    )
                tasks[row["model"]].add(row["task_id"])
                scores[row["model"]].append(bool(row["pass"]))
    declared = set(quality["labels"])
    measured = set(scores)
    if declared != measured:
        raise ValueError(
            f"quality model set differs: missing={sorted(declared - measured)}, "
            f"undeclared={sorted(measured - declared)}"
        )
    task_sets = list(tasks.values())
    if any(task_set != task_sets[0] for task_set in task_sets[1:]):
        raise ValueError("quality models were not evaluated on the same task set")
    return [
        {
            "model": label,
            "score": sum(scores[model]),
            "total": len(scores[model]),
            "src": quality["source"].removesuffix(".jsonl"),
        }
        for model, label in quality["labels"].items()
    ]


def _read_jsonl(name: str) -> list[dict]:
    with (RESULTS / name).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def resolve_standard(policy: dict) -> dict:
    """Informational standard benchmarks (#597). H9 stays the promotion gate.

    Every declared model needs every declared loglikelihood task exactly once,
    with the GGUF hash the policy pins. A generative source holds one model and
    every declared task, each complete (not a --limit run, item rows equal to
    the summary count). Anything else fails here instead of printing.
    """
    std = policy["standard"]
    models = std["models"]
    by_gguf = {spec["gguf"]: model for model, spec in models.items()}

    ll_tasks = std["loglikelihood"]["tasks"]
    ll: dict[tuple[str, str], dict] = {}
    for row in _read_jsonl(std["loglikelihood"]["source"]):
        model = by_gguf.get(row["model"])
        if model is None:
            raise ValueError(f"standard: undeclared GGUF {row['model']}")
        if models[model]["gguf_sha256"] != row["model_sha256"]:
            raise ValueError(f"standard: {row['model']} sha256 differs from the policy")
        if row["task"] not in ll_tasks:
            raise ValueError(f"standard: undeclared task {row['task']}")
        if (model, row["task"]) in ll:
            raise ValueError(f"standard: duplicate result {model} / {row['task']}")
        ll[(model, row["task"])] = row
    missing = [f"{m}/{t}" for m in models for t in ll_tasks if (m, t) not in ll]
    if missing:
        raise ValueError(f"standard: missing loglikelihood results: {', '.join(missing)}")

    gen_tasks = std["generative"]["tasks"]
    gen = []
    for source in std["generative"]["sources"]:
        rows = _read_jsonl(source)
        summaries = {}
        for row in rows:
            if row.get("type") != "summary":
                continue
            if row["model"] not in models or row["task"] not in gen_tasks:
                raise ValueError(f"standard: undeclared {row['model']} / {row['task']} in {source}")
            if row["partial"]:
                raise ValueError(f"standard: partial run in {source}: {row['task']}")
            if row["task"] in summaries:
                raise ValueError(f"standard: duplicate {row['task']} in {source}")
            items = sum(1 for r in rows if r.get("type") == "item" and r["task"] == row["task"])
            if items != row["n_items"]:
                raise ValueError(
                    f"standard: {source} {row['task']}: {items} items, summary says {row['n_items']}"
                )
            summaries[row["task"]] = row
        if set(summaries) != set(gen_tasks) or len({r["model"] for r in summaries.values()}) != 1:
            raise ValueError(f"standard: {source} must hold one model and the tasks {gen_tasks}")
        gen.append({"source": source, **summaries})
    return {
        "models": models,
        "ll": ll,
        "ll_tasks": ll_tasks,
        "ll_source": std["loglikelihood"]["source"],
        "gen": gen,
    }


def standard_markdown(std: dict) -> str:
    models, tasks = std["models"], std["ll_tasks"]
    lines = [
        STD_BEGIN,
        "<!-- Generated by scripts/generate-benchmark-summary.py; do not edit by hand. -->",
        "| Model | " + " | ".join(STD_TASK_LABELS.get(t, t) for t in tasks) + " | Hardware |",
        "| --- | " + " | ".join("---:" for _ in tasks) + " | --- |",
    ]
    for model, spec in models.items():
        cells = [
            f"{std['ll'][(model, t)]['acc']:.1f} ± {std['ll'][(model, t)]['stderr']:.1f}"
            for t in tasks
        ]
        hardware = std["ll"][(model, tasks[0])]["hardware"]
        lines.append(f"| {spec['label']} | " + " | ".join(cells) + f" | {hardware} |")
    chance = []
    for task in tasks:
        values = {std["ll"][(m, task)]["chance"] for m in models}
        chance.append("—" if None in values else f"{min(values):.0f}")
    lines += [
        "| _chance_ | " + " | ".join(chance) + " | |",
        "",
        f"Accuracy % ± standard error. Source: `{std['ll_source'].removesuffix('.jsonl')}`.",
        "",
        "| Model | Endpoint | Hardware | IFEval strict | IFEval instructions | GSM8K "
        "| Length stops | Source |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for run in std["gen"]:
        ife, gsm = run["ifeval"], run["gsm8k"]
        lines.append(
            f"| {models[ife['model']]['label']} | `{ife['endpoint']}` | {ife['hardware']} "
            f"| {ife['passed']}/{ife['n_items']} ± {ife['stderr']:.1f} "
            f"| {ife['instruction_acc']:.1f} % "
            f"| {gsm['passed']}/{gsm['n_items']} ± {gsm['stderr']:.1f} "
            f"| {ife['length_stops'] + gsm['length_stops']} "
            f"| `{run['source'].removesuffix('.jsonl')}` |"
        )
    lines.append(STD_END)
    return "\n".join(lines)


def max_kv_reuse(policy: dict) -> float:
    values = []
    for source in policy["kv_reuse_sources"]:
        for row in load_csv(source):
            if row.get("speedup"):
                values.append(float(row["speedup"]))
    if not values:
        raise ValueError("no KV-reuse speedup values found")
    return max(values)


def markdown_table(rows: list[dict]) -> str:
    lines = [
        MD_BEGIN,
        "<!-- Generated by scripts/generate-benchmark-summary.py; do not edit by hand. -->",
        "| Model | Params | Quant | Backend | Prefill tok/s | Decode tok/s "
        "| Decode min–max | Peak RAM MiB | Source CSV |",
        "| --- | --- | --- | --- | ---: | ---: | :---: | ---: | --- |",
    ]
    for row in sorted(rows, key=lambda item: item["decode"], reverse=True):
        # W1.1: single-run rows are marked as such rather than presented like
        # repeated ones; repeated rows carry their decode min–max and run count so
        # the reader can judge the median against the spread it came from.
        if row.get("single_run", True):
            spread_cell = "_single run_"
        else:
            lo, hi, n = row["spread"]
            spread_cell = f"{lo:.1f}–{hi:.1f} (n={n})"
        lines.append(
            f"| {row['label']} | {row['params']} | {row['quant']} | {row['backend_label']} "
            f"| {row['prefill']:.1f} | **{row['decode']:.1f}** | {spread_cell} | {row['ram']} "
            f"| `{row['source'].removesuffix('.csv')}` |"
        )
    lines.append(MD_END)
    return "\n".join(lines)


def js_data(rows: list[dict], quality: list[dict], kv_reuse: float) -> str:
    payload = [
        {
            "model": row["label"],
            "quant": row["quant"],
            "be": row["backend_key"],
            "prefill": round(row["prefill"], 2),
            "decode": round(row["decode"], 2),
            "ram": row["ram"],
            "src": row["source"].removesuffix(".csv"),
            **({"tag": row["tag"]} if row.get("tag") else {}),
        }
        for row in rows
    ]
    latest = max(row["date"] for row in rows)[:10]
    return "\n".join(
        [
            JS_BEGIN,
            "  // Generated by scripts/generate-benchmark-summary.py; do not edit by hand.",
            f"  const DATA = {json.dumps(payload, indent=2)};",
            f"  const QUALITY = {json.dumps(quality, indent=2)};",
            f"  const MAX_KV_REUSE = {kv_reuse:.2f};",
            f'  const LATEST_RUN = "{latest}";',
            JS_END,
        ]
    )


def replace_block(text: str, begin: str, end: str, replacement: str) -> str:
    start = text.find(begin)
    finish = text.find(end, start + len(begin))
    if start < 0 or finish < 0:
        raise ValueError(f"generated markers not found: {begin} / {end}")
    return text[:start] + replacement + text[finish + len(end) :]


def update(path: Path, expected: str, check: bool) -> bool:
    current = path.read_text(encoding="utf-8")
    if current == expected:
        return True
    if check:
        print(f"stale generated benchmark output: {path.relative_to(ROOT)}", file=sys.stderr)
        print(
            "".join(
                difflib.unified_diff(
                    current.splitlines(keepends=True),
                    expected.splitlines(keepends=True),
                    fromfile=str(path),
                    tofile=f"{path} (generated)",
                    n=2,
                )
            ),
            file=sys.stderr,
        )
        return False
    path.write_text(expected, encoding="utf-8")
    print(f"updated {path.relative_to(ROOT)}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if generated outputs are stale")
    args = parser.parse_args()

    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    if policy.get("schema_version") != 1:
        raise ValueError("unsupported benchmark summary schema")
    rows = resolve_rows(policy)
    quality = resolve_quality(policy)
    kv_reuse = max_kv_reuse(policy)
    standard = resolve_standard(policy)

    markdown = MARKDOWN.read_text(encoding="utf-8")
    markdown = replace_block(markdown, MD_BEGIN, MD_END, markdown_table(rows))
    markdown = replace_block(markdown, STD_BEGIN, STD_END, standard_markdown(standard))
    chart = CHART.read_text(encoding="utf-8")
    chart = replace_block(chart, JS_BEGIN, JS_END, js_data(rows, quality, kv_reuse))

    ok = update(MARKDOWN, markdown, args.check)
    ok = update(CHART, chart, args.check) and ok
    if args.check and not ok:
        print("run: python3 scripts/generate-benchmark-summary.py", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
