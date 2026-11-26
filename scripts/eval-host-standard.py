#!/usr/bin/env python3
"""Standard loglikelihood benchmarks for a GGUF on the Linux host (#597).

Runs llama.cpp's `llama-perplexity` in its HellaSwag, WinoGrande and
multiple-choice (ARC, MMLU) modes over the datasets pinned in
bench/eval/standard-datasets.json and appends one JSONL row per task.

The lane is informational: it measures the weights and the quant, not the
console. H9 (bench/eval/phase7-h9.json) stays the promotion gate. The LAN API
returns no logprobs, so these tasks cannot run on the console.

Usage:
  ./scripts/eval-host-standard.py --fetch
  ./scripts/eval-host-standard.py --model path/to/model.gguf --hardware <label> \
      [--tasks hellaswag,winogrande,arc-challenge,arc-easy,mmlu] \
      --out bench/results/<date>-std-host.jsonl
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from eval_datasets import (  # noqa: E402
    DEFAULT_CACHE,
    EvalError,
    dataset_path,
    fetch,
    sha256_file,
    validate_datasets,
)
SPEC_PATH = ROOT / "bench/eval/standard-datasets.json"
DEFAULT_BIN = ROOT / "build/linux-release/bin/llama-perplexity"
MODES = ("hellaswag", "winogrande", "multiple-choice")

# llama-perplexity result lines (tools/perplexity/perplexity.cpp at the pin).
HELLASWAG_ROW = re.compile(r"^(\d+)\t([0-9.]+)%\t\[", re.M)
WINOGRANDE_FINAL = re.compile(r"Final Winogrande score\((\d+) tasks\): ([0-9.]+) \+/- ([0-9.]+)")
MC_ROW = re.compile(r"^(\d+)\t([0-9.]+)$", re.M)
MC_FINAL = re.compile(r"Final result: ([0-9.]+) \+/- ([0-9.]+)")
MC_CHANCE = re.compile(r"Random chance: ([0-9.]+) \+/- ([0-9.]+)")


def load_spec(path: Path = SPEC_PATH) -> dict:
    spec = json.loads(path.read_text(encoding="utf-8"))
    validate_spec(spec)
    return spec


def validate_spec(spec: dict) -> None:
    datasets = spec.get("datasets") or {}
    tasks = spec.get("tasks") or {}
    validate_datasets(datasets)
    if not tasks:
        raise EvalError("spec needs tasks")
    for name, task in tasks.items():
        if task.get("dataset") not in datasets:
            raise EvalError(f"task {name}: unknown dataset {task.get('dataset')!r}")
        if task.get("mode") not in MODES:
            raise EvalError(f"task {name}: mode must be one of {MODES}")
        if not isinstance(task.get("tasks"), int) or task["tasks"] < 0:
            raise EvalError(f"task {name}: tasks must be an int >= 0 (0 = all)")
        if not isinstance(task.get("n_ctx"), int) or task["n_ctx"] <= 0:
            raise EvalError(f"task {name}: n_ctx must be a positive int")
        # llama-perplexity scores every choice of a task in its own sequence; an ARC
        # task has up to 5 choices, and too few sequences aborts the run mid-way.
        if not isinstance(task.get("n_parallel"), int) or task["n_parallel"] < 4:
            raise EvalError(f"task {name}: n_parallel must be an int >= 4")


def binomial_stderr(p: float, n: int) -> float:
    """Same estimator llama-perplexity prints for WinoGrande and multiple choice."""
    if n < 2:
        return float("nan")
    return math.sqrt(p * (1.0 - p) / (n - 1))


def parse_output(mode: str, text: str) -> dict:
    """Accuracy in percent, its stderr in percent, and the task count."""
    if mode == "hellaswag":
        rows = HELLASWAG_ROW.findall(text)
        if not rows:
            raise EvalError("no HellaSwag result rows in output")
        n, acc = int(rows[-1][0]), float(rows[-1][1])
        return {"n_tasks": n, "acc": acc, "stderr": 100.0 * binomial_stderr(acc / 100.0, n)}
    if mode == "winogrande":
        m = WINOGRANDE_FINAL.search(text)
        if not m:
            raise EvalError("no 'Final Winogrande score' line in output")
        return {"n_tasks": int(m.group(1)), "acc": float(m.group(2)), "stderr": float(m.group(3))}
    if mode == "multiple-choice":
        m = MC_FINAL.search(text)
        rows = MC_ROW.findall(text)
        if not m or not rows:
            raise EvalError("no multiple-choice 'Final result' line in output")
        out = {"n_tasks": int(rows[-1][0]), "acc": float(m.group(1)), "stderr": float(m.group(2))}
        chance = MC_CHANCE.search(text)
        if chance:
            out["chance"] = float(chance.group(1))
        return out
    raise EvalError(f"unknown mode {mode}")


def hellaswag_task_count(data: Path) -> int:
    """llama-perplexity reads six lines per HellaSwag task."""
    with data.open(encoding="utf-8") as f:
        return sum(1 for _ in f) // 6


def perplexity_args(task: dict, data: Path) -> list[str]:
    mode = task["mode"]
    n = str(task["tasks"])
    if mode == "hellaswag":
        # --hellaswag-tasks 0 scores nothing upstream; 0 means "all" in the spec.
        if task["tasks"] == 0:
            n = str(hellaswag_task_count(data))
        return ["-f", str(data), "--hellaswag", "--hellaswag-tasks", n]
    if mode == "winogrande":
        return ["-f", str(data), "--winogrande", "--winogrande-tasks", n]
    return ["-bf", str(data), "--multiple-choice", "--multiple-choice-tasks", n]


def evaluator_version(binary: Path) -> str:
    """The build line the selected llama-perplexity prints, e.g. '0.6.0-dev (build 1, commit d81235049)'."""
    try:
        proc = subprocess.run([str(binary), "--version"], capture_output=True, text=True,
                              timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    for line in (proc.stdout + proc.stderr).splitlines():
        if line.startswith("version:"):
            return line.removeprefix("version:").strip()
    return ""


def llama_pin() -> str:
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "ls-tree", "HEAD", "llama.cpp"],
                             capture_output=True, text=True, check=True).stdout.split()
        return out[2] if len(out) >= 3 else ""
    except (OSError, subprocess.CalledProcessError):
        return ""


def run_task(args: argparse.Namespace, spec: dict, name: str, model_sha: str) -> dict:
    task = spec["tasks"][name]
    data = dataset_path(spec, task["dataset"], args.cache)
    cmd = [str(args.perplexity.resolve()), "-m", str(args.model), "-c", str(task["n_ctx"]),
           "-np", str(task["n_parallel"]), "-t", str(args.threads), *perplexity_args(task, data)]
    start = time.monotonic()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                          errors="replace")
    elapsed = time.monotonic() - start
    if args.log_dir:
        args.log_dir.mkdir(parents=True, exist_ok=True)
        (args.log_dir / f"{args.model.stem}-{name}.log").write_text(proc.stdout, encoding="utf-8")
    if proc.returncode != 0:
        raise EvalError(f"{name}: llama-perplexity exited {proc.returncode}")
    result = parse_output(task["mode"], proc.stdout)
    if task["tasks"] and result["n_tasks"] != task["tasks"]:
        raise EvalError(f"{name}: scored {result['n_tasks']} tasks, spec pins {task['tasks']}")
    chance = result.pop("chance", task.get("chance"))
    return {
        "schema": 1,
        "date": dt.date.today().isoformat(),
        "hardware": args.hardware,
        "model": args.model.name,
        "model_sha256": model_sha,
        "task": name,
        "dataset_sha256": spec["datasets"][task["dataset"]]["sha256"],
        **result,
        "chance": chance,
        "n_ctx": task["n_ctx"],
        "n_parallel": task["n_parallel"],
        "threads": args.threads,
        "evaluator_version": args.evaluator_version,
        "evaluator_sha256": args.evaluator_sha256,
        "llama_cpp_gitlink": llama_pin(),
        "elapsed_s": round(elapsed, 1),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--fetch", action="store_true", help="download and verify the pinned datasets")
    ap.add_argument("--model", type=Path)
    ap.add_argument("--hardware", help="explicit host hardware label (required with --model)")
    ap.add_argument("--tasks", help="comma-separated task ids (default: every task in the spec)")
    ap.add_argument("--out", type=Path, help="JSONL file; rows are appended")
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--perplexity", type=Path, default=DEFAULT_BIN)
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--log-dir", type=Path, help="keep the raw llama-perplexity output here")
    ap.add_argument("--spec", type=Path, default=SPEC_PATH)
    args = ap.parse_args(argv)

    try:
        spec = load_spec(args.spec)
        if args.fetch:
            fetch(spec, args.cache)
            if not args.model:
                return 0
        if not (args.model and args.out and args.hardware):
            ap.error("--model, --hardware and --out are required to run")
        if not args.model.is_file():
            raise EvalError(f"{args.model} not found")
        if not args.perplexity.is_file():
            raise EvalError(f"{args.perplexity} not found; build the linux-release preset")
        names = args.tasks.split(",") if args.tasks else list(spec["tasks"])
        unknown = [n for n in names if n not in spec["tasks"]]
        if unknown:
            raise EvalError(f"unknown task(s): {', '.join(unknown)}")
        model_sha = sha256_file(args.model)
        # Record the binary that scored, not only the repo's gitlink: --perplexity may
        # point at a build of another revision.
        args.evaluator_sha256 = sha256_file(args.perplexity.resolve())
        args.evaluator_version = evaluator_version(args.perplexity.resolve())
        args.out.parent.mkdir(parents=True, exist_ok=True)
        for name in names:
            row = run_task(args, spec, name, model_sha)
            with args.out.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, sort_keys=False) + "\n")
            print(f"{name:14s} {row['acc']:6.2f} ± {row['stderr']:.2f}  (n={row['n_tasks']}, "
                  f"{row['elapsed_s']:.0f} s)")
    except EvalError as e:
        print(f"eval-host-standard: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
