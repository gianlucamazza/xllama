#!/usr/bin/env python3
"""Standard generative benchmarks over an OpenAI-compatible endpoint (#597).

Sends the frozen IFEval and GSM8K subsets in bench/eval/standard-generative.json
to /v1/chat/completions (host llama-server or the console LAN API) at
temperature 0, seed 42, and scores each reply in code. One JSONL row per item,
then one summary row per task with accuracy and binomial stderr.

Same subset on the host and on the console: a gap larger than the stderr points
at the shipped path (template, stops, max tokens, thinking split), not at the
model. Informational only; H9 stays the promotion gate.

IFEval scoring is a stdlib port of the instruction checks in
google-research/instruction_following_eval (revision e6890f8,
Copyright 2023 The Google Research Authors, Apache License 2.0,
https://www.apache.org/licenses/LICENSE-2.0); the port is modified as below.
Two checks differ from upstream, which uses nltk and langdetect: sentences are
split on terminal punctuation, and the case checks do not also require English.
Scores are therefore comparable across runs of this script, not with published
IFEval numbers.

Usage:
  ./scripts/eval-standard-generative.py --fetch
  ./scripts/eval-standard-generative.py --base-url http://<ip>:11434 --model <id> \
      --hardware <label> [--tasks ifeval,gsm8k] --out bench/results/<date>-std-gen.jsonl
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import math
import re
import sys
import time
import urllib.error
import urllib.request
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

SPEC_PATH = ROOT / "bench/eval/standard-generative.json"
SCORERS = ("ifeval", "gsm8k")
SEED = 42


# --- IFEval checks -----------------------------------------------------------


def _compare(relation: str, actual: int, target: int) -> bool:
    if relation == "less than":
        return actual < target
    if relation == "at least":
        return actual >= target
    raise EvalError(f"unknown relation {relation!r}")


def count_words(text: str) -> int:
    # Upstream: nltk RegexpTokenizer(r"\w+"), identical to this.
    return len(re.findall(r"\w+", text))


def count_sentences(text: str) -> int:
    # Upstream uses nltk punkt; this splits on terminal punctuation runs.
    parts = re.split(r"(?<=[.!?])[\"')\]]*\s+", text.strip())
    return len([p for p in parts if re.search(r"\w", p)])


def _highlights(value, num_highlights):
    n = 0
    for h in re.findall(r"\*[^\n\*]*\*", value):
        if h.strip("*").strip():
            n += 1
    for h in re.findall(r"\*\*[^\n\*]*\*\*", value):
        if h.removeprefix("**").removesuffix("**").strip():
            n += 1
    return n >= num_highlights


def _paragraphs(value, num_paragraphs):
    parts = re.split(r"\s?\*\*\*\s?", value)
    n = len(parts)
    for i, p in enumerate(parts):
        if not p.strip():
            if i in (0, len(parts) - 1):
                n -= 1
            else:
                return False
    return n == num_paragraphs


def _nth_paragraph_first_word(value, num_paragraphs, nth_paragraph, first_word):
    parts = re.split(r"\n\n", value)
    n = sum(1 for p in parts if p.strip())
    if nth_paragraph > n:
        return False
    para = parts[nth_paragraph - 1].strip()
    if not para:
        return False
    word = para.split()[0].strip().lstrip("'").lstrip('"')
    got = ""
    for ch in word:
        if ch in {".", ",", "?", "!", "'", '"'}:
            break
        got += ch.lower()
    return n == num_paragraphs and got == first_word.lower()


def _two_responses(value):
    parts = value.split("******")
    valid = []
    for i, p in enumerate(parts):
        if not p.strip():
            if i not in (0, len(parts) - 1):
                return False
        else:
            valid.append(p)
    return len(valid) == 2 and valid[0].strip() != valid[1].strip()


def _json_format(value):
    v = value.strip()
    for prefix in ("```json", "```Json", "```JSON", "```"):
        v = v.removeprefix(prefix)
    v = v.removesuffix("```").strip()
    try:
        json.loads(v)
    except ValueError:
        return False
    return True


def _postscript(value, postscript_marker):
    v = value.lower()
    if postscript_marker == "P.P.S":
        pattern = r"\s*p\.\s?p\.\s?s.*$"
    elif postscript_marker == "P.S.":
        pattern = r"\s*p\.\s?s\..*$"
    else:
        pattern = r"\s*" + postscript_marker.lower() + r".*$"
    return bool(re.findall(pattern, v, flags=re.MULTILINE))


IFEVAL_CHECKS = {
    "punctuation:no_comma": lambda v: "," not in v,
    "length_constraints:number_words": lambda v, relation, num_words: _compare(
        relation, count_words(v), num_words
    ),
    "length_constraints:number_sentences": lambda v, relation, num_sentences: _compare(
        relation, count_sentences(v), num_sentences
    ),
    "length_constraints:number_paragraphs": _paragraphs,
    "length_constraints:nth_paragraph_first_word": _nth_paragraph_first_word,
    "keywords:forbidden_words": lambda v, forbidden_words: not any(
        re.search(r"\b" + w + r"\b", v, flags=re.IGNORECASE) for w in forbidden_words
    ),
    "keywords:existence": lambda v, keywords: all(
        re.search(k, v, flags=re.IGNORECASE) for k in keywords
    ),
    "keywords:frequency": lambda v, keyword, frequency, relation: _compare(
        relation, len(re.findall(keyword, v, flags=re.IGNORECASE)), frequency
    ),
    "keywords:letter_frequency": lambda v, letter, let_frequency, let_relation: _compare(
        let_relation, collections.Counter(v.lower())[letter.lower()], let_frequency
    ),
    "detectable_format:number_highlighted_sections": _highlights,
    "detectable_format:title": lambda v: any(
        t.lstrip("<").rstrip(">").strip() for t in re.findall(r"<<[^\n]+>>", v)
    ),
    "detectable_format:number_bullet_lists": lambda v, num_bullets: (
        len(re.findall(r"^\s*\*[^\*].*$", v, flags=re.MULTILINE))
        + len(re.findall(r"^\s*-.*$", v, flags=re.MULTILINE))
    )
    == num_bullets,
    "detectable_format:json_format": _json_format,
    "detectable_format:multiple_sections": lambda v, section_spliter, num_sections: len(
        re.split(r"\s?" + section_spliter + r"\s?\d+\s?", v)
    )
    - 1
    >= num_sections,
    "detectable_format:constrained_response": lambda v: any(
        o in v.strip() for o in ("My answer is yes.", "My answer is no.", "My answer is maybe.")
    ),
    "detectable_content:number_placeholders": lambda v, num_placeholders: len(
        re.findall(r"\[.*?\]", v)
    )
    >= num_placeholders,
    "detectable_content:postscript": _postscript,
    "change_case:english_lowercase": lambda v: v.islower(),
    "change_case:english_capital": lambda v: v.isupper(),
    "change_case:capital_word_frequency": lambda v, capital_frequency, capital_relation: _compare(
        capital_relation, sum(1 for w in re.findall(r"\w+", v) if w.isupper()), capital_frequency
    ),
    "startend:quotation": lambda v: len(v.strip()) > 1
    and v.strip()[0] == '"'
    and v.strip()[-1] == '"',
    "startend:end_checker": lambda v, end_phrase: v.strip()
    .strip('"')
    .lower()
    .endswith(end_phrase.strip().lower()),
    "combination:repeat_prompt": lambda v, prompt_to_repeat: v.strip()
    .lower()
    .startswith(prompt_to_repeat.strip().lower()),
    "combination:two_responses": _two_responses,
}


def score_ifeval(item: dict, reply: str) -> dict:
    """Strict IFEval: every instruction of the prompt must hold."""
    followed = []
    for iid, kwargs in zip(item["instruction_id_list"], item["kwargs"]):
        check = IFEVAL_CHECKS.get(iid)
        if check is None:
            raise EvalError(f"IFEval key {item['key']}: no check for {iid}")
        args = {k: v for k, v in (kwargs or {}).items() if v is not None}
        followed.append(bool(reply.strip()) and bool(check(reply, **args)))
    return {"pass": all(followed), "instructions": len(followed), "followed": sum(followed)}


# --- GSM8K -------------------------------------------------------------------

GSM8K_PROMPT = (
    "Solve the following math problem step by step. "
    'End your reply with a final line of the form "Answer: <number>".\n\n'
)
_NUMBER = r"-?\$?\d[\d,]*(?:\.\d+)?"


def _norm_number(text: str) -> str | None:
    t = text.replace(",", "").replace("$", "").strip().rstrip(".")
    try:
        value = float(t)
    except ValueError:
        return None
    return str(int(value)) if value == int(value) else repr(value)


def gsm8k_reference(answer: str) -> str:
    m = re.search(r"####\s*(" + _NUMBER + r")\s*$", answer.strip())
    if not m:
        raise EvalError("GSM8K answer without a '#### <number>' line")
    return _norm_number(m.group(1))


def gsm8k_extract(reply: str) -> str | None:
    """The number after the last 'Answer:', else the last number in the reply."""
    tagged = re.findall(r"answer\s*(?:is)?\s*[:=]?[\s*<]*(" + _NUMBER + r")", reply, re.I)
    found = tagged or re.findall(_NUMBER, reply)
    return _norm_number(found[-1]) if found else None


def score_gsm8k(item: dict, reply: str) -> dict:
    got = gsm8k_extract(reply)
    want = gsm8k_reference(item["answer"])
    return {"pass": got is not None and got == want, "extracted": got, "reference": want}


# --- runner ------------------------------------------------------------------


def load_spec(path: Path = SPEC_PATH) -> dict:
    spec = json.loads(path.read_text(encoding="utf-8"))
    validate_spec(spec)
    return spec


def validate_spec(spec: dict) -> None:
    validate_datasets(spec.get("datasets") or {})
    tasks = spec.get("tasks") or {}
    if not tasks:
        raise EvalError("spec needs tasks")
    for name, task in tasks.items():
        if task.get("dataset") not in spec["datasets"]:
            raise EvalError(f"task {name}: unknown dataset {task.get('dataset')!r}")
        if task.get("scorer") not in SCORERS:
            raise EvalError(f"task {name}: scorer must be one of {SCORERS}")
        keys = task.get("keys")
        if not keys or len(set(keys)) != len(keys) or not all(isinstance(k, int) for k in keys):
            raise EvalError(f"task {name}: keys must be a non-empty list of distinct ints")
        if not isinstance(task.get("max_tokens"), int) or task["max_tokens"] <= 0:
            raise EvalError(f"task {name}: max_tokens must be a positive int")


def load_items(spec: dict, name: str, cache: Path) -> list[dict]:
    task = spec["tasks"][name]
    path = dataset_path(spec, task["dataset"], cache)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if task["scorer"] == "ifeval":
        by_key = {r["key"]: r for r in rows}
    else:
        by_key = dict(enumerate(rows))
    missing = [k for k in task["keys"] if k not in by_key]
    if missing:
        raise EvalError(f"task {name}: keys not in dataset: {missing[:5]}")
    items = []
    for k in task["keys"]:
        item = dict(by_key[k], _key=k)
        if task["scorer"] == "ifeval":
            unknown = [i for i in item["instruction_id_list"] if i not in IFEVAL_CHECKS]
            if unknown:
                raise EvalError(f"task {name}: key {k} needs unported checks {unknown}")
        items.append(item)
    return items


def prompt_for(scorer: str, item: dict) -> str:
    return item["prompt"] if scorer == "ifeval" else GSM8K_PROMPT + item["question"]


def chat(base_url: str, model: str, prompt: str, max_tokens: int, timeout: float) -> dict:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": max_tokens,
            "temperature": 0,
            "seed": SEED,
            "stream": False,
        }
    ).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raise EvalError(f"HTTP {e.code}: {e.read()[:200]!r}") from e


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    passed = sum(1 for r in rows if r["pass"])
    p = passed / n if n else 0.0
    out = {
        "n_items": n,
        "passed": passed,
        "acc": round(100.0 * p, 2),
        "stderr": round(100.0 * math.sqrt(p * (1 - p) / (n - 1)), 2) if n > 1 else None,
        "errors": sum(1 for r in rows if r.get("error")),
        "length_stops": sum(1 for r in rows if r.get("finish_reason") == "length"),
    }
    if rows and "instructions" in rows[0]:
        total = sum(r.get("instructions", 0) for r in rows)
        out["instruction_acc"] = round(100.0 * sum(r.get("followed", 0) for r in rows) / total, 2)
    return out


def run_task(args, spec: dict, name: str, out) -> dict:
    task = spec["tasks"][name]
    scorer = task["scorer"]
    items = load_items(spec, name, args.cache)
    if args.limit:
        items = items[: args.limit]
    rows = []
    for i, item in enumerate(items, 1):
        start = time.monotonic()
        row = {"type": "item", "task": name, "key": item["_key"]}
        try:
            resp = chat(args.base_url, args.model, prompt_for(scorer, item), task["max_tokens"],
                        args.timeout)
            choice = resp["choices"][0]
            reply = choice["message"].get("content") or ""
            row["finish_reason"] = choice.get("finish_reason")
            row["usage"] = resp.get("usage")
            row.update(score_ifeval(item, reply) if scorer == "ifeval" else score_gsm8k(item, reply))
            row["content"] = reply
        except (EvalError, OSError, KeyError, ValueError) as e:
            row.update({"pass": False, "error": str(e)})
            if scorer == "ifeval":
                row.update({"instructions": len(item["instruction_id_list"]), "followed": 0})
        row["latency_s"] = round(time.monotonic() - start, 2)
        rows.append(row)
        out.write(json.dumps(row, ensure_ascii=False) + "\n")
        out.flush()
        if args.verbose:
            print(f"{name} {i}/{len(items)} key={item['_key']} pass={row['pass']}", file=sys.stderr)
    summary = {
        "type": "summary",
        "schema": 1,
        "date": dt.date.today().isoformat(),
        "hardware": args.hardware,
        "endpoint": args.endpoint_label,
        "model": args.model,
        "task": name,
        "dataset_sha256": spec["datasets"][task["dataset"]]["sha256"],
        "spec_sha256": sha256_file(args.spec),
        "max_tokens": task["max_tokens"],
        "seed": SEED,
        "partial": bool(args.limit and args.limit < len(task["keys"])),
        **summarize(rows),
    }
    out.write(json.dumps(summary) + "\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--fetch", action="store_true", help="download and verify the pinned datasets")
    ap.add_argument("--base-url", help="e.g. http://127.0.0.1:8080 or http://<xbox-ip>:11434")
    ap.add_argument("--model", help="model id the endpoint expects")
    ap.add_argument("--hardware", help="explicit hardware label (required to run)")
    ap.add_argument("--endpoint-label", default="", help="free text, e.g. 'llama-server d812350'")
    ap.add_argument("--tasks", help="comma-separated task ids (default: every task)")
    ap.add_argument("--out", type=Path, help="JSONL file; rows are appended")
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--timeout", type=float, default=600.0, help="seconds per request")
    ap.add_argument("--spec", type=Path, default=SPEC_PATH)
    ap.add_argument("--limit", type=int, default=0,
                    help="smoke only: first N items; the summary is marked partial")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    try:
        spec = load_spec(args.spec)
        if args.fetch:
            fetch(spec, args.cache)
            if not args.base_url:
                return 0
        if not (args.base_url and args.model and args.hardware and args.out):
            ap.error("--base-url, --model, --hardware and --out are required to run")
        names = args.tasks.split(",") if args.tasks else list(spec["tasks"])
        unknown = [n for n in names if n not in spec["tasks"]]
        if unknown:
            raise EvalError(f"unknown task(s): {', '.join(unknown)}")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("a", encoding="utf-8") as out:
            for name in names:
                s = run_task(args, spec, name, out)
                print(f"{name:8s} {s['acc']:6.2f} ± {s['stderr']}  ({s['passed']}/{s['n_items']}, "
                      f"errors {s['errors']}, length stops {s['length_stops']})")
    except EvalError as e:
        print(f"eval-standard-generative: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
