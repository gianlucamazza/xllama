#!/usr/bin/env python3
# Copyright (c) 2024 Gianluca Mazza
# SPDX-License-Identifier: MIT
"""Measure #725 within completed autopilot sends from one scoped XAML run."""

import argparse
import json
from pathlib import Path
import re


STAMP = re.compile(r"^\d\d:\d\d:\d\d\.\d{3} ")
ACTION = re.compile(r"\[autopilot\] action (\d+) send (start|end)")
SHIFT = re.compile(r"\[xllama\] session: context shift — evicted (\d+) tokens past "
                   r"keep=(\d+), kv now (\d+) of (\d+) \(rebuilt from (\d+)\) \(#169/#723\)")
GENERATE = re.compile(r"\[xllama\] session generate: n=\d+ prefill=(\d+\.\d+)ms "
                      r"\((\d+) tok\) reuse=([01]) drafted=\d+ spec_accept=\d+")


def extended_plan(plan):
    actions = plan["actions"]
    sends = [a for a in actions if a["op"] == "send"]
    # Keep the original conversation and use the long follow-up for another 30 turns.
    insertion = max(i for i, a in enumerate(actions) if a["op"] == "send")
    question = sends[1]["text"]
    actions[insertion:insertion] = [
        {**sends[1], "text": f"{question} Focus on item {i % 12:02d} in follow-up {i + 1}."}
        for i in range(30)
    ]
    sampling = next(a for a in actions if a["op"] == "set_sampling")
    sampling.update(temperature=0, top_k=1, top_p=1, repetition_penalty=1, n_predict=64)
    # Route the selected GGUF chat model on every fresh conversation.
    actions.insert(0, {"op": "set_routing", "routing": 0})
    actions.insert(1, {"op": "set_system_prompt", "text": "You are a helpful AI assistant."})
    plan["total_timeout_s"] = 3600
    return plan


def measure(text, plan):
    expected = [i for i, action in enumerate(plan["actions"]) if action["op"] == "send"]
    blocks = []
    active = None
    current = []
    for line in text.splitlines():
        line = STAMP.sub("", line)
        action = ACTION.fullmatch(line)
        if not action:
            if active is not None:
                current.append(line)
            continue
        index = int(action[1])
        if action[2] == "start":
            if active is not None or len(blocks) >= len(expected) or index != expected[len(blocks)]:
                raise ValueError(f"unexpected send start {index}")
            active, current = index, []
        elif index != active:
            raise ValueError(f"unmatched send end {index}")
        else:
            blocks.append(current)
            active = None
    if active is not None or not expected or len(blocks) != len(expected):
        raise ValueError("incomplete autopilot send sequence")

    rows = []
    previous = None
    for turn, lines in enumerate(blocks, 1):
        generates = [m for line in lines if (m := GENERATE.fullmatch(line))]
        shifts = [line for line in lines if line.startswith("[xllama] session: context shift")]
        if len(generates) != 1 or len(shifts) > 1:
            raise ValueError(f"ambiguous generation or shift at turn {turn}")
        if not shifts:
            continue
        shift = SHIFT.fullmatch(shifts[0])
        if (not shift or generates[0][3] != "1" or
                lines.index(shifts[0]) >= lines.index(generates[0].string)):
            raise ValueError(f"expected a hybrid rebuild with KV reuse at turn {turn}")
        evicted, keep, surviving, context, restored = map(int, shift.groups())
        prefill = int(generates[0][2])
        new = prefill - (surviving - restored)
        if not (evicted > 0 and 0 <= restored <= keep <= surviving <= context and
                new > 0 and surviving + new <= context):
            raise ValueError(f"inconsistent token accounting at turn {turn}")
        rows.append({"turn": turn, "evicted_tokens": evicted, "pinned_tokens": keep,
                     "kept_tokens": surviving, "new_tokens": new,
                     "restored_tokens": restored, "context_tokens": context,
                     "prefill_tokens": prefill, "prefill_ms": float(generates[0][1]),
                     "intervening_turns": None if previous is None else turn - previous - 1})
        previous = turn
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--log", type=Path, help="run-scoped app log for the completed plan")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if args.log is None:
        print(json.dumps(extended_plan(plan), indent=2))
        return 0
    try:
        rows = measure(args.log.read_text(encoding="utf-8", errors="replace"), plan)
    except ValueError as error:
        print(json.dumps({"error": str(error)}, indent=2))
        return 1
    # Preserve a valid single-shift observation even though it misses this campaign's gate.
    print(json.dumps({"completed_sends": sum(a["op"] == "send" for a in plan["actions"]),
                      "shifts": rows}, indent=2))
    return 0 if len(rows) >= 2 else 1


if __name__ == "__main__":
    raise SystemExit(main())
