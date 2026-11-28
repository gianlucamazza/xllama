#!/usr/bin/env python3
# Copyright (c) 2024 Gianluca Mazza
# SPDX-License-Identifier: MIT
"""Repeat the Linux GGUF model-open timing experiment in #703 / #708 using xllama-cli."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import resource
import subprocess
import tempfile
import threading
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[3]
MODEL_SHA = '3d10b6ab8fc91a919534b9558e266255aca0bbc7f6d015963599aa9e74e05b1d'
PROMPTS = ('short-32', 'long-1k', 'standard-512')
TIMEOUT_S = 120


def sha(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def child_cpu():
    r = resource.getrusage(resource.RUSAGE_CHILDREN)
    return r.ru_utime + r.ru_stime


def run(cli, model, prompt, cadence, threads, chat):
    args = [str(cli), '-m', str(model), '-p', prompt, '-c', '2048', '-n', '64',
            '-t', str(threads), '--greedy', '--seed', '1', '--decode-progress-every', str(int(cadence)),
            '--decode-progress-ms', '0']
    if chat:
        args.append('--chat')
    with tempfile.TemporaryFile() as log:
        cpu = child_cpu()
        start = time.perf_counter_ns()
        with subprocess.Popen(args, stdout=subprocess.PIPE, stderr=log) as proc:
            timed_out = threading.Event()
            def expire():
                if proc.poll() is None:
                    timed_out.set()
                    proc.kill()
            watchdog = threading.Timer(TIMEOUT_S, expire)
            watchdog.start()
            try:
                first = proc.stdout.read(1)
                first_ms = (time.perf_counter_ns() - start) / 1e6 if first else None
                rest, _ = proc.communicate()
            finally:
                watchdog.cancel()
                watchdog.join()
        wall = (time.perf_counter_ns() - start) / 1e6
        cpu_ms = (child_cpu() - cpu) * 1000
        log.seek(0)
        lines = log.read().decode('utf-8', errors='replace')
    row = {'exit': proc.returncode, 'first_stdout_byte_ms': first_ms, 'process_ms': wall,
           'child_cpu_ms': cpu_ms, 'stdout_sha256': hashlib.sha256(first + rest).hexdigest(),
           'timed_out': timed_out.is_set()}
    if proc.returncode:
        row['error'] = lines.splitlines()[-1:]
        return row
    prefill = re.search(r'prefill done: (\d+) tokens in (\d+) ms', lines)
    done = re.search(r'done: load=(\d+)ms prompt=([\d.]+) tok/s', lines)
    spec = re.search(r'SPEC_STATS success=1 n_eval=(\d+) t_eval_ms=([\d.]+) '
                     r'decode_tok_s=([\d.]+) .*? peak_ws_mb=(\d+)', lines)
    if not (prefill and done and spec):
        row['error'] = 'incomplete CLI timings'
        return row
    row.update(prompt_tokens=int(prefill[1]), prefill_ms=int(prefill[2]),
               load_ms=int(done[1]), prefill_tok_s=float(done[2]), generated_tokens=int(spec[1]),
               decode_ms=float(spec[2]), decode_tok_s=float(spec[3]), peak_ws_mib=int(spec[4]))
    # The interactive CLI appends one final newline even for an immediate EOG.
    row['first_output_piece_ms'] = first_ms if len(first + rest) > 1 else None
    if cadence:
        ticks = [(int(i), int(ms)) for i, ms in re.findall(
            r'decode progress: (\d+)/\d+ tokens \((\d+) ms\)', lines)]
        if [i for i, _ in ticks] != list(range(1, row['generated_tokens'] + 1)):
            row['error'] = 'incomplete per-step times'
        row['completed_decode_step_ms'] = [ms for _, ms in ticks]
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('base', 'candidate', 'model', 'out'):
        parser.add_argument('--' + arg, type=Path, required=True)
    parser.add_argument('--prompt-name', choices=PROMPTS, help='repeat one complete prompt cell')
    args = parser.parse_args()
    if sha(args.model) != MODEL_SHA:
        parser.error('model SHA-256 differs from the catalogue file')
    bins = {'base': args.base.resolve(), 'candidate': args.candidate.resolve()}
    selected = (args.prompt_name,) if args.prompt_name else PROMPTS
    prompts = {p: ROOT / 'bench' / 'prompts' / (p + '.txt') for p in selected}
    cpu = next(line.partition(':')[2].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
               if line.startswith('model name'))
    metadata = {'utc_start': datetime.now(timezone.utc).isoformat(), 'cpu': cpu,
                'kernel': platform.release(), 'affinity': sorted(os.sched_getaffinity(0)),
                'model_sha256': MODEL_SHA, 'cli_sha256': {k: sha(v) for k, v in bins.items()},
                'timeout_s': TIMEOUT_S, 'prompt_file_sha256': {k: sha(v) for k, v in prompts.items()},
                'prompt_argument_sha256': {k: hashlib.sha256(v.read_text().rstrip('\n').encode()).hexdigest()
                                           for k, v in prompts.items()}}
    failed = False
    with args.out.open('x', encoding='utf-8') as out:
        print(json.dumps(metadata), file=out, flush=True)
        for name, path in prompts.items():
            prompt = path.read_text(encoding='utf-8').rstrip('\n')
            chat = name == 'standard-512'
            # Fresh processes with shared OS file cache. Keep each warmup and failure.
            schedule = [('warmup', 'base'), ('warmup', 'candidate')]
            for order in (('base', 'candidate'), ('candidate', 'base'), ('base', 'candidate')):
                schedule.extend(('paired', arm) for arm in order)
            schedule.extend((('step-timing', 'candidate'), ('two-thread', 'candidate')))
            for index, (kind, arm) in enumerate(schedule):
                threads = 2 if kind == 'two-thread' else 4
                result = run(bins[arm], args.model.resolve(), prompt, kind == 'step-timing', threads, chat)
                failed |= bool(result['exit'] or 'error' in result)
                print(json.dumps(dict(prompt=name, chat=chat, order=index, role=kind, arm=arm,
                                      threads=threads, **result)), file=out, flush=True)
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
