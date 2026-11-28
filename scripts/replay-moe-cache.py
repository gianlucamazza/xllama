#!/usr/bin/env python3
# Copyright (c) 2024 Gianluca Mazza
# SPDX-License-Identifier: MIT
"""Reconstruct fixed-slot LRU counters for archived UA50TRC1 console runs."""

import argparse
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import struct

# On-disk v1 layout from the contributor's trace writer in the project's
# evidence/2026-11-16-moe-24b-console tag, patches/0005-ua50-cache-ab-llama.patch.
FILE_SIZE = 499904
ROLE_COUNT = 114
FRAME_CAPACITY = 511
REQUESTS_PER_FRAME = ROLE_COUNT * 4
KEY_CAPACITY = FRAME_CAPACITY * REQUESTS_PER_FRAME
SIZES_OFFSET = 128
FRAMES_OFFSET = SIZES_OFFSET + ROLE_COUNT * 8
KEYS_OFFSET = FRAMES_OFFSET + FRAME_CAPACITY * 64
FOOTER_OFFSET = KEYS_OFFSET + KEY_CAPACITY * 2
COUNTERS = ("requests", "requested_bytes", "hits", "hit_bytes", "misses", "evictions",
            "read_calls", "read_bytes", "failures")


def replay_profile(directory, receipt, profile, slots):
    run = receipt["runs"][profile]
    record_bytes = (directory / f"{profile}.output.json").read_bytes()
    if hashlib.sha256(record_bytes).hexdigest() != run["output_sha256"]:
        raise ValueError(f"{profile}: output SHA-256 mismatch")
    record = json.loads(record_bytes)
    if record["schema"] != "xllama-cache-ab-output-v1":
        raise ValueError(f"{profile}: unsupported output record")
    capture = record["cache_ab"]
    with (directory / f"{profile}.trace.bin").open("rb") as stream:
        trace = stream.read(FILE_SIZE + 1)
    if len(trace) != FILE_SIZE or trace[:8] != b"UA50TRC1":
        raise ValueError(f"{profile}: invalid v1 trace length or magic")
    version, flags, length, roles, frame_cap, key_cap, frames, requests, trace_slots = (
        struct.unpack_from("<9I", trace, 8)
    )
    if (version, length, roles, frame_cap, key_cap) != (
        1, FILE_SIZE, ROLE_COUNT, FRAME_CAPACITY, KEY_CAPACITY
    ):
        raise ValueError(f"{profile}: unsupported trace layout")
    status = capture["trace_status"]
    checkpoint_ok = trace[64:96].hex() == receipt["model"]["sha256"]
    if flags != 1 or status != "complete_replayable" or not checkpoint_ok:
        identity = "" if checkpoint_ok else ", checkpoint mismatch"
        raise ValueError(f"{profile}: trace status {status}, flags {flags}{identity}")
    trace_hash = hashlib.sha256(trace).hexdigest()
    if trace_hash != run["trace_sha256"]:
        raise ValueError(f"{profile}: trace SHA-256 mismatch")
    if capture["trace_flags"] != flags or capture["profile"] != profile.upper():
        raise ValueError(f"{profile}: output profile or completion flags disagree")
    if not (trace_slots == capture["slots"] == run["cache_slots"] == slots):
        raise ValueError(f"{profile}: cache capacity mismatch")
    if not (0 < frames <= FRAME_CAPACITY and frames == capture["decode_count"]
            and requests == frames * REQUESTS_PER_FRAME and requests == capture["key_count"]):
        raise ValueError(f"{profile}: invalid frame or request count")
    for i in range(frames):
        frame = struct.unpack_from("<4I", trace, FRAMES_OFFSET + i * 64)
        first = i * REQUESTS_PER_FRAME
        if frame != (i, first, first + REQUESTS_PER_FRAME, 3):
            raise ValueError(f"{profile}: incomplete or discontinuous frame {i}")

    sizes = struct.unpack_from(f"<{ROLE_COUNT}Q", trace, SIZES_OFFSET)
    keys = trace[KEYS_OFFSET:KEYS_OFFSET + 2 * requests]
    cache = OrderedDict()
    requested_bytes = hits = hit_bytes = evictions = 0
    for (key,) in struct.iter_unpack("<H", keys):
        role = key >> 6
        if role >= len(sizes) or not sizes[role]:
            raise ValueError(f"{profile}: invalid expert role or length")
        size = sizes[role]
        requested_bytes += size
        if key in cache:
            hits += 1
            hit_bytes += size
            cache.move_to_end(key)
        else:
            if len(cache) == slots:
                cache.popitem(last=False)
                evictions += 1
            cache[key] = None

    misses = requests - hits
    computed = (requests, requested_bytes, hits, hit_bytes, misses, evictions,
                misses, requested_bytes - hit_bytes, 0)
    footer = struct.unpack_from(f"<{len(COUNTERS)}Q", trace, FOOTER_OFFSET)
    for name, actual, recorded in zip(COUNTERS, computed, footer):
        if actual != recorded or actual != capture["counters"][name]:
            raise ValueError(f"{profile}: replay disagrees with recorded {name}")
    if (run["route_requests"], run["cache_hits_from_log"], run["reader_logical_bytes_from_log"]) != (
        requests, hits, requested_bytes - hit_bytes
    ):
        raise ValueError(f"{profile}: summary counters disagree with the replay")

    return {
        "trace_sha256": trace_hash,
        "route_sha256": hashlib.sha256(trace[SIZES_OFFSET:FRAMES_OFFSET] + keys).hexdigest(),
        "frames": frames, "slots": slots,
        **dict(zip(COUNTERS, computed)),
        "request_hit_rate": hits / requests, "byte_hit_rate": hit_bytes / requested_bytes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="console evidence directory with result.json")
    args = parser.parse_args()
    try:
        receipt = json.loads((args.directory / "result.json").read_bytes())
        if receipt["schema"] != "xllama-moe-console-run-v1":
            raise ValueError("unsupported console receipt")
        arms = {p: replay_profile(args.directory, receipt, p, slots)
                for p, slots in (("a2", 2), ("b512", 512))}
        if arms["a2"]["route_sha256"] != arms["b512"]["route_sha256"]:
            raise ValueError("profiles have different ordered requests or slice lengths")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"replay refused: {exc}\n")
    print(json.dumps({"checkpoint_sha256": receipt["model"]["sha256"], **arms}, indent=2))


if __name__ == "__main__":
    main()
