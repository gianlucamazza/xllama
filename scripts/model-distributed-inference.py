#!/usr/bin/env python3
# Copyright (c) 2024 Gianluca Mazza
# SPDX-License-Identifier: MIT
"""Host-only DC-0 sensitivity model. Write hypothetical steady-state decode cases to CSV."""

import argparse
import csv
import io
import json
import math
import sys
from dataclasses import dataclass, fields, replace

GIB = 2**30
GB = 10**9
NODES = (1, 2, 4, 8, 16, 32)
STRATEGIES = ("replica", "pipeline", "tensor", "expert", "hybrid")


@dataclass(frozen=True)
class Shape:
    params_b: float
    layers: int
    hidden: int
    kv_heads: int = 8
    head_dim: int = 128
    window: int = 2048
    kv_bytes: int = 2
    act_bytes: int = 2
    vocab: int = 128000
    experts: int = 64
    top_k: int = 4
    shared_fraction: float = 0.20


# Synthetic shapes. The dictionary keys label study classes, not checkpoint identities.
SHAPES = {p: Shape(p, layers, hidden) for p, layers, hidden in (
    (8, 32, 4096), (14, 40, 5120), (32, 64, 5120),
    (70, 80, 8192), (120, 96, 10240), (256, 128, 12288),
)}


@dataclass(frozen=True)
class Inputs:
    cap_gib: float = 4.0           # Chosen app working-set limit
    reserve_gib: float = 0.65      # Non-model RSS, scratch, copies and safety margin
    bpw: float = 4.8               # Effective weight bits/parameter, all tensor types
    mem_gbs: float = 13.0          # Effective kernel weight/attention read GB/s
    gflops: float = 75.0           # Effective kernel arithmetic GFLOP/s
    layer_us: float = 60.0         # Fixed serial overhead for each layer
    tcp_mbps: float = 800.0        # Application goodput, megabits/s
    rtt_ms: float = 0.65
    copy_gbs: float = 7.0          # Effective host staging bandwidth
    sync_us: float = 60.0          # Per serialized exchange or collective round
    ssd_mbs: float = 400.0         # Effective read MB/s
    ssd_ms: float = 0.25           # Disjoint read service/queue latency
    cache_gib: float = 0.5         # Bounded expert cache per rank
    hit: float = 0.6               # Steady state when cache does not hold the whole shard
    active_w: float = 75.0
    idle_w: float = 12.0
    coord_active_w: float = 35.0   # Incremental coordinator + switch if N > 1
    coord_idle_w: float = 8.0
    node_eur: float = 250.0
    coord_eur: float = 180.0
    eur_kwh: float = 0.28
    years: float = 3.0
    utilization: float = 0.35


SCENARIOS = {
    "optimistic": replace(Inputs(), cap_gib=4.5, reserve_gib=0.5, bpw=4.5,
                          mem_gbs=22, gflops=130, layer_us=30, tcp_mbps=930,
                          rtt_ms=0.25, copy_gbs=10, sync_us=25, ssd_mbs=850,
                          ssd_ms=0.10, hit=0.85, active_w=60),
    "realistic": Inputs(),        # Central illustrative planning case, uncalibrated
    "pessimistic": replace(Inputs(), cap_gib=3.25, reserve_gib=0.8, bpw=5.3,
                           mem_gbs=8, gflops=45, layer_us=120, tcp_mbps=350,
                           rtt_ms=2.0, copy_gbs=3, sync_us=150, ssd_mbs=150,
                           ssd_ms=0.9, hit=0.25, active_w=95),
}


def validate(shape, cfg, streams, group, legs, remote_fraction):
    for obj in (shape, cfg):
        for field in fields(obj):
            if not math.isfinite(getattr(obj, field.name)):
                raise ValueError(f"{field.name} must be finite")
    if any(getattr(shape, name) <= 0 for name in
           ("params_b", "layers", "hidden", "kv_heads", "head_dim", "window",
            "kv_bytes", "act_bytes", "vocab", "experts", "top_k")):
        raise ValueError("shape dimensions must be positive")
    if (shape.hidden % shape.head_dim or shape.kv_heads > shape.hidden // shape.head_dim
            or (shape.hidden // shape.head_dim) % shape.kv_heads
            or shape.top_k > shape.experts):
        raise ValueError("invalid head dimensions or expert counts")
    if not 0 <= shape.shared_fraction <= 1:
        raise ValueError("invalid shared weight fraction")
    if 2 * shape.vocab * shape.hidden >= shape.params_b * GB:
        raise ValueError("untied vocabulary tables exhaust total parameters")
    if any(getattr(cfg, field.name) < 0 for field in fields(cfg)):
        raise ValueError("scenario values must be nonnegative")
    if any(getattr(cfg, name) <= 0 for name in
           ("cap_gib", "bpw", "mem_gbs", "gflops", "tcp_mbps", "copy_gbs",
            "ssd_mbs", "years", "utilization")):
        raise ValueError("capacities, rates, years and utilization must be positive")
    if cfg.hit > 1 or cfg.utilization > 1 or cfg.reserve_gib >= cfg.cap_gib:
        raise ValueError("invalid hit rate, utilization or reserve")
    if streams < 1 or group < 1 or legs not in (1, 2):
        raise ValueError("invalid streams, expert group or transport legs")
    if remote_fraction is not None and not 0 <= remote_fraction <= 1:
        raise ValueError("remote fraction must be between zero and one")


def transport(cfg, byte_count, exchanges):
    """Serialized goodput, staging at sender + receiver, RTT and synchronization."""
    return (byte_count * 8 / (cfg.tcp_mbps * 1e6) + 2 * byte_count / (cfg.copy_gbs * GB)
            + exchanges * (cfg.rtt_ms / 1000 + cfg.sync_us / 1e6))


def kernel(cfg, weight_bytes, attention_bytes, flops, layers):
    return max((weight_bytes + attention_bytes) / (cfg.mem_gbs * GB),
               flops / (cfg.gflops * GB)) + layers * cfg.layer_us / 1e6


def cost(cfg, nodes, rate, layout=None):
    coordinator = int(nodes > 1)
    nodes_config = layout if layout is not None else [{}] * nodes
    active_w = sum(node.get("active_w", cfg.active_w) for node in nodes_config)
    idle_w = sum(node.get("idle_w", cfg.idle_w) for node in nodes_config)
    capex = sum(node.get("node_eur", cfg.node_eur) for node in nodes_config)
    active_w += coordinator * cfg.coord_active_w
    idle_w += coordinator * cfg.coord_idle_w
    mean_w = cfg.utilization * active_w + (1 - cfg.utilization) * idle_w
    mean_eur_h = (capex + coordinator * cfg.coord_eur) / (8760 * cfg.years)
    mean_eur_h += mean_w / 1000 * cfg.eur_kwh
    return active_w / rate, 1e6 * mean_eur_h / (3600 * cfg.utilization * rate)


def evaluate(shape, cfg, family, strategy, nodes, streams=1, group=2, legs=2,
             remote_fraction=None, layout=None, expert_legs=2):
    """Memory fits and serialized token estimates. Link counters state their scope."""
    if family == "dense" and strategy in ("expert", "hybrid"):
        raise ValueError("expert strategies require an MoE shape")
    groupsize = min(group, nodes) if strategy == "hybrid" else 1
    if strategy == "expert":
        groupsize = nodes
    stages = nodes // groupsize if strategy in ("pipeline", "hybrid", "expert") else 1
    blank = {field: None for field in METRICS}
    blank.update(status="layout", params_b=shape.params_b)
    if nodes % groupsize or stages > shape.layers or groupsize > shape.experts:
        return blank
    weights = shape.params_b * GB * cfg.bpw / 8
    table = shape.vocab * shape.hidden * cfg.bpw / 8
    lookup = shape.hidden * cfg.bpw / 8
    per_layer = (weights - 2 * table) / shape.layers
    external_input = stages > 1
    expert_bytes = per_layer * (1 - shape.shared_fraction) if family == "moe" else 0
    # Uniform grouped-query attention requires Q/K/V/O; the MoE router is shared.
    shared_params = (per_layer - expert_bytes) * 8 / cfg.bpw
    attention_params = 2 * shape.hidden * (shape.hidden + shape.kv_heads * shape.head_dim)
    router_params = shape.hidden * shape.experts if family == "moe" else 0
    if shared_params < attention_params + router_params:
        return blank
    per_expert = expert_bytes / shape.experts
    active = per_layer - expert_bytes + per_expert * shape.top_k
    kv_layer = 2 * shape.window * shape.kv_heads * shape.head_dim * shape.kv_bytes
    kv_io = kv_layer * (1 + 1 / shape.window)  # history read plus current KV append
    attention_flops = 4 * shape.hidden * shape.window
    activation = shape.hidden * shape.act_bytes
    slots = streams if strategy == "pipeline" else 1
    rank_peaks, stage_caps, stage_times = [], [], []
    available = 0.0
    if layout is not None:
        if (strategy != "pipeline" or len(layout) != nodes
                or sum(item["layers"] for item in layout) != shape.layers):
            return blank
    if groupsize == 1:
        remote = 0.0
    else:
        remote = 1 - 1 / groupsize if remote_fraction is None else remote_fraction
    expert_wire = expert_legs * 2 * shape.layers * shape.top_k * remote * activation
    expert_calls = expert_legs * shape.layers * shape.top_k * remote
    disk_bytes = disk_delay = 0.0
    layout_ok = True
    for index in range(stages):
        layers = shape.layers // stages + int(index < shape.layers % stages)
        stage_cfg = cfg
        if layout is not None:
            node = layout[index]
            layers = node["layers"]
            stage_cfg = replace(cfg, **{k: v for k, v in node.items() if k != "layers"})
            validate(shape, stage_cfg, streams, group, legs, remote_fraction)
        available += stage_cfg.cap_gib - stage_cfg.reserve_gib
        stage_caps.append(stage_cfg.cap_gib)
        input_local = index == 0 and not external_input
        output_local = index == stages - 1
        fixed = table * (input_local + output_local)
        endpoint_io = lookup * input_local + table * output_local
        resident = fixed + layers * (per_layer - expert_bytes)
        shard = layers * per_expert * math.ceil(shape.experts / groupsize)
        if strategy == "tensor":
            # Replicate endpoint tensors. Count whole KV heads on each rank.
            query_heads = shape.hidden // shape.head_dim
            layout_ok = (query_heads % nodes == 0
                         and (shape.kv_heads % nodes == 0 or nodes % shape.kv_heads == 0))
            kv_share = math.ceil(shape.kv_heads / nodes) / shape.kv_heads
            rank_peaks.append((fixed + layers * (per_layer / nodes + kv_layer * kv_share)) / GIB
                              + stage_cfg.reserve_gib)
            io = endpoint_io + layers * active / nodes
            flops = 2 * io * 8 / cfg.bpw + layers * attention_flops / nodes
            stage_times.append(kernel(stage_cfg, io, layers * kv_io * kv_share, flops, layers))
            continue
        if strategy == "hybrid":
            cache = min(cfg.cache_gib * GIB, shard)
            layout_ok &= cache >= per_expert
            hit = 1.0 if cache >= shard else cfg.hit
            misses = layers * shape.top_k * (1 - hit)
            disk_bytes += misses * per_expert
            disk_delay += misses * (per_expert / (cfg.ssd_mbs * 1e6) + cfg.ssd_ms / 1000)
            shard = cache
        rank_peaks.append((resident + shard + slots * layers * kv_layer) / GIB + stage_cfg.reserve_gib)
        io = endpoint_io + layers * active
        flops = 2 * io * 8 / cfg.bpw + layers * attention_flops
        stage_times.append(kernel(stage_cfg, io, layers * kv_io, flops, layers))
    if not layout_ok:
        status = "layout"
    elif any(peak > cap for peak, cap in zip(rank_peaks, stage_caps)):
        status = "budget"
    else:
        status = "fits"
    network_s = 0.0
    wire = expert_wire
    nic_s = 0.0
    if stages > 1:
        # Coordinator owns the input lookup, embeds one ID and samples the full f32 row.
        feedback = activation + 4 * shape.vocab
        wire += legs * (stages - 1) * activation + feedback
        network_s += transport(cfg, legs * (stages - 1) * activation + feedback,
                               legs * (stages - 1) + 2)
        # Critical shared egress/ingress: indirect worker -> host -> worker, full duplex.
        if legs == 2:
            busiest_direction = (stages - 1) * activation + max(activation, 4 * shape.vocab)
        else:
            busiest_direction = max(activation, 4 * shape.vocab)
        nic_s = 8 * busiest_direction / (cfg.tcp_mbps * 1e6)
    if expert_calls:
        network_s += transport(cfg, expert_wire, expert_calls)
    scope = "no_tensor_payload_http_excluded" if strategy == "replica" else "cluster_payload"
    if strategy == "tensor" and nodes > 1:
        # Two ring all-reduces per layer; node egress, also received at same rate.
        scope = "per_rank_tx"
        collectives = 2 * shape.layers
        wire = collectives * 2 * (nodes - 1) / nodes * activation
        rounds = collectives * 2 * (nodes - 1)
        network_s = (8 * wire / (cfg.tcp_mbps * 1e6) + 2 * wire / (cfg.copy_gbs * GB)
                     + rounds * (cfg.rtt_ms / 2000 + cfg.sync_us / 1e6))
    host_compute = lookup / (cfg.mem_gbs * GB) if external_input else 0.0
    latency = sum(stage_times) + host_compute + network_s + disk_delay
    rate = aggregate = joules = eur = None
    if status == "fits":
        rate = 1 / latency
        aggregate = rate
        if strategy == "replica":
            aggregate *= nodes
        elif strategy == "pipeline":
            aggregate = min(streams / latency, 1 / max(max(stage_times), nic_s))
        joules, eur = cost(cfg, nodes, aggregate, layout)
    host_floor = (table + max(activation, 4 * shape.vocab)) / GIB if external_input else 0.0
    return dict(status=status, params_b=shape.params_b, peak_rank_gib=max(rank_peaks),
        host_floor_gib=host_floor,
        cluster_available_gib=available * nodes / stages, wire_bytes_tok=wire, wire_scope=scope,
        ssd_bytes_tok=disk_bytes, kernel_ms=1000 * (sum(stage_times) + host_compute),
        net_ms=1000 * network_s, ssd_ms=1000 * disk_delay, single_tok_s=rate,
        aggregate_tok_s=aggregate, active_j_tok=joules, eur_mtok=eur)


METRICS = ("status", "params_b", "peak_rank_gib", "host_floor_gib", "cluster_available_gib",
           "wire_bytes_tok", "wire_scope", "ssd_bytes_tok", "kernel_ms", "net_ms",
           "ssd_ms", "single_tok_s", "aggregate_tok_s", "active_j_tok", "eur_mtok")


def overrides(obj, pairs):
    result = {}
    types = {f.name: f.type for f in fields(obj)}
    for pair in pairs:
        key, sep, raw = pair.partition("=")
        if not sep or key not in types:
            raise ValueError(f"unknown override {pair!r}")
        result[key] = types[key](raw)
    return replace(obj, **result)


def read_layout(path):
    allowed = {"layers", "cap_gib", "reserve_gib", "mem_gbs", "gflops", "layer_us",
               "active_w", "idle_w", "node_eur"}
    with open(path, encoding="utf-8") as stream:
        layout = json.load(stream)
    if not isinstance(layout, list) or not layout:
        raise ValueError("layout must be a nonempty JSON array")
    for node in layout:
        if (not isinstance(node, dict) or "layers" not in node or node.keys() - allowed
                or type(node["layers"]) is not int or node["layers"] < 1
                or any(type(value) not in (int, float) for value in node.values())):
            raise ValueError("invalid pipeline node in layout")
    return layout


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenario", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    p.add_argument("--family", nargs="+", choices=("dense", "moe"), default=["dense", "moe"])
    p.add_argument("--model", nargs="+", type=int, choices=SHAPES, default=list(SHAPES))
    p.add_argument("--nodes", nargs="+", type=int, choices=NODES)
    p.add_argument("--strategy", nargs="+", choices=STRATEGIES, default=list(STRATEGIES))
    p.add_argument("--streams", type=int, default=1, help="pipeline KV slots and in-flight cap")
    p.add_argument("--group", type=int, default=2, help="hybrid expert ranks per stage")
    p.add_argument("--legs", type=int, choices=(1, 2), default=2,
                   help="wire legs for a worker boundary (2 = coordinator relay)")
    p.add_argument("--expert-legs", type=int, choices=(1, 2), default=2,
                   help="wire legs per one-way expert dispatch (2 = relay)")
    p.add_argument("--layout", help="JSON pipeline stages with layers and optional local budgets/rates")
    p.add_argument("--remote-fraction", type=float, default=None,
                   help="remote expert selections per selection within a multi-rank group")
    p.add_argument("--set", action="append", default=[], metavar="INPUT=VALUE")
    p.add_argument("--shape", action="append", default=[], metavar="FIELD=VALUE")
    args = p.parse_args()
    writer = None
    output = io.StringIO()
    try:
        layout = read_layout(args.layout) if args.layout else None
        if layout is not None and args.strategy != ["pipeline"]:
            raise ValueError("--layout requires --strategy pipeline")
        node_counts = args.nodes or ([len(layout)] if layout else NODES)
        for scenario in args.scenario:
            cfg = overrides(SCENARIOS[scenario], args.set)
            for model in args.model:
                shape = overrides(SHAPES[model], args.shape)
                validate(shape, cfg, args.streams, args.group, args.legs, args.remote_fraction)
                for family in args.family:
                    for strategy in args.strategy:
                        if family == "dense" and strategy in ("expert", "hybrid"):
                            continue
                        def case(n):
                            return evaluate(shape, cfg, family, strategy, n, args.streams,
                                            args.group, args.legs, args.remote_fraction, layout, args.expert_legs)
                        baseline = None if layout else case(1)["aggregate_tok_s"]
                        for n in node_counts:
                            row = dict(scenario=scenario, study_class_b=model, family=family,
                                       strategy=strategy, nodes=n)
                            metrics = case(n)
                            row.update({k: metrics[k] for k in METRICS})
                            row["profile"] = json.dumps({
                                "inputs": args.set, "shape": args.shape, "streams": args.streams,
                                "group": args.group, "legs": args.legs, "expert_legs": args.expert_legs,
                                "remote_fraction": args.remote_fraction, "layout": layout,
                            }, separators=(",", ":"))
                            rate = row["aggregate_tok_s"]
                            row["scaling_efficiency"] = rate / (n * baseline) if baseline and rate else None
                            if writer is None:
                                writer = csv.DictWriter(output, fieldnames=row)
                                writer.writeheader()
                            writer.writerow({k: f"{v:.7g}" if isinstance(v, float) else v
                                             for k, v in row.items()})
        if writer is None:
            raise ValueError("no strategy applies to the selected model family")
    except (OSError, ValueError) as error:
        p.error(str(error))
    sys.stdout.write(output.getvalue())


if __name__ == "__main__":
    main()
