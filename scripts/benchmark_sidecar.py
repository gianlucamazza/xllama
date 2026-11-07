#!/usr/bin/env python3
"""Shared schema-1 provenance checks for benchmark writers and validators."""

import json
import math
import re


def validate_sidecar(data):
    """Raise ValueError for metadata that cannot support a publishable claim."""
    if not isinstance(data, dict):
        raise ValueError("sidecar must be a JSON object")
    required = {
        "schema_version", "model", "model_bytes", "model_sha256", "quant", "build",
        "hardware", "n_ctx", "prompt_sha256", "sampling", "warmup_runs",
        "recorded_runs", "thermal_rule", "ambient_c", "power",
    }
    missing = sorted(required - data.keys())
    if missing:
        raise ValueError(f"missing sidecar fields: {', '.join(missing)}")
    if type(data["schema_version"]) is not int or data["schema_version"] != 1:
        raise ValueError("unsupported sidecar schema")
    for field in ("model", "quant", "build", "hardware", "thermal_rule"):
        if not isinstance(data[field], str) or not data[field].strip():
            raise ValueError(f"{field} must be a non-empty string")
    for field in ("model_sha256", "prompt_sha256"):
        if not isinstance(data[field], str) or not re.fullmatch(r"[0-9a-f]{64}", data[field]):
            raise ValueError(f"{field} must be lowercase SHA-256")
    for field, minimum in (("model_bytes", 1), ("n_ctx", 1), ("warmup_runs", 1), ("recorded_runs", 3)):
        if type(data[field]) is not int or data[field] < minimum:
            raise ValueError(f"{field} must be an integer >= {minimum}")
    if not isinstance(data["sampling"], dict):
        raise ValueError("sampling must be a JSON object")
    try:
        json.dumps(data["sampling"], allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("sampling must contain finite JSON values") from exc
    _finite_number(data["ambient_c"], "ambient_c")
    power = data["power"]
    if not isinstance(power, dict):
        raise ValueError("power must be a JSON object")
    for field in ("wattmeter", "sample_interval_ms", "idle_w", "prefill_w", "decode_w"):
        if field not in power:
            raise ValueError(f"missing power field: {field}")
    if not isinstance(power["wattmeter"], str) or not power["wattmeter"].strip():
        raise ValueError("power.wattmeter must be a non-empty string")
    if type(power["sample_interval_ms"]) is not int or power["sample_interval_ms"] <= 0:
        raise ValueError("power.sample_interval_ms must be a positive integer")
    for field in ("idle_w", "prefill_w", "decode_w"):
        _finite_number(power[field], f"power.{field}")
        if power[field] < 0:
            raise ValueError(f"power.{field} must be non-negative")


def _finite_number(value, field):
    if type(value) not in (int, float) or (type(value) is float and not math.isfinite(value)):
        raise ValueError(f"{field} must be a finite number")
