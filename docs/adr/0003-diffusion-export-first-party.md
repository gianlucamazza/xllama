# ADR 0003: Export SD-Turbo with a first-party torch.onnx exporter

Status: Proposed (2026-09-30)

Not applied. Needs explicit owner acceptance before any structural change to
`diffusion/` (#263).

## Context

The host-only SD-Turbo export toolchain (`diffusion/requirements.txt`) pins
`torch==2.9.1`, `transformers==4.57.6`, `optimum-onnx[onnxruntime]==0.1.0` and
`diffusers==0.39.0`. Two open Dependabot alerts sit on that set:
transformers GHSA-xrqw-3rrv-vx5w (patched in 5.10.0) and torch
GHSA-rrmf-rvhw-rf47 (patched in 2.13.0).

The pin cannot move piecemeal. Checked 2026-09-30:

- `optimum-onnx` 0.1.0 (latest release, 2025-12-23) requires
  `transformers>=4.36,<4.58.0`. Its `main` branch keeps the same cap
  (`pyproject.toml`: `optimum~=2.2.0`, `transformers>=4.36,<4.58.0`).
- Current releases: torch 2.14.0, transformers 5.17.0, diffusers 0.40.0,
  onnx 1.23.1, onnxruntime 1.30.0.
- optimum-onnx 0.1.0 also drives the deprecated TorchScript tracer
  (`dynamo=False`), which is the reason for the torch 2.9 pin.

So no supported resolution exists while `optimum-cli export onnx` is in the
path. The packages are not shipped in the MSIX, and export is restricted to
first-party `stabilityai/sd-turbo` weights. That is a mitigation, not a fix.

`optimum` is used in two places:

- `export_sd_turbo.sh` calls `optimum-cli export onnx --task text-to-image`.
- `generate_onnx.py` uses `ORTStableDiffusionPipeline` for a host sanity image.

`export_taesd.py` already exports with `torch.onnx.export` directly.

The console contract is the component set `convert_fp16.py` produces
(`text_encoder`, `unet`, `vae_decoder`) and their I/O names, as read by
`uwp/diffuse.cpp` and `diffusion/validate_pipeline.py`: `input_ids` →
`last_hidden_state`; `sample`, `timestep`, `encoder_hidden_states` →
`out_sample`; `latent_sample` → `sample`.

## Decision

Remove `optimum` from the export toolchain and export the three components with
a first-party script, `diffusion/export_sd_turbo.py`:

- Load `stabilityai/sd-turbo` with `diffusers` (`CLIPTextModel`,
  `UNet2DConditionModel`, `AutoencoderKL.decode`) and call
  `torch.onnx.export(..., dynamo=True)` per component. Keep the I/O names,
  dtypes and dynamic axes above byte-for-byte. The `timestep` shape and dtype
  must match what `diffuse.cpp` feeds today.
- Pin a coherent patched set: `torch>=2.13` (CPU wheel), `transformers>=5.10`,
  `diffusers>=0.40`, `onnx`, `onnxscript` (dynamo exporter),
  `onnxruntime` for host validation.
- Replace the `ORTStableDiffusionPipeline` sanity path in `generate_onnx.py`
  with `validate_pipeline.py`, which already runs the ONNX graphs directly.
- Keep `convert_fp16.py`, `scripts/merge_onnx_external_data.py` and
  `export_taesd.py` (move TAESD to `dynamo=True` in the same change only if its
  output also passes the comparison below).

## Acceptance (before closing the alerts)

1. The whole pinned environment resolves and installs in isolation with no
   overridden constraints.
2. Before the trial, record fixed inputs and tolerances: prompt set, seed,
   steps, and per-component max-abs/cosine bounds against the current export.
3. Export with the documented first-party inputs. Check external-data merging,
   tensor shapes, opset, and that `convert_fp16.py` still produces
   DirectML-loadable fp16 graphs.
4. Compare component outputs and final images against the current export
   within the pre-recorded tolerances (`gen_golden_vectors.py` /
   `golden_vectors.json`).
5. On Xbox Series S, with the CI MSVC package, pass the TAESD/diffusion live
   gate (`validate-console.sh taesd`) with peak-memory evidence.
6. Update `diffusion/README.md`, `requirements.txt` and the runbook. Rerun
   security scanning, and close the alerts only once the patched graph is shown.

## Consequences

The export no longer depends on optimum's release cadence, and the torch pin
can follow security fixes. The cost is a small exporter the project owns (three
`torch.onnx.export` calls and their shapes). The dynamo exporter may emit
different op patterns, and DirectML compatibility is only proven by step 5. A
failed comparison leaves the current pins in place with the alerts documented.

## Alternatives

- Wait for optimum-onnx to lift the transformers cap. There is no date, and
  `main` still carries the cap.
- Override the dependency constraints. The issue rules this out, and it
  produces an unsupported graph.
- Dismiss the alerts on the host-only argument. That is a mitigation claim,
  not a remediation.
