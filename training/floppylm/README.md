# Native FloppyLM training

The `floppylm` method trains the complete scalar E0 model with the shared ggml CPU
engine. It consumes a host-exported bundle, not a GGUF base model. The Xbox does
not require Python. This branch is experimental; functional validation is not an
E0 campaign, a saturation result, or a speedup claim.

Decisions: [ADR 0004](../../docs/adr/0004-floppylm-training.md) and
[ADR 0005](../../docs/adr/0005-floppylm-numeric-validation.md).

## Prepare and run

Host preparation and reference tests require PyTorch 2.11.0 and NumPy 2.4.4.
`config.json` contains `GPTConfig` overrides and `spec.json` contains `TrainSpec`
values. The immutable Python snapshot and file hashes are in
[reference/provenance.json](reference/provenance.json). Do not modify that snapshot
to make the native tests pass.

```sh
python training/floppylm/prepare.py \
  --config config.json --spec spec.json --train train.bin --val val.bin \
  --out /absolute/path/bundle --threads 1 --checkpoint-every 100
./build/linux-test/bin/xllama-cli --validate-train-job /absolute/path/bundle/job.json
bg ./build/linux-test/bin/xllama-cli --train-job /absolute/path/bundle/job.json
```

The bundle exports initialized fp32 master weights and a fixed stream of little-endian
64-bit byte offsets, using the pinned reference's initialization and NumPy RNG. Train,
validation and offsets are size- and SHA-256-checked before a run starts. Validation
uses the first 1 MiB of the supplied validation file; the test split is never opened.
The bundle digest binds checkpoints to weights, config, optimizer protocol and data.

The job accepts only `schema_version`, `name`, `method`, `device`, `bundle_path`,
`out_dir`, and optional `checkpoint_path`. LoRA fields and unknown keys are errors.
The output directory must not exist. On Xbox, set `device` to `device` and use
LocalState-relative bundle/output paths. Files referenced by a bundle have fixed
names next to `bundle.json`, so the same bundle bytes work on both hosts.

## Execution contract

- All parameters are trainable; tied embeddings contribute both gradient paths.
- Forward uses reconstructed scalar weights and fp16-rounded norm weights; AdamW
  updates fp32 master weights through straight-through gradients.
- Supported: ternary/2-bit/4-bit core, 4-bit embeddings, row16/row8log/tensor16 scales,
  GELU/ReLU²/SwiGLU, optional QK-norm. Full fp32 graph, CPU only, no inference KV cache.
- A fixed logical reduction order matches the pinned PyTorch CPU codec. It does not
  follow the compiling machine's SIMD width. The reduction structure is attributed
  in [PYTORCH-LICENSE](PYTORCH-LICENSE).
- One sequence per graph execution; gradients accumulate for the configured batch,
  then global clipping and one AdamW update run. No weight decay on norms.
- WSD warmup/stable/cooldown boundaries follow the frozen reference. Cooldowns copy
  model, moments and offset cursor; the trunk never advances through their batches.
- Sliding validation scores each target exactly once, including short data and tails,
  on the unpacked FLP2 model. Results include actual bytes per section and training
  tokens/FLOPs, including the reference's attention term.
- Native limits: 256 byte tokens, d ≤ 1024, layers ≤ 32, ff ≤ 16d, ctx ≤ 2048,
  batch ≤ 32, threads ≤ 16 on Linux (the existing console CPU budget on UWP), 16M parameters and a 512 MiB ggml tensor allocation cap.
  Invalid configurations fail explicitly; limits never silently change a job.

## Checkpoints and interruption

`checkpoint-NNNNNNNN.flc` is a checksummed FLC1 envelope containing CBOR model and
optimizer state, trunk/cooldown phase and cursor, completed artifacts, counters and
elapsed time. Each file is published by rename to an immutable name. FLP2 is strictly
inference-only and cannot be used as a native checkpoint.

To resume, copy the job JSON, set `checkpoint_path` to an existing `.flc`, and choose
a new `out_dir`. Completed cooldown artifacts are republished from the checkpoint.
SIGINT/SIGTERM on Linux and the device cancellation flag are checked between
microbatches. Cancellation checkpoints the last complete optimizer step. If a batch
was partially computed, its next resume recomputes that batch from the same offsets.

`running.json` records creation; `result.json` is the terminal record with
`completed`, `failed`, or `interrupted`. A process killed without a cooperative signal
may lack a terminal record; its immutable checkpoints remain resumable. Disk errors
are errors, never reported as successful completion.

`wall_limit_seconds` is at most 28,800 seconds and persists across resumes in the same
lineage. The operator must also account for the cumulative eight-hour experimental
budget across independent lineages. Unit/reference tests and compilation are separate
from experimental training. No unattended campaign is started by preparation or tests.

## Verification

```sh
cmake --preset linux-test -DPython3_EXECUTABLE="$(command -v python)"
cmake --build build/linux-test -j4
ctest --test-dir build/linux-test --output-on-failure
```

CTest includes native unit tests, 42 independent forward/gradient/codec cases,
45 codec boundary cases, six supplied-gradient AdamW steps, and WSD evaluation/resume/data-integrity tests.
ADR 0005 reports end-to-end update differences separately from optimizer correctness:
near-zero gradients can have different fp32 cancellation errors which AdamW amplifies.
`tests/floppylm/high_precision.py` provides an independent explicit-attention float64
diagnostic with the same quantized inputs and fp32 RMS epsilon. It is never a training
fallback.

For Xbox, use the exact MSVC artifact built for the branch. Preserve existing
LocalState control files before staging `training/job.json` and `train.flag`; restore
them after validation. Never use the fresh-install helper for this validation.

After deploying the selected CI package, the console harness verifies the expected
package, uploads and re-downloads the bundle to check transfer identity, runs both
cooldowns, retrieves FLP2 artifacts, and restores the sampled process/control-file
state even on failure:

```sh
source ~/.config/xllama/xbox-env
export XLLAMA_EXPECTED_PFN='the exact package full name from the selected CI artifact'
python scripts/validate-floppylm-console.py --bundle /absolute/path/bundle \
  --out /private/path/console-proof
```

The harness evidence directory contains private backups and console logs. Publish
only the selected numeric proof and artifact hashes, never the private backups.
