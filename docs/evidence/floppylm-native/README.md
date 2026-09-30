# FloppyLM native functional validation — 2026-09-30

## Scope and provenance

[Functional evidence](functional.json) records the exact source, CI run, MSIX digest,
bundle identities, configurations, data hashes, accounting, resume and restoration results.
The tested runtime is commit `6869867f8ada05b7b6c7970486c8989cecd41d2d`, MSVC
package `1.6.0.1106`, [CI run 36708622018](https://github.com/gianlucamazza/xllama/actions/runs/36708622018).
Linux, both Windows variants and CodeQL passed. Later branch changes add verification
and evidence only; they do not alter this runtime.

These are synthetic functional fixtures. They establish neither E0 saturation nor
scientific quality, byte-budget parity between candidate models, or a performance advantage.
Linux used a Debug build; Xbox used the Release MSVC package. Their timings are not comparable.

## Results

| Fixture | Configuration                         | Export bytes at T / 2T | Targets per evaluation |
| ------- | ------------------------------------- | ---------------------- | ---------------------- |
| d16     | GELU, ternary, row16, ctx 8           | 4,114 / 4,114          | 34                     |
| d80     | SwiGLU, 2-bit, row16, QK-norm, ctx 64 | 58,328 / 58,318        | 272                    |

Both fixtures trained on Xbox, resumed an intermediate checkpoint, and reproduced
both artifacts and all cooldown metrics exactly. Linux and Xbox produced the same
four artifact byte streams in these fixtures. The independent frozen Python reader
verified every section, configuration, round trip and sliding bpb (difference < 1e-5).
The four `.flp` files alongside this document are the downloaded Xbox artifacts.

Each console trial checked the exact package identity, uploaded and downloaded the
bundle, and restored 39 control/generated files. Its fixture directories were removed
and the app returned to its initial stopped state. Private owner backups and logs are
retained outside the repository. The [final restoration proof](restoration.json) verifies the six original files, root
and training inventories and stopped process state. Base runtime sources were restored
through package `1.6.0.1107`, rebuilt from `ed242cc` with a higher CI revision. This
restores the original runtime source without uninstalling; the package version and
bytes deliberately differ from the initial `1.6.0.1102` installation.

## Numerical diagnosis

[The reference matrix](reference-matrix.log) reports every update difference and the
number of coordinates exceeding the superseded blanket tolerance. Four larger fixtures
exceed that old criterion; none is hidden or removed.

[The independent float64 diagnostic](numeric-diagnostic.log) isolates the d80 QK-norm
coordinate: reference fp32 gradient `2.631563855715058e-10`, native fp32
`-1.3154931366443634e-08`, explicit-attention float64 `-1.4031063257404525e-08`.
AdamW amplifies the cancellation difference into a `0.00030852668` weight gap.
This observation applies to the recorded coordinate, not to global backend superiority.

[Accepted ADR 0005](../../adr/0005-floppylm-numeric-validation.md) defines the resolution:
unchanged forward/gradient tolerances, identical-gradient optimizer tests, and a
separately measured end-to-end sensitivity envelope. Epsilon, configurations and
quantized forward values are unchanged. The old update criterion is explicitly superseded.

## Reproduction

Use the [native training runbook](../../../training/floppylm/README.md) and the config,
spec and environment in `functional.json`. Train data is `bytes(range(256)) * 4`.
Validation is `bytes(range(35))` for d16 and
`bytes(range(256)) + bytes(range(17))` for d80. The pinned preparation tool exports
seed 7 weights and offsets. Dataset and bundle hashes must match the recorded values.
Run `verify.py` on the result directory and its original bundle; it uses the frozen
reference and emits the hashes and independently computed metrics in this evidence.
