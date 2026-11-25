# Image generation next to a resident chat model (#548)

Series S, Developer Mode, Game designation (`app_memory_limit_mb=5120`).
Driver: [`scripts/measure-diffusion-coexistence.sh`](../../../scripts/measure-diffusion-coexistence.sh).
One cell per model. Each cell:

1. restarts the app;
2. makes the model resident with one `/v1/chat/completions` turn;
3. runs `POST /v1/images/generations` (SD-Turbo fp16, 1 step, seed 42, 512×512).

The `none` cell is the headless `diffuse.flag` pipeline, with no chat model.

Columns in `cells.csv`:

- `process_peak_ws_mb` is the pipeline's own `peak_working_set_mb()`, taken from a `diffuse-result.csv` row written during the cell.
- `max_sampled_private_ws_mb` is the Device Portal's PrivateWorkingSet, sampled every 2 s, so it can miss short peaks.
- GPU memory is not observable from outside the app. A GPU or host allocation failure shows up as a failed generation and as its line in the cell log.

Logs keep only the lines added during the cell, with paths redacted (`scripts/redact_recorded_path.py`).

## Before: `main` 13afe27, CI package 1.8.1.1776 (`cells.csv`)

| Cell                  | Working set after context | Outcome                                        | Peak WS |
| --------------------- | ------------------------- | ---------------------------------------------- | ------- |
| none (headless)       | n/a                       | ok                                             | 2537 MB |
| `lfm25-350m`          | 345 MB                    | ok                                             | 3555 MB |
| `lfm25-1.2b-instruct` | 889 MB                    | ok                                             | 3555 MB |
| `qwen25-coder-1.5b`   | 1191 MB                   | ok                                             | 3741 MB |
| `llama32-3b`          | 1838 MB                   | ok                                             | 4442 MB |
| `lfm25-8b-a1b`        | 3574 MB                   | **fail**: UNet load `bad allocation`, HTTP 500 | n/a     |

The pipeline adds 2550–3210 MB over the resident session. The largest increase, 3210 MB, is in the `lfm25-350m` cell. With `lfm25-8b-a1b` resident, the 5120 MB limit leaves too little room.

## After: `fix/diffusion-memory-admission`, CI package 1.8.1.1796 (`after-fix/cells.csv`)

`include/xllama/diffusion/residency.h` releases the resident session before the pipeline only when the working set plus `kDiffusionHeadroomMb` (3250, the largest measured increase rounded up) would exceed the app limit. The package was installed in place, so LocalState and the provisioned models were kept.

| Cell           | Session                   | Outcome | Peak WS |
| -------------- | ------------------------- | ------- | ------- |
| `lfm25-8b-a1b` | released (one log line)   | ok      | 3578 MB |
| `llama32-3b`   | kept (1838 + 3250 ≤ 5120) | ok      | 4331 MB |
| `lfm25-350m`   | kept                      | ok      | 3555 MB |

The next chat turn after a release reloads the model. Small models never pay that reload.
