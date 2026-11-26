# Context architecture stack: console validation (2026-11-25)

Series S, Developer Mode, Game designation (`app_memory_limit_mb=5120`). Packages
from `build-uwp` on `feat/kv-memory-admission`, the tip of the stack
#535 → #567 → #554 → #568 → #569 → #571, installed in place (LocalState and
provisioned models kept).

## Gates

Gates were first run on 1.8.1.1805. `coderpaste` failed there and passed on
1.8.1.1814, which adds the log line for the GUI refusal. Every log in this
directory comes from 1.8.1.1814.

| Gate                             | Result  | Note                                                                                                                                                                                            |
| -------------------------------- | ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `validate-console.sh longchat`   | PASS    | #169 shift over turn boundaries, no context-full error                                                                                                                                          |
| `validate-console.sh kvsnap`     | PASS    | returning turn prefills a delta (19 tok, 3% of cold)                                                                                                                                            |
| `validate-console.sh genroom`    | PASS    | 513 tokens left for a 512-token reply                                                                                                                                                           |
| `validate-console.sh coderpaste` | PASS    | 1805: FAIL. The 24 KB paste was refused correctly ("DOES NOT FIT", app alive), but #567 moved the GUI refusal before the session, so "prompt too long" was no longer logged. 1814 logs it; PASS |
| `validate-api.sh chat`           | PASS    | includes the single-slot 503                                                                                                                                                                    |
| `validate-api.sh stream`         | PASS    |                                                                                                                                                                                                 |
| `validate-api.sh tools`          | PASS    |                                                                                                                                                                                                 |
| `validate-api.sh embed`          | NOT RUN | `embed-nomic-v2-moe` is not provisioned on this console (404 before any load); environment, not a result                                                                                        |

## Memory admission (#571)

`memory-1814.txt` holds the `memory estimate` lines and the load working-set
lines from the app log of the run. No gate load was refused.

On the CPU path the estimate exceeds the working set after the context:

| Load                       | Estimate (host) | Working set after context |
| -------------------------- | --------------- | ------------------------- |
| `lfm25-350m`, n_ctx 2048   | 370 MB          | 344–355 MB                |
| coding model at n_ctx 4096 | 730 MB          | 567–573 MB                |

The working-set figure includes the app's base working set. The only GPU load,
a model with all 17/17 layers on D3D12, logged host 164 MB plus device 660 MB,
and its working set reached 886 MB. So D3D12 buffers occupy the process's
system memory, as the D2-r2 peak-RAM measurements in `docs/gguf-gpu-decode.md`
already showed. The admission in 1814 checked device bytes only against the GPU
budget. The fix in #571:

- the D3D12 backend reports an integrated GPU (`IGPU`);
- the dry run counts its buffers against the app limit as well as against the
  GPU budget;
- the estimate line logs the working set in use before the load, so the next
  run calibrates `kSessionMemoryMarginMb` directly.

## Calibration on 1.8.1.1819 (the fix above)

Each model was loaded by one API turn (`Say hi.`, 16 tokens). "Grown" is the
working set after the context minus the "in use" figure the estimate line now
logs. `memory-1819.txt` holds the lines. After that, `longchat` and
`coderpaste` both PASS (`console-*-1819.log`).

| Model                             | n_ctx | Estimate (host, incl. D3D12) | In use | After context | Grown   |
| --------------------------------- | ----- | ---------------------------- | ------ | ------------- | ------- |
| `lfm25-1.2b-instruct` (GPU 17/17) | 2048  | 825 MB (device 660)          | 126 MB | 891 MB        | 765 MB  |
| `lfm25-350m`                      | 2048  | 370 MB                       | 123 MB | 365 MB        | 242 MB  |
| `qwen25-coder-1.5b`               | 4096  | 1361 MB                      | 139 MB | 1216 MB       | 1077 MB |
| `llama32-3b`                      | 2048  | 1969 MB                      | 118 MB | 1858 MB       | 1740 MB |
| `lfm25-8b-a1b`                    | 2048  | 3693 MB                      | 157 MB | 3593 MB       | 3436 MB |

The estimate alone covers the growth on every load, so
`kSessionMemoryMarginMb` (192) is headroom on top. The largest catalogue model,
`lfm25-8b-a1b`, is admitted with 157 + 3693 + 192 = 4042 MB against the
5120 MB limit.
