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
