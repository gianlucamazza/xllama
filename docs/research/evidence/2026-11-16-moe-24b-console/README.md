# LFM2.5-24B-A2B streamed on a project Series S (2026-11-16)

First run of the optional SSD expert-streaming path on a console the project
owns. It reproduces the CPU stage that vetax-by reported in
[Discussion #435](https://github.com/gianlucamazza/xllama/discussions/435)
(#481, CPU stage; plan [#482](../../../superpowers/plans/2026-11-16-ssd-moe-upstream.md)).

## Identity

| Item     | Value                                                                                                                                                                                                         |
| -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Console  | Series S in Developer Mode, Game designation (quota limit 5368709120 bytes = 5120 MiB), operator-declared                                                                                                     |
| Package  | `GianlucaMazza.xllama` 1.7.0.1640, CI run [43871242484](https://github.com/gianlucamazza/xllama/actions/runs/43871242484), in-place upgrade                                                                   |
| Source   | tag `evidence/2026-11-16-moe-24b-console` (`75fe271`, a test build, not for `main`): the seventeen fork patches of `61bf63a`, byte-identical to its `PATCH-LOCK.json`, on base `4db5152`, llama.cpp `d812350` |
| Model    | `LiquidAI/LFM2.5-24B-A2B-GGUF` @ `7be3a8c`, `LFM2.5-24B-A2B-Q4_K_M.gguf`, 14526258464 bytes, SHA-256 `d3334405…` (checked before and after each run)                                                          |
| Profiles | fork's `cache-a2-v1` (2 expert slots) then `cache-b512-v1` (512 slots); one CPU worker, nonfused path, context 512, fixed Russian greeting, greedy                                                            |

The model was downloaded by the 1.7.0.1621 build through a LocalState
manifest override (with the #440 and #447 downloader fixes), then the test
package was installed over it.

## Result

| Profile | Stop    | Samples | First token |       Rate | Expert requests |     Cache hits | Read (logical) | Max sampled usage |
| ------- | ------- | ------: | ----------: | ---------: | --------------: | -------------: | -------------: | ----------------: |
| A2      | EOS/EOT |     199 |      67.6 s | 0.72 tok/s |          112176 |              0 |      212.86 GB |           805 MiB |
| B512    | EOS/EOT |     199 |      48.5 s | 1.00 tok/s |          112176 | 39330 (35.1 %) |      138.34 GB |          2062 MiB |

- Both runs end on the model's own EOS after a closed `<think>` block, with a
  final answer, normal owner/file/writer teardown and SHA verified after.
- The 199 generated token ids are identical between A2 and B512.
- The figures match the fork's report (69.5 / 49.5 s, 1.43 / 1.02 s per
  token, 0 / 39330 hits of 112176, 212.86 / 138.34 GB, about 2.01 GiB).
- "Read" is application-level reader bytes, not measured SSD traffic.
  "Max sampled usage" is the in-process MemoryManager reading, not a peak.

## What this does not show

- Numerical parity: identical tokens across cache sizes are not a
  resident-versus-streamed logit comparison (#480).
- Speed: one worker, nonfused, 512-token context, 48-token prompt. The rate is
  a capacity result, not a comparison with the dense peer of #273.
- GPU: the selected-expert GPU path was not run.

## Files

- `a2.output.json`, `b512.output.json`: the pilot's output records (schema
  `xllama-cache-ab-output-v1`), unmodified.
- `a2.trace.bin`, `b512.trace.bin`: the pilot's ordered cache traces.
- `a2.pilot.log`, `b512.pilot.log`: the `[cpu-pilot]` lines of `xllama.log`
  for each run, without the 250 ms runtime quota samples.
- `result.json`: identities, the table above, file digests.
