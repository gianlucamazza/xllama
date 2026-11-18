# 2026-11-16 LFM2.5-8B-A1B resident rows on CI build 1.7.0.1652

Resident counterpart of the streamed 8B run
([evidence](../../../docs/research/evidence/2026-11-16-moe-8b-console/README.md)):
the H2 checkpoint re-measured, and the Q4_K_M file that the streamed path ran.

- Package: CI `xllama-appx` from run 43889558933, `main` `568e883`, MSIX
  SHA-256 `f816bcbe4abbbf19051feb2c409ebd5275793287c00a090e48df8abfad0a4d61`,
  installed in place over the test packages (LocalState kept).
- Harness: `scripts/bench-xbox-ort.sh` at `main`, `--hardware xbox-series-s
--threads 6 --runs 4 --n-predict 512 --ignore-eog --timeout 600`, default
  prompt `bench/prompts/standard-512.txt` (293 prompt tokens). Run 1 is warmup
  and dropped. The host tag is an operator declaration.
- No external power, ambient or thermal data, so no sidecar: 3-run medians,
  not headline measurements.

| row                           | CSV                                  | decode tok/s median (runs) | H2 (2026-07-30) | prefill tok/s median | TTFT ms | peak WS MB |
| ----------------------------- | ------------------------------------ | -------------------------- | --------------- | -------------------- | ------- | ---------- |
| LFM2.5-8B-A1B UD-IQ3_S CPU t6 | lfm25-8b-a1b-udiq3s-cpu-t6-noeog.csv | 14.42 (14.42/14.42/14.42)  | 14.50           | 61.25                | 4784    | 3560       |
| LFM2.5-8B-A1B Q4_K_M CPU t6   | lfm25-8b-a1b-q4km-cpu-t6-noeog.csv   | no run: context refused    | —               | —                    | —       | —          |

## UD-IQ3_S: decode unchanged, prefill 3.2× faster

Decode is 0.6 % below H2's 14.50 on MSIX 1.5.2.798, within its own spread;
H2's verdict (no faster than the dense 3B at about 14 tok/s) stands. Prefill
went from 18.97 to 61.25 tok/s on the same prompt length, which moves time to
first token from about 15 s to 4.8 s. The packages differ by four months of changes; the cause
was not isolated here.

## Q4_K_M: loads, cannot run resident

All four runs ended as harness timeouts (rows tagged `-timeout`, kept as
recorded). The app log (`logs/lfm25-8b-a1b-q4km-app.log`) shows why: the
weights load (1154.75 MiB CPU + 3754.12 MiB repacked, working set 5004 MB
after load) and then the 272 MiB prefill compute buffer cannot be allocated
inside the 5120 MiB app limit, so no context is created. This checkpoint runs
on a Series S only through the streamed path, at 2.47 tok/s with a 512-slot
cache.
