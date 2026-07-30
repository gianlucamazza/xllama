# UWP Constraints

> **SSOT for UWP/AppContainer constraints** (numbered §1–§13, plus §5b–§5f on DirectML and CPU threading): the measured GPU
> budget (3801 MB), the 2 GB per-file limit, disk budget, no-mmap, `887A0036`,
> `weakly_canonical`, thread cap, and the DirectML low-bit GEMM analysis. Other
> docs link to a `§n` here rather than restating these.

Platform limitations relevant to running LLM inference on Xbox Dev Mode, and how xllama addresses each one.

## 1. No POSIX `mmap`

**Problem**: `mmap()` is unavailable in the UWP sandbox.

**Status**: Not an issue for the ORT GenAI path — ORT loads the model internally using Win32 file APIs compatible with UWP. For the GGUF/llama.cpp path the `patches/0001` guards disable the desktop-only `_WIN32` mmap on the AppContainer, so GGUF weights are read into the heap (buffered). Enabling an AppContainer mmap via `CreateFileMappingFromApp`/`MapViewOfFileFromApp` was **tried and reverted (2026-07-14): no benefit**. ⚠️ The original attribution — "load is dominated by the AVX2 tensor repack" — predates PR #155: the repack path was dead code on Xbox at the time, and actually enabling it made loads _faster_ (lfm25 1645→1284 ms, `phase13-repack-{before,after}.csv`). The mmap conclusion (no benefit) stands on its own measurement; the mechanism claim does not.

## 2. Sandboxed Filesystem

**Problem**: UWP apps can only read/write their `LocalFolder` (`ApplicationData::Current::LocalFolder`) and locations explicitly granted by the user. Arbitrary path access (e.g. `C:\Users\...`) is denied.

**Workaround**: The bundled model is included as `DeploymentContent` in the MSIX and placed under `Package.InstalledPath\models\`. On first launch, `resolve_model_path` (`src/bridge/path_utils.cpp`) copies it to `LocalState\models\` so subsequent writes (log, bench CSV) land in the writable sandbox. Device Portal is used for one-off file transfers in dev.

## 3. No `dlopen` / Dynamic Backend Loading

**Problem**: UWP does not allow loading unsigned DLLs at runtime.

**Workaround**: ORT GenAI and its dependencies (`onnxruntime-genai.dll`, `onnxruntime.dll`, `DirectML.dll`) are NuGet packages restored at build time and included in the MSIX as **app-local** DLLs with `<DeploymentContent>true</DeploymentContent>`. No system-wide DLL loading. Which backends are compiled in is a build-time choice (`XLLAMA_USE_ORT` / `XLLAMA_USE_LLAMA`): the `default` and `llamacpp` variants link a single backend, while the shipping **`unified`** build links both and dispatches **per model at runtime** (`Backend::Auto` → `model_uses_llama_backend()`: `*.gguf` → llama.cpp, else ORT GenAI). See `docs/architecture.md`.

## 4. No JIT Compilation

**Problem**: UWP blocks `VirtualAlloc` with `PAGE_EXECUTE_*`, preventing JIT-compiled kernels.

**Impact**: Minimal. GGML and ORT GenAI kernels are pre-compiled C/C++. Only ggml-jit (experimental, unused here) is affected.

## 5. DirectML: Works on GPU (Headless) but Loses to CPU on Small Models

**Background — pool estimate corrected (2026-07-07)**: the GPU budget measured
in-app (`QueryVideoMemoryInfo(LOCAL).Budget`) is **3801 MB** (package designated Game — verified 2026-07-08, see §5) on
Series S — the earlier "~768 MB pool" was a coarse inference from the
Phi-3.5-mini OOM bracketing and is superseded. Disk was then the operative
sizing constraint, but the Dev Mode allocation was raised to **90 GB** on
2026-07-08 (§9, superseded note) — the binding limits now are the **2 GB ONNX
protobuf / per-file ceiling** (§8; §9 caveat) and the RAM/GPU budget.

**Effect on DirectML EP**: with a DML model variant and the headless path
(§7), the EP loads and executes on the GPU. Very large models can still OOM
(`0xC0000005`, null-deref in the DML allocator).

DirectML EP test results (Series S):

| Model        | Variant              | On-disk | Result                                                                                              |
| ------------ | -------------------- | ------- | --------------------------------------------------------------------------------------------------- |
| Phi-3.5-mini | GPU INT4 AWQ         | ~2.2 GB | GPU OOM (`0xC0000005`), v0.3.1 (2026-05-23)                                                         |
| SmolLM2-360M | INT4 CPU, DML config | 403 MB  | v0.3.1: silent CPU fallback (71.7 tok/s); v0.3.4: DML fused node fails (`80070057`, CPU-int4 graph) |
| SmolLM2-360M | INT4 **DML build**   | 285 MB  | ✅ **GPU execution, decode completes — 8.8 tok/s** (v0.3.4 headless, 2026-07-07)                    |

**Interpretation note** (updated 2026-07-07, evening): settled — **the DML EP
executes on the GPU** when the process is D3D12-clean. In headless bench mode
(v0.3.4, no XAML compositor) the profiled run yields **`VERDICT: GPU`** (fused
DML node on `DmlExecutionProvider`, 96% of kernel time) with 411 MB of weights
resident on the GPU (`gpu-mem post-load`). The earlier `887A0036` init failure
was the Agility-factory vs XAML-compositor device conflict (§7). Exp 1's
"~71 tok/s ≈ CPU baseline" was a silent CPU fallback on a pre-614 OS.

**Hardware utilization matrix** (2026-07-07, v0.3.6 with separate
prefill/decode timing; SmolLM2-360M, 2 runs each, median-ish first run shown):

| Variant  | prefill 285 tok | prefill ~1050 tok | decode (short ctx) | decode (long ctx) |
| -------- | --------------- | ----------------- | ------------------ | ----------------- |
| CPU int4 | **220 tok/s**   | 198 tok/s         | **68.0 tok/s**     | **50.9 tok/s**    |
| GPU int4 | 152 tok/s       | 334 tok/s         | 8.8 tok/s          | 8.3 tok/s         |
| GPU fp16 | 169 tok/s       | **354 tok/s**     | 46.8 tok/s         | 36.5 tok/s        |

Readings:

1. **Prefill: the GPU scales with batch size, the CPU does not.** ⚠️ The
   crossover claim that used to sit here — "between ~285 and ~1050 prompt
   tokens" — was an interpolation across the only two prompt lengths ever
   measured, on the pre-#91 asset that `dml_text_model_ok()` now excludes. A
   proper sweep of the shipping `-v2` asset (2026-07-21,
   `bench/results/phase12-dml-crossover.csv`, 10 prompt lengths, each row a median of 3 runs)
   shows the curve is **not** monotone and no single interpolated crossover
   exists. See §5b.
2. **The int4 decode collapse is a non-fused DML kernel, not a missing/CPU one**
   (corrected 2026-07-08 — see §12). The `MatMulNBits` op **is** present in the
   graph (225 nodes, `bits=4 block_size=32`, verified) and **does run on the GPU**:
   the profiled run shows the whole model as one `DmlFusedNode_0_0` on
   `DmlExecutionProvider` (96% of kernel time), with no `MatMulNBits` on CPU — so
   it is neither absent nor a CPU fallback. But DirectML's `MatMulNBits` kernel
   (`DmlOperatorMatMulNBits.cpp`) is a **non-fused** `DML_DEQUANTIZE`
   (int4→fp16 on GPU) + full `DML_GEMM`: it materialises the fp16 weight tensor,
   so int4 decode moves **more** bandwidth than plain fp16 (read 4-bit, write
   fp16, read fp16). That is why int4-DML (8.8) is _slower_ than fp16-DML (46.8
   on the pre-fix graph; 44.4 on the current `-v2` decomposed graph).
   The builder also targets DML with `int4_accuracy_level=0` (fp16 compute) while
   CPU gets `=4` (fused int8 MLAS `SQNBitGemm`) — the reason CPU int4 reaches 68.
   There is **no fused low-bit GPU GEMM** on DirectML through 1.15.x, so the
   ~180 tok/s "bandwidth ceiling" is **not** reachable on DML by any config we
   control; fp16 is the best DML decode config, and it still loses to CPU int4.
3. CPU decode degrades ~25% from short to ~1 k context; GPU fp16 similarly.

**Verdict**: CPU int4 remains the decode winner (68 vs fp16-DML 46.8 pre-fix /
44.4 on the shipping `-v2` graph vs int4-DML 8.8); GPU fp16 is superior only for
prompt-heavy prefill. The int4-DML gap is a
DirectML kernel-_design_ limit (non-fused low-bit GEMM), not a hardware limit and
not fixable by our quantization config — see §12 for the full analysis and the
config tests that confirm it. Effective bandwidth: CPU ~13 GB/s, GPU fp16
~34 GB/s, against a ~224 GB/s theoretical bus. **#91 note**: the throughput
numbers in this section were measured on the pre-fix fused-RMSNorm graph,
whose text output was numerically wrong (broken
`(Skip)SimplifiedLayerNormalization` kernel — `dml-rmsnorm-fix-runbook.md`).
The shipping `-v2` decomposed graph measures 236.7 prefill / 44.4 decode /
1268 MB — same conclusions (CPU int4 wins decode, GPU wins long prefill), and
GPU text routing is live again behind the routing threshold (see §5b).

### §5b — Prefill vs prompt length on the shipping `-v2` asset (2026-07-21)

`scripts/bench-prompt-sweep.sh`, SmolLM2-360M, `n_ctx` 2048, median of 3 runs
after a dropped warmup; every point below was re-measured at least once and
reproduced. Raw rows: `bench/results/phase12-dml-crossover.csv`.

| prompt tok | CPU prefill | GPU prefill | GPU prefill time | break-even answer | context allows |
| ---------- | ----------- | ----------- | ---------------- | ----------------- | -------------- |
| 172        | 259 tok/s   | 146 tok/s   | 1.18 s           | GPU never wins    | 1876           |
| 320        | 253         | 268         | 1.20 s           | 8 tok             | 1728           |
| 557        | 233         | 362         | 1.54 s           | 121 tok           | 1491           |
| 791        | 221         | **447**     | 1.77 s           | 291 tok           | 1257           |
| 1098       | 209         | 289         | 3.80 s           | 211 tok           | 950            |
| 1289       | 203         | **119**     | **10.4 s**       | GPU never wins    | 759            |
| 1384       | —           | 208         | 6.67 s           | —                 | 664            |
| 1480       | —           | 227         | 6.51 s           | —                 | 568            |
| 1574       | 196         | **636**     | 2.47 s           | 975 tok           | **474**        |
| 1671       | —           | 681         | 2.45 s           | —                 | 377            |

> **Superseded in part by §5c (2026-07-21, same day).** Finding 1 below reads the
> anomaly as a band in _prompt length_. It is not: a controlled experiment holding
> the prompt byte-identical and varying only `n_predict` showed the controlling
> variable is `max_length`. The measurements in this table are correct; the
> interpretation of finding 1, and the threshold in finding 3 that follows from
> it, are not. Read §5c before using either.

Three findings:

1. **There is a pathological band, roughly 1100–1500 tokens.** GPU prefill time
   rises to 3.8–10.4 s while the CPU stays on its monotone 5.2–8.0 s line. The
   1289-token point takes **10.4 s to prefill fewer tokens than the 1574-token
   point does in 2.5 s**. Each row is a median of 3 runs after a dropped warmup,
   and the 1098 / 1289 / 1574 points were re-measured as separate runs 3.5-16
   minutes later, reproducing to three digits (3.803/3.800 s, 10.83/9.95 s,
   2.473/2.477 s); CPU points taken in between stayed on their monotone line, so
   session drift does not explain the band. The mechanism is **not explained**:
   it is not memory pressure (the fastest point, 1574, has the largest working
   set at 2869 MB). Resolving it needs the per-node ORT profiler
   (`scripts/profile-dml-run.sh`, §11).
2. **The right criterion is two-dimensional, not a prompt-length threshold.**
   The GPU buys a one-off prefill saving and pays it back at a fixed rate on
   every generated token, so what matters is prompt length **and** answer
   length. The "break-even answer" column is that number: above it the CPU wins.
3. ~~**Past ~1550 tokens the GPU wins unconditionally**~~ — **retracted, see
   §5d.** The arithmetic here is right for a single turn of a single
   conversation and wrong for the app. It omits the asymmetric model load and
   the fact that DirectML cannot reuse a KV cache, and it measures total turn
   time when the app streams tokens, so what the user waits for is the first
   one. Corrected: from the second turn onward the CPU wins at every prompt
   length the context trimmer allows.

Caveats: measured for one model and one `n_ctx`. Both are fixed in the shipping
config, so the number is valid for what ships — but it is not a general law, and
it must be re-measured when the asset, the model, or `n_ctx` changes. Both are
now controllable from the console bench path (`bench_ctx.txt` /
`bench_npredict.txt`, exposed as `--ctx` / `--n-predict`), which is what #130
uses to test whether the band's edges track `n_ctx`. Note also that `n_gen_tok`
is blank for these rows: the column was added after they were taken
(`src/bridge/bench.cpp`), so turn times here are derived from the rates.

### §5c — The anomaly is in `max_length`, not prompt length (2026-07-21)

§5b read the slowdown as a band in prompt length. A controlled experiment says
otherwise. One prompt file, 1289 tokens, **byte-identical across every run**;
`n_ctx` 2048 unless stated; only `n_predict` varied, which moves
`max_length = min(n_ctx, n_prompt_tok + n_predict)`. Median of 2 runs after a
dropped warmup; pairs reproduced to three digits.

| `max_length` | `n_ctx`  | GPU prefill | time   | peak WS |
| ------------ | -------- | ----------- | ------ | ------- |
| 1297         | 2048     | 511 tok/s   | 2.52 s | 1905 MB |
| 1400         | 2048     | 472         | 2.73 s | 1906 MB |
| 1545         | 2048     | **150**     | 8.61 s | 1897 MB |
| 1650         | 2048     | 195         | 6.61 s | 1969 MB |
| 1801         | 2048     | **131**     | 9.83 s | 1970 MB |
| 1801         | **3072** | 131         | 9.83 s | 1971 MB |
| 1950         | 2048     | 212         | 6.09 s | 2483 MB |
| 2048         | 2048     | **612**     | 2.11 s | 1970 MB |

Taken twice, two hours apart, on two different MSIX builds (1.4.0.624 and
1.4.0.628); every point reproduced. The table is the second set — the one whose
rows carry `max_length` as a column. The first set is not committed: it predates
that column, so its rows differ only in `prompt_tok_s` and cannot be told apart.
That is the same defect `n_prompt_tok` fixed in #128, and the reason to
re-measure rather than annotate by hand.

1. **The prompt never changed, so this is not a prompt-length effect.** A 4.1×
   swing in prefill throughput on identical input tokens (511 → 131 → 612).
2. **`n_ctx` has no effect of its own.** The control row — `n_ctx` 3072 holding
   `max_length` at 1801 — reproduces the 2048 row to the digit (131.09 tok/s in
   both). This matches the code: on the ORT path `params.n_ctx` appears in exactly one
   place, the `min()` that computes `max_length` (`src/bridge/inference.cpp`,
   `src/bridge/session.cpp`). It is never passed to ORT GenAI as a context size;
   the model's real context is `context_length: 8192` in `genai_config.json`.
3. **The valley is interior, with both edges clean.** Fast at 1297–1400, fast
   again at 2048, slowest near 1800. It is a dip, not a step.
4. **It is not memory pressure, and saturating is not expensive.** The fastest
   row (2048) has a _smaller_ working set than the 1950 row (1970 vs 2483 MB).
5. **The shipping default sat inside the valley.** `n_predict` 256 on a
   1289-token prompt gives `max_length` 1545 — 150 tok/s where 612 was available.

**Consequence.** `Session::generate` now requests the full `m_n_ctx` as
`max_length` and bounds generation with the `n_predict` cap in `run_decode` —
which is what the KV-reuse chat path already did, so the two paths are now
symmetric. This is a straight win: faster and no more memory.

`run_inference` (CLI and bench) deliberately keeps the old `min()`. It is the
instrument that found this, and sweeping `max_length` independently of `n_ctx`
is the only way to re-measure the valley on a new asset or driver.

The mechanism remains **unknown** — suspected shape-bucketed kernel selection in
DirectML, which the per-node profiler (§11) cannot localise on its own because
the graph collapses into a single `DmlFusedNode_0_0` at 96% of kernel time.
Because §5b's finding 3 rests on the prompt-length reading, `token_threshold`
(1550) is calibrated on a misattributed variable. The re-derivation is in §5d,
and its conclusion is that there is no correct single prompt-length threshold —
the value is a product judgement, not a number to re-measure. It also has to stay
below the context trimmer's budget, which is #133.

Raw rows: `bench/results/phase12-maxlen-band.csv`.

### §5d — What the GPU is actually for: TTFT, turn one only (2026-07-21)

§5b compared **total turn time**. The app streams tokens (`on_token` → 40 ms
flush in `FlushTokenBuffer`), so that is the wrong quantity: what the user waits
for is the **first** token, and the rate afterwards is invisible as long as it
beats reading speed (~10 tok/s). Both backends clear that by 3–6×.

Read as TTFT, the two backends do not compete — they own different turns.

| P = 1400 | turn 1     | turn 2+               |
| -------- | ---------- | --------------------- |
| CPU      | 7.03 s     | **0.11 s** (KV reuse) |
| DML      | **2.32 s** | 1.56 s                |

Three things the earlier criterion left out, all measured 2026-07-21
(`bench/results/phase12-threshold-rederivation.csv`, `phase12-kv-reuse.csv`):

1. **Asymmetric model load** _(historical — eliminated by PR #161)_. At
   measurement time `EnsureSession` kept one session (`avoid 2× model in RAM`,
   and the working sets confirm it: 1303 + 2869 MB do not coexist) and a
   GPU-routed turn loaded the CPU model to tokenize, then destroyed it to load
   DirectML: 2786 ms against 1570 ms. Since PR #161 the routing count never
   loads a model (resident-session count or chars/5.0 estimator), so this term
   of the criterion no longer applies to current builds; the ownership moved
   to the process-wide `session_hub()` (PR #164).
2. **DirectML still rejects continuous decoding.** Verified, not assumed:
   `prefill2_reuse_ms = 0.0`, `n_p2_reuse = 0`. So it re-prefills the whole
   context every turn while the CPU reuses its KV — 68.8× on the measured turn 2.
3. **Routing is sticky**, decided on turn 1, when the app cannot know whether the
   conversation will continue.

Net, in total turn time (positive = DirectML ahead):

| P    | N=1         | N=2     | N=3     |
| ---- | ----------- | ------- | ------- |
| 1250 | **+0.64 s** | −2.43 s | −5.66 s |
| 1400 | **+1.54 s** | −1.57 s | −4.94 s |
| 1571 | **+2.72 s** | −0.55 s | −4.01 s |
| 1685 | **+3.13 s** | −0.25 s | −3.71 s |

The per-turn breakdown at 1400 says why, and it is not the valley: prefill costs
DirectML 1.55 s per extra turn (no KV reuse), **decode costs it 2.05 s per turn,
every turn**, for the §12 reason — no fused low-bit GEMM on DirectML. Decode
dominates and never amortises.

**Consequence.** DirectML routing pays for a _single-turn_ conversation above
~1250 tokens and loses from the second turn at every length the context trimmer
allows. Whether that is worth taking depends on how many long-prompt
conversations are one-shot, which is not measurable from here — hence the TTFT
instrumentation added to the UI, which makes it observable in real use.

### §5e — Per-process warm-up, DirectML only (2026-07-21)

Same `Session::generate` call, same timer, differing only in call ordinal within
the process:

| prompt → prompt | cold      | warm       | corrected for length |
| --------------- | --------- | ---------- | -------------------- |
| 1380 → 1495     | 597 tok/s | 1021 tok/s | **1.64×**            |
| 792 → 907       | 452 tok/s | 839 tok/s  | **1.72×**            |
| CPU control     | 198 tok/s | 198 tok/s  | 1.00×                |

Strong evidence for lazy kernel compilation on first use, which is the standing
hypothesis for the §5c valley. Two consequences: **every DirectML prefill figure
in `bench/results/` is a cold-process number** (the bench path bypasses
`Session` and never warms), and the app pays the cost once per model load, not
once per turn — since PR #158+#164 that load-time cost is paid by the warm-up
during pre-load, so **in-app turns run at the warm regime**.

**2026-07-25 addendum (PR #158).** Re-measured on the Session/LAN-API path
(960-token request, same process, build 1.4.0.675): turn-1 5.90 s vs turn-2
1.77 s wall, and decode goes 18.2 → 23.1 tok/s — the first-use cost hits
**decode kernels too**, not just prefill. A load-time warm-up now pays this
inside the "loading model" phase (`SessionParams::dml_warmup`,
`detail::create_ort`). First cut (build 678, ~2-token warm-up prompt, one
generate step) recovered only part of the gap — turn-1 prefill 682 vs 899 tok/s
warm, decode 19.4 vs 23.3 — because one `GenerateNextToken` _is_ the prefill
compute (zero decode steps ran) and a tiny prompt never exercises the
long-sequence prefill GEMMs. The shipped warm-up therefore uses a ~256-word
prompt and 3 generate steps; compilation is at least coarsely shape-dependent.
**Validated (build 680):** warm-up 2034 ms at load; turn-1 prefill 867 tok/s
(97% of the 898 warm rate) and decode 23.3 (full parity).

**Scope of the win** _(updated — the caveat below resolved the same day)_: at
first cut both surfaces created the session _lazily on the first turn_, so the
warm-up only moved the cost inside the same perceived wait (turn-1 wall 6.69 s
with warm-up vs 5.90 s without). **PR #164 closed exactly that gap**: sessions
now pre-load when a model becomes Ready (`PreloadSessionAsync`), so load +
warm-up run before the user's first send and the first turn pays
prefill+decode only — confirmed on-console after the 1.5.0.0 migration (first
DML request: prefill 873 tok/s on 961 tok, decode 23.4, both at warm parity).

### §5f — CPU threading: prefill does not scale, and t8 is worse than recorded (2026-07-21)

`docs/recommended-config.md` previously recommended `intra_op_num_threads: 4`
(now corrected to 6, below). That 4 came from a sweep whose every row has
`prompt_tok_s = 0.00` — a **decode** optimum. Prefill had never
been measured against threads, which mattered because prefill is what GPU routing
exists to compensate for.

P = 1380, `bench/results/phase12-cpu-threads.csv`:

| `intra_op_num_threads`      | prefill   | decode |
| --------------------------- | --------- | ------ |
| unset (what actually ships) | 198.9     | 46.9   |
| 4                           | 199.1     | 47.3   |
| **6**                       | **215.9** | 46.5   |
| 8                           | 87.0      | 9.9    |

- The "more threads help prefill" expectation is **refuted here**: 8 threads hits
  the Series S pathology first. Best is 6, worth +8.5% — not the ~2× that would
  have removed the case for GPU routing.
- The t8 pathology is **worse than documented**. It was recorded as a decode
  collapse attributed to bandwidth saturation; it takes prefill down 2.3× as
  well, which bandwidth does not explain.
- The shipped asset sets **no** `intra_op_num_threads` at all, so the documented
  recommendation of 4 has never run in production. t4 and the default measure the
  same, so nothing was lost — but the doc described a configuration nobody ran.

**2026-07-25 — the conditional 3-length sweep ran**
(`bench/results/phase12b-threads-sweep.csv`, build 1.4.0.675, pristine device
config verified first — a stale t4 swap from an earlier sweep was found on the
device and removed):

| P (tok) | unset prefill / decode | t4           | t6               |
| ------- | ---------------------- | ------------ | ---------------- |
| 39      | 240.5 / 84.5           | 237.3 / 80.6 | **251.1** / 80.7 |
| 285     | 244.8 / 74.3           | 246.9 / 75.5 | **256.3** / 72.6 |
| 960     | 221.4 / 51.8           | 221.4 / 51.9 | **234.8** / 51.2 |

t6 prefill +4.4% / +4.7% / +6.1% — consistent, but below §5f's single-length
+8.5%. Decode deltas sit inside the closing-control drift (unset re-measured at
the end: 245.2 / 72.0 vs 244.8 / 74.3 at the start, ≈ −3% decode over the
session), so decode is neutral within noise. t4 ≈ unset confirmed on all three
lengths. **Shipped 2026-07-25** with the 1.5.0.0 identity migration (the forced
re-provision was the "next models-v1 republish" the ship condition bundled
with); the release's `genai_config.json` now sets `intra_op_num_threads: 6`.

**2026-07-26 — the GGUF path had its own thread gap all along (#168).**
Everything above concerns the ORT knob (`intra_op_num_threads`). On the
llama.cpp side the app set only `n_threads` (decode), never
`n_threads_batch` — and llama.cpp runs prefill on `n_threads_batch`, whose
default is 4 regardless of `n_threads`. So while these sweeps tuned ORT
prefill, **GGUF prefill ran on 4 of the 6 usable cores in every measurement
ever published**, the +62% repack rows included. Fixed in PR #177 (`n_threads_batch = n_threads`).
**Measured on-console the same day** (build 698 → 711, `lfm25-350m` Q4_K_M,
3 recorded runs per point, `bench/results/phase13b-threadsbatch-{before,after}.csv`):
prefill **390.7 → 438.1 tok/s (+12.1%)** at P=298 and **388.6 → 429.2
(+10.5%)** at P=1000, decode neutral (94.9 vs the pre-fix spread), peak RAM
unchanged, load slightly faster. No livelock at 6 prefill threads — the
measurement doubled as the t7/t8 spin-wait check. Every GGUF prefill row
recorded before build 711 is a 4-prefill-thread figure (see the
comparability note in `bench/README.md`).

### §5 (continued) — disk, availability and the App/Game lever

**Effect on disk**: models too large to fit the Dev Mode partition also fail before reaching `OgaCreateModel`. This is a distinct failure mode — see §9.

Disk budget failures (deploy-time or LocalState copy):

| Model        | Variant  | On-disk | Failure           |
| ------------ | -------- | ------- | ----------------- |
| Phi-3.5-mini | INT4 CPU | ~2.7 GB | Above disk budget |
| SmolLM2-1.7B | INT4 CPU | 1.4 GB  | Above disk budget |
| SmolLM2-360M | INT4 CPU | 403 MB  | ✅ Works (CPU EP) |

Note: DirectML itself _is_ available in Dev Mode (NuGet `Microsoft.AI.DirectML 1.15.4`). The memory pool constraint applies to model weight size, not to the API itself.

**Current approach**: CPU EP (`"provider_options": []` in `genai_config.json`) remains the default for the interactive app (decode-heavy chat). DML fp16 is measured-viable for prompt-heavy workloads and profiling tooling exists (§11). The remaining GPU/CPU unlock levers — larger models via no-bundle deploy,
llama.cpp CPU kernel A/B, per-workload routing — are tracked in `ROADMAP.md`
Phase 3.5. (int4-on-DML decode is **not** a lever: it is blocked by DirectML's
non-fused low-bit kernel, see §12.) See §7 for GPU pool detail and §9 for disk budget.

**Platform lever — App vs Game designation (settled 2026-07-08)**: Dev Home
can flip a sideloaded package between **App** and **Game** (tile → View
details → App type); Game grants Game OS resources (full GPU access, more
RAM). Checked on console: **xllama is already designated Game** — the
designation persists per package family across forward upgrades. Therefore
the measured figures in this document (3801 MB GPU budget, the utilization
matrix, the per-token DML dispatch overhead) are **Game-mode numbers**, and
the earlier assumption labelling them "App-mode" was wrong. Consequence: the
GPU decode gap cannot be blamed on App-mode scheduling — it is a DML/kernel
issue, consistent with the fused-int4 analysis above. Operational note:
re-check the designation after any package reinstall (it can reset to App).

## 6. Limited Thread Count

**Problem**: Dev Mode apps share CPU resources. Xbox Series S has 8 Zen 2 cores; typically 6–7 are available.

**Workaround**: `detect_threads()` in `src/bridge/platform.cpp` reads `hardware_concurrency()` at runtime and ORT GenAI respects the system thread pool. Thread count can be overridden via `InferenceParams`.

## 7. GPU Memory Pool — Detail

**Measured budget (2026-07-07, in-app `QueryVideoMemoryInfo(LOCAL).Budget`):
3801 MB** on Series S (package designated Game, verified 2026-07-08 — see §5). The "~768 MB" figure previously documented
here was inferred from OOM bracketing and is superseded by this direct
measurement.

When `OgaCreateModel` initialises the DirectML execution provider, the DML allocator attempts to reserve GPU memory for model weights. If the model's total weight size exceeds the available pool, the allocator returns a null pointer; subsequent use of that pointer produces a STATUS_ACCESS_VIOLATION fault.

The fault manifests before any inference call — at model load time. There is no recovery path short of using a smaller model or switching to CPU EP.

**The binding limit is the inference working set, not just the weights (measured 2026-07-15).** A native-DML Llama-3.2-1B fp16 (2.49 GB weights, GQA) **loads** fine — `gpu-mem post-load: 2878/3801 MB` — but every inference call then OOMs: `AppendTokenSequences … DmlCommittedResourceAllocator … 8007000E Not enough memory`, even at `context_length=2048`. The DML inference working set (graph-capture command allocators + prefill activations + the `past_present_share_buffer` KV buffer) needs more than the ~900 MB left after weights. So the usable DML-fp16 ceiling on Series S is set by **weights + working set ≤ 3801 MB**, which in practice caps fp16 at ~360-500 M (e.g. `smollm2-360m-dml-fp16-v2`, ~725 MB weights). Any fp16 model large enough to require the >2 GB external-data path (§8) is ≥~1 B → over this wall. (For **text** models the ceiling is live again: the `-v2` RMSNorm-decomposed asset routes GPU prefill — #91 postmortem in `dml-rmsnorm-fix-runbook.md`; the ceiling also governs diffusion.)

**Distinct failure mode — DML EP init `887A0036` in XAML apps** (2026-07-07,
GPU-truth run, ORT GenAI 0.13.2 / ORT 1.24.4 / DirectML 1.15.4) — **root cause
found at the exact source line and fixed architecturally in v0.3.4**:

`OgaCreateModel` threw `887A0036 DXGI_ERROR_ALREADY_EXISTS` at
`onnxruntime-genai/src/dml/dml_helpers.cpp(140)` (`CreateDmlObjects`): ORT GenAI
creates its D3D12 device through the **Agility SDK device factory** —
`ID3D12SDKConfiguration1::CreateDeviceFactory(614, module_path)` succeeds on
Xbox OS 26100 via the in-box runtime ≥ 614 (no app-local `D3D12Core.dll`
needed; verified absent from the MSIX), and the factory's `CreateDevice`
collides with the process-wide D3D12 device the **XAML compositor**
(D3D11on12) created at `Window.Activate()`. Two different D3D12 runtimes cannot
share a process. Not OOM, not the profiling config, not our telemetry
(reproduced 3× including after removing the `gpu_mem_info` pre-load probe).
Exp 1 (May) passed because the then-OS had in-box < 614, so ORT fell back to
plain `D3D12CreateDevice` (line 144) which coexists with the compositor device
— the OS update flipped the branch. NuGet **0.14.0 / 0.14.1** still lack the
fallback when the factory returns `ALREADY_EXISTS` (they predate the fix).

**Fix (v0.3.4): headless bench mode** — with `bench.flag` present, `wWinMain`
skips `Application::Start` entirely and runs `main_loop()` under a minimal
`CoreApplication` `IFrameworkView` (CoreWindow activated for the PLM watchdog,
no compositor, no in-process D3D12 device). Result: DML EP initialises, weights
load onto the GPU (411 MB), profiled kernels run on `DmlExecutionProvider`
(**`VERDICT: GPU`**). Config prerequisite: DML graph capture requires
`past_present_share_buffer: true` in `genai_config.json`.

**Upstream fix (merged + shipping via vendor pin)**: we patched
`CreateDmlObjects` to fall back to the system D3D12 runtime when the Agility
device factory cannot create a device. Upstream PR
[microsoft/onnxruntime-genai#2280](https://github.com/microsoft/onnxruntime-genai/pull/2280)
**merged 2026-07-13** onto Microsoft `main` (commit `ff53d6b9`). Validated on
console with a patched DLL (XAML path): the same XAML + DML scenario that threw
`887A0036` loads in 886 ms and completes decode at 8.8 tok/s; CPU path
unaffected (67.2 tok/s). **Gap:** the shipping NuGet pin is still GenAI
**0.14.1** (`a30f479`, 2026-06-02), which does **not** include #2280 — xllama
overlays a pinned `onnxruntime-genai.dll` from
[`vendor-dlls-v1`](https://github.com/gianlucamazza/xllama/releases/tag/vendor-dlls-v1)
(`vendor/onnxruntime-genai-patched/SHA256SUMS`, same doctrine as PatchedOrt).
Drop `-PatchedGenAI` when an official NuGet release includes #2280.

**Plain ORT DML coexists with the XAML compositor — ✅ MEASURED 2026-07-09.**
The `887A0036` conflict was measured with **ORT GenAI**, whose DML EP goes
through the Agility device factory. The diffusion pipeline (`uwp/diffuse.cpp`)
uses **plain ORT DirectML** (`OrtSessionOptionsAppendExecutionProvider_DML`,
ORT 1.24.4), and the headless requirement was _inherited_ from the GenAI finding,
never falsified for plain ORT. `diffuse-inproc.flag` (`App.cpp`) runs
`run_diffuse()` on a background MTA thread **inside the live XAML process**;
on console it ran the full SD-Turbo pipeline to a coherent 512×512 PNG **with
the compositor alive** — no `887A0036`, no OOM (te 1006 ms, UNet 2991 ms, VAE
1576 ms, **total 5.57 s**, actually faster than the 6.9 s headless run, warm GPU).
**Conclusion**: the device conflict is specific to GenAI's Agility factory;
plain ORT DML shares the compositor's device fine. Image generation can run
**in-app without the restart flow** — the headless `diffuse.flag` path is no
longer required for diffusion (kept for bench parity). GPU headroom held:
compositor + sequential-lifetime weights (~2.4 GB) inside the 3801 MB budget.

**Diagnosis**: SEH `0xC0000005` in `OgaCreateModel`. WDP minidump (`type=2`) and the `xllama.log` entry `OgaCreateModel failed: ...` confirm the cause.

**Source note**: the GPU budget (3801 MB) is measured per-process via
`QueryVideoMemoryInfo(LOCAL).Budget` with the package designated Game. The historical "~768 MB"
estimate came from OOM bracketing (Phi-3.5-mini vs SmolLM2-360M) and proved to
be a strong underestimate. We do not document the underlying Xbox OS memory
partition layout — treat any claim about the internal platform architecture as
informed inference unless backed by a Microsoft source.

## 8. AppContainer Filesystem Walk (`weakly_canonical`)

**Problem**: ORT Runtime 1.24.4 calls `std::filesystem::weakly_canonical()` in `ValidateExternalDataPath()` for models that have a separate `.onnx.data` file. MSVC STL implements `weakly_canonical` by walking path segments from the root: `Q:\` → `Q:\Users` → `Q:\Users\UserMgr0` → ... The intermediate segment `UserMgr0` (Xbox AppContainer user manager) is not accessible from the AppContainer → `ACCESS_DENIED` → exception → crash.

The `\\?\` long-path prefix does not help: it bypasses MAX_PATH but not the access check.

**Fix A (default, ≤2 GB)**: before MSIX packaging, merge `model.onnx.data` into `model.onnx` to produce a self-contained model file. With no external data file, `ValidateExternalDataPath` is never called and `weakly_canonical` is never invoked. Tool: `scripts/merge_onnx_external_data.py`. CI runs this automatically as part of `build-uwp`. **Caps at the 2 GB protobuf single-file ceiling.**

**Fix B (patched ORT DLL, shipping since 1.1.8.0 — console-validated 2026-07-15)**: for models whose `.onnx.data` is >2 GB (un-mergeable), a patched `onnxruntime.dll` applies two AppContainer fixes (`patches/onnxruntime-extdata-appcontainer.patch`). The shipping `build-uwp.yml` installs the pinned DLL from the [`vendor-dlls-v1`](https://github.com/gianlucamazza/xllama/releases/tag/vendor-dlls-v1) release (hash in `vendor/onnxruntime-patched/SHA256SUMS`); full source rebuild is `build-uwp-ort-patched.yml` (1–3 h). See `docs/fp16-extdata-runbook.md`.

1. **`weakly_canonical` guard** (against the **1.24.4** call sites): use the `std::error_code` overload + lexical fallback so path walk failures do not throw. **Upstream note:** Microsoft landed a related fix on ORT `main` in [#28509](https://github.com/microsoft/onnxruntime/pull/28509) (`GetWeaklyCanonicalPath` + NT-volume AppContainer fallback, 2026-05-15) — **not** present in NuGet DirectML **1.24.4**, which is why the vendor pin still carries our guard.
2. **`ReadFileIntoBuffer` chunk 1 GB → 16 MB** (`env.cc`): a single ~0.5–1 GB `ReadFile` of a large un-quantized tensor (e.g. 128k-vocab embedding) locks too many pages in the AppContainer → intermittent `errcode 1450` (ERROR_NO_SYSTEM_RESOURCES). **Still open on ORT `main`** (chunk remains 1 GB as of 2026-07); tracked for upstream contribution.

**Validated**: a 1.86 GB-extdata int4 model that crashed pre-patch now loads and runs, 6/6 restarts, 0 crashes. Note: Fix B unblocks _loading_, but fp16 >2 GB models still hit the GPU **inference** budget wall (§7).

**Diagnosis**: Win32 probes on the model path — `GetFileAttributesW` and `CreateFile2` with `GENERIC_READ` succeed on `model_dir\model.onnx`, but the crash occurs inside the ORT segment-walking loop. Confirmed by matching the call site to `onnxruntime/core/framework/tensorprotoutils.cc` L337/338/346.

## 9. Disk Budget (Dev Mode Partition)

> **Superseded (2026-07-08)**: the Dev Mode storage allocation was raised to
> **90 GB** via Dev Home → Manage Dev Storage. The figures below describe the
> default allocation and remain valid as the baseline for a fresh Dev Mode
> activation; with the enlarged allocation, disk is no longer the binding
> constraint for model sizing (GPU budget and RAM are). One caveat to verify:
> community reports a ~2 GB per-file limit in Dev Mode, relevant for merged
> `model.onnx` files above that size.

**Observed**: the Xbox Series S Dev Mode partition (`Q:\`) provides approximately **2.2–2.5 GB of free space** after a clean Dev Mode activation, before any sideloaded package.

**Effect on model selection**: peak disk usage during deploy is approximately **2× the MSIX size**, because the package is uploaded to a staging area before installation completes. A 400 MB MSIX requires ~800 MB free during deploy, and ~400 MB residual after.

**Working budgets** (empirical, Series S Dev Mode, with xllama installed):

| Budget                      | Conservative | Borderline | Over budget |
| --------------------------- | ------------ | ---------- | ----------- |
| MSIX size                   | < 600 MB     | 600–800 MB | > 800 MB    |
| Model on-disk (merged ONNX) | < 400 MB     | 400–600 MB | > 600 MB    |

**Failure mode**: deploy fails with `0x80070070` (ERROR_DISK_FULL) if free space drops below the staging requirement. Alternatively, the first-launch copy from `InstalledPath` to `LocalState` fails silently, and `resolve_model_path` falls through to the default path — resulting in a model-not-found error at runtime rather than a visible install error.

**Source**: repeated `deploy.sh` runs against `https://<XBOX_IP>:11443/api/devices/file/usage`. Series X Dev Mode has the same partition layout but available free space was not measured by this project.

For models not distributed through the catalogue, see the USB/Device Portal
provisioning paths in `docs/model-selection.md`, `docs/device-portal.md` and
`src/bridge/path_utils.cpp`.

See also `docs/model-selection.md` for a consolidated checklist.

## 10. Win32 APIs Available in WINAPI_PARTITION_APP (Xbox Dev Mode)

Reference for future work — APIs that do **not** require a desktop-only guard on Xbox UWP:

| API                                                           | Header   | Notes                          |
| ------------------------------------------------------------- | -------- | ------------------------------ |
| `CreateThread`, `WaitForSingleObject`, `CloseHandle`, `Sleep` | kernel32 | PARTITION_APP since 10.0.14393 |
| `FreeLibrary`                                                 | kernel32 | PARTITION_APP                  |
| `GlobalMemoryStatusEx`                                        | kernel32 | PARTITION_APP since 10.0.15063 |
| `GetModuleFileNameW`                                          | kernel32 | PARTITION_APP                  |
| `SetThreadPriority`, `GetCurrentThread`                       | kernel32 | PARTITION_APP                  |
| SRWLOCK, CONDITION_VARIABLE, Interlocked\*                    | kernel32 | PARTITION_APP                  |
| `QueryPerformanceCounter/Frequency`                           | kernel32 | PARTITION_APP                  |
| `_aligned_malloc` / `_aligned_free`                           | CRT      | Available                      |

APIs that are **desktop-only** and require `#if WINAPI_PARTITION_DESKTOP` guards: `RegOpenKeyEx`, `RegQueryValueExA`, `SetThreadAffinityMask`, `SetThreadInformation(ThreadPowerThrottling)`, `<winevt.h>` includes.

### §10b — The Desktop **contract** is absent, whatever the manifest says (2026-07-30)

`AppxManifest.xml` declares two target device families, `Windows.Xbox` and
`Windows.Desktop`. That declaration says which families may install the package;
it does **not** mean the Desktop API contracts exist at runtime on any of them.

Measured on the console (Dev Mode, package 1.5.2.x) by the `[caprec]` probe in
`App.cpp`:

```
[caprec] AppRecordingManager present=0 GraphicsCaptureSession present=1
```

- **`Windows.Media.AppRecording`** — Desktop Extension SDK, 10.0.16299. **Not on
  the console.** This is the API by which an app records its own audio/video
  through the SoC video encoder, so the cheapest possible route to a real demo
  recording is closed. An SDKReference to the Desktop extension was added to
  compile against it and removed again the same day: the projection compiles,
  the type is not there to activate.
- **`Windows.Graphics.Capture`** — Universal contract. **Present.** Self-capture
  is therefore not closed, only more expensive: that path encodes in-process,
  which competes for the ~6 usable cores (§6) with whatever inference a demo is
  meant to be showing. Unmeasured; do not assume it is usable.

The general rule, and the reason the probe logs rather than assumes:
**`ApiInformation::IsTypePresent` is the only statement about a contract that is
worth anything.** A projection existing at compile time, an SDKReference in the
vcxproj, and a `TargetDeviceFamily` in the manifest are all compile-time or
install-time facts, and none of the three implies the type can be activated on
the device in front of you.

### §10c — A second `ContentDialog` kills the process, silently (2026-07-30)

XAML allows exactly **one** `ContentDialog` open per `XamlRoot`. Showing a second
throws. That alone would be ordinary — the trap is where the exception lands.

Every dialog in this app is opened from a `winrt::fire_and_forget` coroutine, and
`fire_and_forget`'s promise implements `unhandled_exception()` as
**`std::terminate()`**. So the failure is not an exception you can catch at the
call site and not an error on screen: the process disappears. No `xllama.log`
line, no crash dialog, and — because the autopilot marker is written by the app
itself — **no `autopilot-done.txt`**. From the host side that is indistinguishable
from a hung run, so an automated gate reports a timeout for what is actually an
instant death, and the operator goes looking for a deadlock that is not there.

It stayed unreachable for a long time by accident, not by design: the first-run
disclaimer was the only programmatic dialog, and every gate script seeds it away.
The `show_pane` autopilot op made it reachable, which is how it was found.

The guard is an `std::atomic<bool>` owned by `MainPageController` (`m_pane_open`),
set from the dialog's `Opened` / `Closed` events by `ApTrackDialog`, and checked
before opening a pane: a collision now fails with a readable `error:` in the
marker instead of taking the process with it. Verified by running `show_pane`
under the unaccepted disclaimer — the only collision actually reachable today.

**The rule this generalises to:** in a `fire_and_forget` that touches XAML, an
uncaught throw is a process kill, not an error path. Anything that can throw
there needs either a guard before it or a `try`/`catch` inside it. Prefer the
guard, because a caught exception still leaves the UI in whatever state the
half-finished coroutine left it.

## 11. GPU Truth — EP Attribution Without PIX

PIX for Xbox is GDK tooling gated behind the managed partner program; it is **not available for Dev Mode UWP**. GPU-vs-CPU execution truth is instead established by converging three surfaces (all verified against primary sources, ORT GenAI 0.13.2 / ORT 1.24.4):

1. **ORT profiling JSON (primary, definitive)**. `genai_config.json` → `session_options` accepts `enable_profiling` (a string: the profile file _path prefix_, producing `<prefix>_<timestamp>.json`) and `log_severity_level` (0 = VERBOSE). Every `<node>_kernel_time` event in the trace carries `args.provider` — literally `"DmlExecutionProvider"` or `"CPUExecutionProvider"`. Heavy kernels (MatMul/Attention) tagged CPU = silent fallback. This works regardless of ORT build flavor and log routing. Tooling: `scripts/profile-dml-run.sh` (config swap + run + fetch) and `scripts/analyze_ort_profile.py` (per-provider summary + greppable `VERDICT:` line).
   - _Profile location ladder_: the relative prefix resolves against the process CWD, which in AppContainer may be the read-only install root (ORT's profiler ofstream then fails silently). Step 1: the fetch script checks LocalState root **and** `models\<name>\`. Step 2: `--absolute-prefix` renders `genai_config-dml-profile.tpl.json` with an absolute LocalState path. Step 3 (definitive): `set_cwd_to_local_folder()` pins CWD to LocalState at bench startup (v0.3.2+ MSIX).
2. **Device Portal telemetry (corroborating)**. `GET /api/resourcemanager/systemperf` exists on the Xbox device family and reports `GPUData.AvailableAdapters[]` with `EnginesUtilization[]` (0–1 per engine) and `DedicatedMemoryUsed`. System-wide, ~1 Hz — run a control pass with the CPU config to calibrate background noise. Tooling: `scripts/xbox-gpu-sample.sh`, integrated as `--gpu-sample` in the bench/profile scripts.
3. **In-app GPU memory (corroborating)**. `IDXGIAdapter3::QueryVideoMemoryInfo(LOCAL)` is callable from the AppContainer and is _per-process_: `CurrentUsage` climbing toward the model size after `OgaCreateModel` means the weights are resident on the GPU; `Budget` is the OS-granted ceiling (measured **3801 MB** on Series S with the package designated Game — trust this value over any hard-coded constant). Implemented as `gpu_mem_info()` in `src/bridge/platform.cpp`, logged pre-load/post-load/post-decode and exported as `gpu_mem_mb,gpu_budget_mb` bench CSV columns.

**Node-placement log caveat**: at `log_severity_level: 0` ORT emits "Node placements" lines from `session_state.cc`, but (a) only in full (non-minimal) ORT builds, and (b) ORT-core session logs may not route through the `OgaSetLogCallback` sink into `xllama.log`. Absence of the lines is not evidence — the profiling JSON is the primary probe.

**Fallbacks if the above is inconclusive** (not implemented; documentation only):

- Op inventory of the model graph (python + `onnx`, count `node.op_type` incl. `com.microsoft.*` domains) cross-referenced against the profiler's per-op CPU list to identify which ops force fallback.
- Full-vs-minimal ORT build probe: `strings onnxruntime.dll | grep "Node placements"` on the NuGet-restored DLL (affects only the log probe, not profiling).
- D3D12 debug layer: not viable in Dev Mode UWP (`DML_CREATE_DEVICE_FLAG_DEBUG` needs `DirectML.Debug.dll` and ORT creates the DML device internally); DXGI HRESULTs already surface through the SEH translator.

## 12. Why int4 Decode Is Slow on DirectML (not a missing kernel)

Desk-check dated 2026-07-08. Corrects the earlier wording that the int4 decode
collapse (8.8 tok/s) was a "missing fused int4 DML kernel" — the op is present
and GPU-executed; the limit is that DirectML's kernel is **non-fused**.

**Ground truth (verified locally)**:

- The ORT GenAI model builder `-p int4 -e dml` emits **225
  `com.microsoft::MatMulNBits`** nodes (`bits=4, block_size=32`) — a fused-op
  graph, not `DequantizeLinear`+`MatMul`. (`onnx` op inventory of the built
  `model.onnx`.)
- The profiled on-console run
  (`bench/results/profiles/20260707T144203Z/ort_profile_*.json`) shows the entire
  model as a single **`DmlFusedNode_0_0` on `DmlExecutionProvider`** (1549 ms,
  96% of kernel time); the only CPU kernels are `Gather` + `Cast` (0.1 ms). No
  `MatMulNBits` runs on CPU → **not a silent CPU fallback**.

**Root cause (verified against ORT source)**:

- DirectML **does** register `MatMulNBits`
  (`onnxruntime/core/providers/dml/.../Operators/OperatorRegistration.cpp`), but
  `DmlOperatorMatMulNBits.cpp` implements it as **`DML_DEQUANTIZE` (int4→fp16 on
  the GPU) + a full `DML_GEMM`** — it materialises the fp16 weight tensor. There
  is **no fused low-bit GPU GEMM**. In memory-bound decode (M=1) this reads the
  4-bit weights, writes a full fp16 weight matrix to VRAM, and reads it back —
  strictly **more** bandwidth than plain fp16, which is exactly why int4-DML
  (8.8) is _slower_ than fp16-DML (46.8 pre-fix; 44.4 on the `-v2` graph).
- The GenAI builder defaults `int4_accuracy_level = 4` for the CPU EP but **`0`
  for non-CPU EPs**. CPU's level 4 activates MLAS's fused int8 low-bit GEMM
  (`SQNBitGemm`) — the reason CPU int4 hits 68 tok/s. DML gets level 0 (fp16
  compute) and its kernel has no int8 fast path to switch into.
- ORT 1.19/1.20 int4 improvements ("QDQ INT4", int4 embeddings) were added to the
  **CPU and CUDA EPs only**; no DirectML fused low-bit kernel has shipped through
  DirectML 1.15.x. Published DirectML-GenAI INT4 models (Phi-3, Llama-3.1) note
  compute runs at fp16/fp32 accuracy — consistent with level 0.

**Consequence for xllama**: int4-on-DML has **no path to beat fp16-on-DML** for
decode by any quantization config we control, and fp16-on-DML (44.4 on the
shipping `-v2` graph; 46.8 pre-fix) still loses to CPU int4 (68). So **CPU int4 stays the decode default**, and the GPU's real
win is prefill (§5, reading 1 — realized for text by the `-v2` RMSNorm-decomposed
asset, #91 postmortem) and larger-model bandwidth. A genuinely fused
low-bit GPU GEMM would be a DirectML-team feature, not an ORT-side PR; treat GPU
int4 decode as blocked upstream, not a local TODO.

**Config tests** (`int4_block_size=128`, `int4_accuracy_level=4` variants of
SmolLM2-360M) were **built but closed inconclusive 2026-07-09** — not console-
benched, and not a local lever. The §12 desk-check above stands (the kernel
structure predicts neither materially beats 8.8 tok/s); see `ROADMAP.md`
the historical Phase 3.5 / Phase 5 evidence in `CHANGELOG.md` and
`bench/results/`.

## 13. Training and adapters (RE inventory)

> Cross-link: full training pillar SSOT is
> [`training-architecture.md`](training-architecture.md). This section records
> **platform facts** only (what the Xbox UWP stack can and cannot do).

**NuGet pins are inference-only.** `uwp/packages.config` ships
`Microsoft.ML.OnnxRuntime.DirectML` 1.24.4 and
`Microsoft.ML.OnnxRuntimeGenAI.DirectML` 0.14.1 — not an ORT **Training**
package. Those pins provide no ORT train loop. The llamacpp/unified variants
instead link the separate ggml-opt partial-FT engine (Lane B, gates PASS); it
does not turn ORT GenAI into a training runtime.

**ORT GenAI adapters ≠ training.** `strings` on the pinned
`onnxruntime-genai.dll` exposes `OgaCreateAdapters` / `OgaLoadAdapter` /
`OgaSetActiveAdapter` (inference-time adapter load) and the diagnostic
`No adapter is available for DML`. GenAI does not expose a gradient/train API
on this pin. DML adapter load is blocked; CPU-only adapter load is unproven in
xllama.

**llama.cpp** provides `llama_adapter_lora_*` for **runtime LoRA load** (forward)
and a separate WIP `llama-finetune` example that cites multi-ten-GB class memory
for small FP32 models — not a practical Series S path.

**Memory.** GPU budget remains **3801 MB** (§5/§7). Host PEFT on SmolLM2-360M
(r=8) trains ~1.64 M params — adapter state is PEFT-sized (tens of MB), so the
binding constraint for ORT ODT is **missing software**, not only VRAM for
LoRA-class adapters. Full fine-tune of catalogue models remains out of budget;
the ggml-opt Lane B experiment limits training to a supported last-block subset.

**AppContainer.** A train engine must be **app-local, signed, linked at build
time** (§3); Lane B follows that rule. No Python PEFT inside the sandbox.
Writable artefacts only under LocalState (§2).

**Architecture gate.** `training_device_supported(Device)` is compile-time true
only when `XLLAMA_DEVICE_TRAIN` is present (llamacpp/unified builds). The
`DeviceGgmlPartialFt` capability is **available** (host + console marker gates
PASS 2026-07-20: peak working set 1195 MB, marker reproduced on the Series S);
ORT-only builds reject device jobs and still emit a failing
`training/result.done` marker instead of hanging the harness.
