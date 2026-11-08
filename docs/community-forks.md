# Community forks

What the public forks of xllama contain, what was taken, and why the rest was
not. Reviewed on 2026-10-14 against `main` (llama.cpp `d812350`). Each verdict
names its evidence and what would reopen it. Two independent agent reviews
(Codex, Grok) challenged the first draft; their corrections are folded in below.
All forks keep the MIT licence.

| Fork                                               | Taken                                                     | Not taken                                            |
| -------------------------------------------------- | --------------------------------------------------------- | ---------------------------------------------------- |
| [SamirCharleston](#samircharleston-vision-spike)   | llama.cpp warnings on every load (credited in the commit) | batch cap, repack opt-out, mmap revival, vision code |
| [bjoern2088](#bjoern2088-xbox-one-x)               | —                                                         | Xbox One X port                                      |
| [http-hadi](#http-hadi-d3d12f)                     | evidence for [#273], [#274], [#275] and the [#313] design | the vendored D3D12F backend                          |
| miguelenes                                         | already in `main` (#261, `4100bc2`)                       | —                                                    |
| nyzxor, hjotha, mr-wfj                             | —                                                         | no commits ahead                                     |
| [maxmorrison74](#maxmorrison74-xllama-on-steroids) | optional debug dashboard idea (rebuilt in-tree)           | rebrand, background, soundtrack, footer credit       |

## SamirCharleston: vision-spike

Branch `vision-spike`, 18 commits on a July base, by Samir Charleston.

**Taken: llama.cpp warnings on CPU-only loads** ([#345]). `main` installed a llama log
callback only on the d3d12 path, after GPU init, so a CPU load's warnings went
to llama's default stderr logger, which UWP never shows. `install_llama_log`
(`src/bridge/llama_gpu.h`) now sets one non-throwing callback once, before the
first load on every path (session, `run_inference`, device train). It is set
exactly once because `llama_log_set` writes process-global llama and ggml state
without a lock.

**Not taken: `n_batch`/`n_ubatch` capped at `min(n_ctx, 512)` when unset.**

- The pin already defaults `n_ubatch` to 512, and chat's compute reserve is
  sized by it, so the cap buys chat nothing.
- It would cost two things:
  - Embeddings set `n_ubatch = n_batch` and truncate input to `n_batch`
    (`src/bridge/session.cpp`), so `embed-bge-m3` would drop from 2048 to 512
    input tokens.
  - llama.cpp clamps an explicit `n_ubatch` to `n_batch`, so the cap would
    silently disable the `--ubatch` and `bench_ubatch.txt` knob above 512
    ([#172]).
- Qwen2.5-Coder 1.5B already passes on the console
  ([model-matrix.md](model-matrix.md)). The 3B console load is a historical
  receipt (benchmark; model under a non-commercial research licence); that
  model is not in the catalogue.
- Reopen if a console load fails at context creation on `main`.

**Not taken: `use_extra_bufts = false`.** It turns off llama.cpp's CPU repack
buffers, which gave the measured GGUF prefill and load gains
([benchmarks.md](benchmarks.md), PR #155). The fork's premise of a ~1.3 GB
second buffer is wrong: repacked weights replace the CPU weight buffer. Without
mmap there is one real transient cost. The repack buffer type is not host
memory, so the loader stages each repacked tensor in its own read buffer before
repacking it (`llama-model-loader.cpp`, one tensor at a time). That is the
largest tensor, not a model copy. Reopen with a measured load peak that this
staging pushes past the budget; the fix would then be bounded staging, not
disabling repack.

**Not taken: AppContainer mmap revival.**

- It was measured with no load or peak benefit, and reverted
  ([benchmarks.md](benchmarks.md), [uwp-constraints.md](uwp-constraints.md) §1).
- Since repack (#155) the case is weaker still: repacked weights cannot live in
  the mapping, so the mapping and the repack buffer would both be held.
- Reopen with a specific new hypothesis and a measurement on today's load path.

**Not taken as code: vision.**

- The branch adds a MobileNetV2 DirectML probe, a SmolVLM-256M caption probe
  through llama.cpp mtmd, `/v1/vision/caption`, and image context in chat.
- [ROADMAP.md](../ROADMAP.md) closed the vision workstream (H16.7) at a desk
  review. The reason was the surface budget: five new C++ surfaces against at
  most one. It was not a measured capability failure, and this branch is that
  ledger again.
- SmolVLM-256M is a desk data point for [#275]. Reopen through a single new
  input path with console numbers, not a larger demo.

## bjoern2088: Xbox One X

19 commits, by bjoern2088:

- `/arch:SSE2` in place of AVX2;
- the d3d12 shaders rebuilt as `cs_5_1` DXBC through fxc;
- `dot4add_i8packed` replaced by a scalar multiply-add.

No result on an Xbox One was published. The product targets Xbox Series S
consoles (not yet tested on Series X; see [positioning.md](positioning.md)).
Taken into the shipping build, these flags
would cost Series S its AVX2 CPU kernels and the SM 6.4 packed dot used by
the measured GPU decode; a separate SKU would not, but it would be a second
build to validate and maintain for a console the project does not target.
Reopen if Xbox One becomes a goal, as a separate build lane.

## http-hadi: D3D12F

Branch `feat/d3d12-full` and its release branches vendor
`koza-coder/llama.cpp-directx12` v0.2.7:

- MIT, about 15k lines;
- nearly every ggml op, including `MUL_MAT_ID`, SSM and flash attention;
- a selector between it and `main`'s backend;
- shaders baked in CI on WARP, the software rasterizer.

The `ci-logs-*` branches are that CI's output: correctness and op coverage on
WARP, including a 1-bit 27B model, MoE and hybrid architectures. There is no
tok/s from a console. One model's perplexity differs from the CPU by 1.19%.
Upstream has already moved one release on.

**Not taken: the backend.** It would mean 15k lines of vendored code with only
emulator evidence, a second GPU path beside the measured one, and shader
baking that relies on WARP posing as an AMD GPU.

**Used as design input for [#313].** Its tiled matmul dequantizes each weight
once into groupshared memory, which is the fix #313 needs: today's prefill
re-reads weights per column. It computes in float after dequantizing, though,
while `main` ships int8 dot products held to rel ≤ 1e-5 against the CPU's q8
path (Gate A in [gguf-gpu-decode.md](gguf-gpu-decode.md)). Both reviews
recommended a multi-column int8 kernel in `main`'s own shaders, with these
conditions:

- same Gate A;
- one column keeps the current GEMV, so decode does not move;
- a prefill margin declared per model before measuring;
- measured in the UI process.

Its MoE and SSM kernels are evidence for [#273] and [#275], not a plan.

## maxmorrison74: xllama-on-steroids

Fork [xllama-on-steroids](https://github.com/maxmorrison74/xllama-on-steroids)
and closed PR [#351](https://github.com/gianlucamazza/xllama/pull/351), by
Max Morrison. The useful idea was an on-console view of the resident model,
LAN activity, and inference statistics. That view is rebuilt in-tree as an
**opt-in debug dashboard** (default off; chat UI unchanged when off; real
readings only). Credit: `CHANGELOG.md`, `CONTRIBUTORS.md`.

**Not taken:** the "Xllama on Steroids" rebrand (DisplayName, AppxManifest
identity, version, workflow artifact names, README), the packaged background
image, the soundtrack, or the footer signature. Process-wide compute admission
from the same fork is a separate PR and is not required here.

[#172]: https://github.com/gianlucamazza/xllama/issues/172
[#273]: https://github.com/gianlucamazza/xllama/issues/273
[#274]: https://github.com/gianlucamazza/xllama/issues/274
[#275]: https://github.com/gianlucamazza/xllama/issues/275
[#313]: https://github.com/gianlucamazza/xllama/issues/313
[#345]: https://github.com/gianlucamazza/xllama/pull/345

## 2026-11-07 pinned re-review: bjoern2088

Compared upstream `ea40b28deb5c9601863dd5b99709360b1bdee5bf` with
`bjoern2088/main` at `43cbd38b70301fe6308db35af9be657f07624725`.
Merge base: `ebc6a6eb18269576a78d9e3c78d678c509673de0`.
`git rev-list --left-right --count` reports 57 upstream-only and 19 fork-only
commits. These are ancestry counts, not a count of applicable improvements.
No branch was imported or merged. No Xbox, D3D12, compiler or performance
execution was performed during this review.

### Exclusive commits (newest first)

| Commit    | Scope                                                                    |
| --------- | ------------------------------------------------------------------------ |
| `43cbd38` | SM5.1-compatible reduction; 1D groupshared array, manual reduction steps |
| `5ef051b` | fxc trivial-shader CI diagnostic                                         |
| `780c53e` | fxc-compatible load helpers and sign extension                           |
| `935e50b` | Find fxc.exe in SDK                                                      |
| `47e7064` | Use fxc for DXBC targets                                                 |
| `7ab471d` | CI cs_5_1 DXBC rebuild                                                   |
| `6cf2224` | Shader script cs_5_1 DXBC rebuild                                        |
| `5cce1c0` | Earlier CI cs_6_0 rebuild                                                |
| `8e451ef` | Earlier script cs_6_0 rebuild                                            |
| `279b0f2` | q6_k t128 generated header                                               |
| `28271c6` | q6_k t64 generated header                                                |
| `ffddd3f` | q4_k t128 generated header                                               |
| `c69a347` | q4_k t64 generated header                                                |
| `9c50a4e` | q4_0 t128 generated header                                               |
| `538a1d8` | q4_0 t64 generated header                                                |
| `3a93777` | get_rows_q6_k generated header                                           |
| `184b515` | Common shader scalar dot and compatibility changes                       |
| `10b1b5f` | xllama UWP AVX2 to SSE2                                                  |
| `4512fab` | ggml UWP AVX2 to SSE2                                                    |

### Technical verdict

The fork is an Xbox One X compatibility experiment. It adds no multi-column
weight reuse and no new quantization format. The existing upstream Q4_0/Q4_K/Q6_K
kernels already use CPU-quantized q8 activations. `SV_GroupID.y` still selects one
activation column; #313 is open and remains an independent optimization.

The diff from the merge base touches 12 files: two UWP vcxproj files, the package
build action, shader compiler script, common HLSL include, and seven generated
headers. It replaces both UWP `/arch:AVX2` options with `/arch:SSE2`, replaces
`dot4add_i8packed` by four signed scalar products, flattens the groupshared
reduction array, manually enumerates the reduction steps, and double-loads
aligned words in `ld32`/`xld32` even when the requested address is aligned.
Those changes target compatibility, with potential Series S regressions that
have not been measured here. They must not replace the shipping Series S lane.

The seven checked-in headers all say `AUTO-GENERATED by compile_sm60.sh`,
`cs_6_0`, and `bytes DXIL`. No `compile_sm60.sh` is tracked in that fork.
Parsing the DXBC container chunk table of every header confirms a `DXIL` chunk
in all seven blobs. Meanwhile the fork's current
`scripts/compile-gpugemv-shader.sh` and CI action generate cs_5_1 through fxc.
Therefore checked-in blobs and current regeneration instructions do not describe
one reproducible shader build. The leading `DXBC` container magic alone does
not distinguish legacy shader bytecode from a DXIL container. An import would
require an explicit shader profile/backend identity and fresh artifact receipts.

The present script also labels fxc output as `DXIL` and retains `_dxil` names.
The CI discovers the newest installed fxc instead of pinning/recording a known
compiler identity. A successful fxc compilation is a build result; it is not
proof of console pipeline creation, numerical correctness, or speed.

No new benchmark/result files differ from the merge base. Existing Series S
receipts cannot validate a different ISA, shader model, compiler or Xbox One X.
The prior decision in this document remains: no automatic import; reopen Xbox
One only as a separately scoped build lane if it becomes a product goal.

### Isolated experiment gates

For #313, extend upstream's existing shaders and selftest instead of importing
this compatibility branch. Preserve current one-column GEMV for decode; add a
selectable tiled multi-column int8 path with an explicit fallback. The owning
contract remains `docs/gguf-gpu-decode.md`; this review does not redefine its
historical gate outcomes.

Before any console comparison, require host emulation vs ggml q8 dot for all
three types, 64/128 threads, columns 1/7/512, boundary rows, strides and invalid
shape rejection. Run the existing console selftest with the declared Gate A
(rel <= 1e-5; GET_ROWS bitwise); rerun H9 on the gate models and shipping UI
process. Predeclare per-model prefill gain, acceptable decode noninferiority,
TTFT and memory budgets before collecting any optimized result. Include short
prompts so tiling overhead cannot disappear behind long-prompt averages.

Record source include hashes, generated blob hashes, compiler version/binary
hash, complete compile flags, entry point, profile, macro definitions, upstream
and experiment SHAs, model hashes, package identity/hash, device family, timing
scope, warmup, all raw repeats and failed/excluded runs. Compare randomized or
alternating matched baseline/experiment arms with confidence intervals. Keep
Series S, Series X and host analyses separate. Publish a negative or inconclusive
result as such. Hardware absence blocks promotion and the stable release gate;
it does not justify extrapolating host results.

## Live public fork inventory, 2026-11-07

The GitHub REST fork collection returned 10 public forks (all represented below). Branches were enumerated and comparisons pinned against upstream `ea40b28deb5c9601863dd5b99709360b1bdee5bf`. Ahead/behind are ancestry counts; squash-merged equivalents can remain ahead. Comparisons inspect commit subjects and changed-file scope; they do not imply full code or hardware validation.

| Fork / branch                                                                           | SHA                                        | Ahead / behind |
| --------------------------------------------------------------------------------------- | ------------------------------------------ | -------------- |
| `weiting-tw/xllama` / `ccr-dc7c3963-oookvu`                                             | `c48a6aca6c733db6512afcd5f5018c5e13233509` | 1 / 9          |
| `weiting-tw/xllama` / `main`                                                            | `da1aec40b84207b494f954c01ca5449783be90b8` | 0 / 9          |
| `maxmorrison74/xllama-on-steroids` / `codex/process-compute-admission-macos-build`      | `02ae793efed85b1de7bc0e880aebb966355eaeb2` | 1 / 34         |
| `maxmorrison74/xllama-on-steroids` / `codex/xllama-on-steroids-branding`                | `b8aad358b1b3f6a790a921a2c3aba38d7b664d57` | 17 / 34        |
| `maxmorrison74/xllama-on-steroids` / `dependabot/submodules/version-updates-dd6c1a1710` | `ab41616326a785ef41445cc7c0a47b920989992c` | 1 / 34         |
| `maxmorrison74/xllama-on-steroids` / `feat/floppylm-training`                           | `ba0ebcedb385aca2e9646cec65b3a11fc7b6db89` | 8 / 71         |
| `maxmorrison74/xllama-on-steroids` / `feat/moe-xbox-smoke-probe`                        | `5cba82fc1c69d39b6f4331fc43d8c0460b4f7fd9` | 22 / 34        |
| `maxmorrison74/xllama-on-steroids` / `main`                                             | `574b492b4a158442bfe13e8100ab87286036af8a` | 2 / 34         |
| `t91012696-glitch/xllama` / `feat/floppylm-training`                                    | `ba0ebcedb385aca2e9646cec65b3a11fc7b6db89` | 8 / 71         |
| `t91012696-glitch/xllama` / `main`                                                      | `d6593de281b5b0b0bfbd249ed153512772dc162c` | 1 / 34         |
| `duranlisandro/xllama_matrix` / `build/head-20261016`                                   | `5271350761f75da6953a1644ea12f72e2bdd334f` | 0 / 34         |
| `duranlisandro/xllama_matrix` / `fix/347-run-scoped-logs`                               | `8386256442fa7bdf299f608fc42253693ee3fc96` | 2 / 29         |
| `duranlisandro/xllama_matrix` / `main`                                                  | `5271350761f75da6953a1644ea12f72e2bdd334f` | 0 / 34         |
| `duranlisandro/xllama_matrix` / `pr/sse-chat-streaming`                                 | `6add2a55b18e0a28724c6bc88047a9350ebd177d` | 12 / 34        |
| `http-hadi/xllama` / `feat/d3d12-full`                                                  | `750c20bc3df2637c7a11443c4a4cc038d7125373` | 8 / 57         |
| `http-hadi/xllama` / `feat/floppylm-training`                                           | `ba0ebcedb385aca2e9646cec65b3a11fc7b6db89` | 8 / 71         |
| `http-hadi/xllama` / `gpu-for-all`                                                      | `3aecc96d9a1dd2af58de15e82cdfd30a7c49ae12` | 3 / 57         |
| `http-hadi/xllama` / `gpu-only-fix`                                                     | `0bded61bb152c252718e2e6d4f056ab716caeff8` | 18 / 57        |
| `http-hadi/xllama` / `main`                                                             | `ebc6a6eb18269576a78d9e3c78d678c509673de0` | 0 / 57         |
| `http-hadi/xllama` / `package/d3d12f`                                                   | `0bded61bb152c252718e2e6d4f056ab716caeff8` | 18 / 57        |
| `http-hadi/xllama` / `probe/no16`                                                       | `750c20bc3df2637c7a11443c4a4cc038d7125373` | 8 / 57         |
| `http-hadi/xllama` / `probe/warp`                                                       | `33f4ce87651b0d060b2e75e37065a79e87380a2f` | 2 / 57         |
| `http-hadi/xllama` / `release/d3d12f`                                                   | `750c20bc3df2637c7a11443c4a4cc038d7125373` | 8 / 57         |
| `bjoern2088/xllama` / `main`                                                            | `43cbd38b70301fe6308db35af9be657f07624725` | 19 / 57        |
| `hjotha/xllama-1` / `feat/floppylm-training`                                            | `ba0ebcedb385aca2e9646cec65b3a11fc7b6db89` | 8 / 71         |
| `hjotha/xllama-1` / `main`                                                              | `83cac358f0cf0a314220a28cb37eaac1509c245c` | 0 / 65         |
| `SamirCharleston/xllama` / `main`                                                       | `4bad1d58d3c38dbab3d484e1dab5a611cc20c58e` | 0 / 99         |
| `SamirCharleston/xllama` / `vision-spike`                                               | `c104faed1ff83d8efbf15908ad644d32d764f67c` | 18 / 99        |
| `mr-wfj/xllama` / `main`                                                                | `4bad1d58d3c38dbab3d484e1dab5a611cc20c58e` | 0 / 99         |
| `nyzxor/xllama` / `main`                                                                | `1b3ecbc4f1b1cc71aa9063bd2b8dd7fa9356ba7d` | 0 / 659        |

- **weiting-tw/xllama:** Laya host feasibility spike; compare against upstream decision-model spike before any new issue/import. Host-only feasibility does not establish console support.
- **maxmorrison74/xllama-on-steroids:** Branding/UI changes remain out of scope; compute-admission idea already rebuilt upstream. MoE/layer-streaming prototype and claimed Series X resource receipts need separate validation; no inference acceptance inferred.
- **t91012696-glitch/xllama:** Exclusive main change hard-codes build_revision to 2000 instead of github.run_number; reject for upstream because it breaks per-build unique version stamping. Training branch matches upstream feature branch pin.
- **duranlisandro/xllama_matrix:** Run-scoped logs and SSE work were already handled upstream; ancestry-exclusivity is not missing functionality after squash merges. Compare semantic deltas before opening duplicates.
- **http-hadi/xllama:** D3D12F remains separate vendored backend with WARP/CI evidence; no console throughput established. Use tiled kernel ideas only after own q8 Gate A experiment.
- **bjoern2088/xllama:** Compatibility-only Xbox One X experiment; not imported. Detailed pinned review above.
- **hjotha/xllama-1:** Main has no exclusive commits; training branch matches upstream feature branch pin.
- **SamirCharleston/xllama:** Existing vision-spike decision unchanged; load warnings already rebuilt upstream. No automatic import of context cap, repack or mmap changes.
- **mr-wfj/xllama:** No exclusive commits; no action.
- **nyzxor/xllama:** No exclusive commits; no action.

http-hadi also has 36 `ci-logs-*` branches. Their tips were enumerated in the branch listing; these are diagnostic output lanes, not separate product candidates. No shader or console claim is inferred from their names.

The maxmorrison74 `feat/moe-xbox-smoke-probe` branch (`5cba82fc1c69d39b6f4331fc43d8c0460b4f7fd9`) adds `bench/results/20261102T103941Z-series-x-streaming/`. Read-only inspection of `run.json` and `device-os-info.json` finds a declared Xbox Series X resource-probe run, package `GianlucaMazza.xllama_1.6.1.12_x64__pj67f1fcj4n14`, OS-info Platform `Xbox Series X`, and an explicit caveat that raw CSVs were hard-coded `xbox-series-s`. These are third-party fork receipts for RAM ceiling and disk bandwidth; they are not a test of upstream main, inference correctness, model load, console/API gates, or release candidate. Keep the raw labels and caveat intact. Request exact source/package hashes and rerun the upstream acceptance protocol before satisfying #374 or changing upstream hardware wording.
