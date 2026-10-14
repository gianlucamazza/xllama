# Community forks

What the public forks of xllama contain, what was taken, and why the rest was
not. Reviewed on 2026-10-14 against `main` (llama.cpp `d812350`). Each verdict
names its evidence and what would reopen it. Two independent agent reviews
(Codex, Grok) challenged the first draft; their corrections are folded in below.
All forks keep the MIT licence.

| Fork                                             | Taken                                                     | Not taken                                            |
| ------------------------------------------------ | --------------------------------------------------------- | ---------------------------------------------------- |
| [SamirCharleston](#samircharleston-vision-spike) | llama.cpp warnings on every load (credited in the commit) | batch cap, repack opt-out, mmap revival, vision code |
| [bjoern2088](#bjoern2088-xbox-one-x)             | —                                                         | Xbox One X port                                      |
| [http-hadi](#http-hadi-d3d12f)                   | evidence for [#273], [#274], [#275] and the [#313] design | the vendored D3D12F backend                          |
| miguelenes                                       | already in `main` (#261, `4100bc2`)                       | —                                                    |
| nyzxor, hjotha, mr-wfj                           | —                                                         | no commits ahead                                     |

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
- Qwen2.5-Coder 1.5B and 3B already pass on the console
  ([model-matrix.md](model-matrix.md)).
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

No result on an Xbox One was published. The product is Series S|X
([positioning.md](positioning.md)). Taken into the shipping build, these flags
would cost Series S|X its AVX2 CPU kernels and the SM 6.4 packed dot used by
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

[#172]: https://github.com/gianlucamazza/xllama/issues/172
[#273]: https://github.com/gianlucamazza/xllama/issues/273
[#274]: https://github.com/gianlucamazza/xllama/issues/274
[#275]: https://github.com/gianlucamazza/xllama/issues/275
[#313]: https://github.com/gianlucamazza/xllama/issues/313
[#345]: https://github.com/gianlucamazza/xllama/pull/345
