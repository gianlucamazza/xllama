# MoE streaming for every candidate model: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the optional SSD expert-streaming path model-generic, so every model in the [candidate list](../../research/moe-streaming-candidates.md) can be admitted, verified and measured through one mechanism instead of one hard-coded 24B pin.

**Architecture:** One WinRT-free header owns the decision "which architectures stream which tensors" and "which checkpoints are admitted, with which identity and template". The native provider and the loader ask that table instead of testing `arch == "lfm2moe"`. Checkpoint identity comes from the bundled catalogue entry (a new optional `moe_streaming` object), never from the mutable LocalState override. Each architecture enters through the same ladder: tiny random-weight fixture parity on the host, then console CPU lifecycle, then benchmarks.

**Tech Stack:** C++17, doctest, pinned llama.cpp `d812350` (`build_moe_ffn` / `MUL_MAT_ID`), Python 3 + gguf-py for fixtures, C++/WinRT for the console activation.

**Spec:** [moe-streaming-candidates.md](../../research/moe-streaming-candidates.md) (models, filters, order) and the staged contribution plan [2026-11-16-ssd-moe-upstream.md](2026-11-16-ssd-moe-upstream.md) (PR 1–4 boundaries). This plan slots into PR 2 (native) and PR 3 (activation) of that sequence and does not replace it.

## Global Constraints

- Default resident path, `SessionHub` ownership and compute admission unchanged; streaming stays opt-in and experimental. No fallback to a resident load after a streamed refusal.
- Only the **bundled** catalogue can admit a checkpoint for streaming. A LocalState `manifest.json` override may add a download entry, but its `moe_streaming` object is ignored.
- Identity per checkpoint: exact file name, exact byte count, SHA-256, `general.architecture`, SHA-256 of the GGUF `tokenizer.chat_template`. Any mismatch refuses before allocation.
- Routed tensors are only the per-expert matrices: `ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps`, `ffn_gate_up_exps`. Shared experts (`*_shexp`), routers (`ffn_gate_inp`), expert biases and scales (`*_exps.bias`, `*_exps.scale`), norms and `exp_probs_b` stay resident.
- Native changes stay in patch files against the pin; nothing is committed inside `llama.cpp/`.
- GPU expert execution covers Q4_K and Q6_K only; other expert types are CPU-only and must be reported as such, not silently rejected or silently run on the GPU.
- Platform limits are in [uwp-constraints.md](../../uwp-constraints.md) (§1 no mmap, §9 disk); do not restate them in code comments.

## Review Focus

1. A GGUF whose `general.architecture` is supported but which uses a routed role the table does not list (for example a future merged tensor): load must refuse, not stream a partial set and keep the rest resident.
2. Same file name and size, different SHA-256 (a reconverted upload, as happened with LFM2.5-1.2B-Thinking): refuse with the expected and actual digests in the log.
3. A LocalState override that copies a bundled id and adds `moe_streaming`: the override's object is ignored; only the bundled pin admits.
4. Gemma 4 layers that hold both a dense FFN and a MoE block: only the MoE matrices are routed; the dense `ffn_gate/up/down` stay resident.
5. gpt-oss expert biases: resident and applied after the streamed product; a fixture must show logits equal to the resident run.

---

## File structure

| File                             | Responsibility                                                              |
| -------------------------------- | --------------------------------------------------------------------------- |
| `include/xllama/moe_streaming.h` | Architecture role table, checkpoint pin type, admission checks. WinRT-free. |
| `src/bridge/moe_streaming.cpp`   | Table data and check implementations.                                       |
| `tests/test_moe_streaming.cpp`   | Host tests for the table and the checks.                                    |
| `uwp/models/manifest.json`       | `moe_streaming` objects on admitted entries.                                |
| `uwp/model-downloader.cpp`       | Parse `moe_streaming`; drop it from override entries.                       |
| `scripts/make-moe-fixture.py`    | Tiny random-weight MoE GGUF per architecture (gguf-py).                     |
| `tests/test_make_moe_fixture.py` | Fixture generator contract (unittest).                                      |
| `patches/moe/*.patch` (PR 2)     | Loader predicate and provider admission call the table.                     |

### Task 1: architecture role table

**Files:** Create `include/xllama/moe_streaming.h`, `src/bridge/moe_streaming.cpp`, `tests/test_moe_streaming.cpp`; modify `tests/CMakeLists.txt`, `CMakeLists.txt` (source list).

**Interfaces — Produces:**

- `enum class MoeRole { gate_exps, up_exps, down_exps, gate_up_exps };`
- `struct MoeArch { std::string_view arch; std::initializer_list<MoeRole> roles; bool has_shared_experts; bool has_expert_bias; };`
- `const MoeArch* moe_arch(std::string_view general_architecture);` — `nullptr` when the architecture is not supported.
- `std::optional<MoeRole> moe_routed_role(std::string_view arch, std::string_view tensor_name);` — parses `blk.<n>.<base>.weight`; returns the role only for the four routed bases listed for that architecture; `nullopt` for everything else, including `.bias` and `.scale`.
- `bool moe_is_unrouted_expert_tensor(std::string_view arch, std::string_view tensor_name);` — true for a `blk.<n>.ffn_*_exps.weight` the table does not route for that architecture (Review Focus 1).

Table rows (pin `d812350`, verified against `src/models/*.cpp`): `lfm2moe` {gate,up,down}; `bailingmoe2` {gate,up,down}, shared; `ernie4_5-moe` {gate,up,down}, shared; `smallthinker` {gate,up,down}; `gpt-oss` {gate,up,down}, bias; `gemma4` {gate,up,down,gate_up}; `afmoe` {gate,up,down}, shared.

- [ ] **Step 1: failing tests** — `moe_arch_known` (each of the seven names returns its row; `"llama"` and `"lfm2"` return `nullptr`), `moe_routed_role_basic` (`blk.3.ffn_up_exps.weight` → `up_exps` for `lfm2moe`), `moe_routed_role_resident` (`blk.3.ffn_up_shexp.weight`, `blk.3.ffn_gate_inp.weight`, `blk.0.ffn_up_exps.bias`, `blk.0.ffn_gate_exps.scale`, `blk.1.ffn_up.weight` → `nullopt`), `moe_routed_role_gemma4_merged` (`blk.2.ffn_gate_up_exps.weight` → `gate_up_exps` for `gemma4`, `nullopt` for `lfm2moe`), `moe_routed_role_malformed` (`ffn_up_exps.weight`, `blk.x.ffn_up_exps.weight`, empty → `nullopt`), `moe_unrouted_expert_tensor` (`ffn_gate_up_exps` on `lfm2moe` → true; on `gemma4` → false).
- [ ] **Step 2:** `./build/linux-test/tests/xllama-tests --test-case='*moe_*'` fails to compile (header missing).
- [ ] **Step 3:** implement; table as a `constexpr` array in the `.cpp`.
- [ ] **Step 4:** same command passes; full `ctest` green.
- [ ] **Step 5:** commit `feat: add the MoE streaming architecture table`.

### Task 2: checkpoint pin and admission check

**Files:** Modify the three files of Task 1.

**Interfaces — Consumes:** `moe_arch`. **Produces:**

- `struct MoePin { std::string id, filename, sha256, arch, template_sha256; uint64_t bytes = 0; };`
- `struct MoeObserved { std::string filename, sha256, arch, template_sha256; uint64_t bytes = 0; };`
- `struct MoeAdmission { bool ok = false; std::string reason; };`
- `MoeAdmission check_moe_checkpoint(const MoePin& pin, const MoeObserved& got);` — fields compared in this order, first mismatch wins: `arch` supported, `filename`, `bytes`, `sha256`, `arch` equal, `template_sha256`. `reason` names the field and both values.
- `struct MoeCatalogueFile { std::string filename, sha256; uint64_t bytes = 0; };`
- `std::optional<MoePin> moe_pin_from_catalogue(std::string_view id, std::string_view arch, std::string_view template_sha256, const std::vector<MoeCatalogueFile>& files, bool from_override);` — the caller (WinRT manifest reader) extracts the fields; returns `nullopt` when `from_override` is true, when `arch` or `template_sha256` is empty, when `template_sha256` is not 64 lowercase hex, or when there is not exactly one `.gguf` file.

- [ ] **Step 1: failing tests** — `moe_pin_admits_exact`, one test per field mismatch asserting `ok == false` and that `reason` contains the field name, `moe_pin_unsupported_arch`, `moe_pin_override_ignored` (`from_override = true` → `nullopt`), `moe_pin_needs_single_gguf` (two `.gguf` rows → `nullopt`; a `LICENSE` row next to one `.gguf` is accepted).
- [ ] **Step 2:** run, expect failure.
- [ ] **Step 3:** implement; no JSON parsing in this unit.
- [ ] **Step 4:** run, expect pass.
- [ ] **Step 5:** commit `feat: admit MoE streaming checkpoints from the bundled catalogue only`.

### Task 3: catalogue entries

**Files:** Modify `uwp/models/manifest.json`, `uwp/model-downloader.cpp` (carry `moe_streaming` on `ManifestEntry`; clear it in override entries), `docs/research/moe-streaming-candidates.md` (status column), `docs/architecture.md` (one line in the module map).

Entries added, each `role: "experimental"`, hidden from the model picker, with `moe_streaming`:

| id                     | file                         | bytes       | sha256      | arch      | template_sha256 |
| ---------------------- | ---------------------------- | ----------- | ----------- | --------- | --------------- |
| `lfm25-24b-a2b-q4-k-m` | `LFM2.5-24B-A2B-Q4_K_M.gguf` | 14526258464 | `d3334405…` | `lfm2moe` | `ea663864…`     |
| `lfm25-8b-a1b-q4-k-m`  | `LFM2.5-8B-A1B-Q4_K_M.gguf`  | 5155564768  | `4923ec14…` | `lfm2moe` | `6d65c880…`     |

(Full digests from the Hugging Face listing at the pinned revision; `hf_base_url` pinned to a revision, not `main`.) Other architectures get entries only after Task 5 passes for them.

- [ ] **Step 1:** add `tests/test_moe_catalogue.py` (unittest, run by CI with the other modules): every entry with `moe_streaming` has exactly one `.gguf` file with 64-hex `sha256`, an `arch` from Task 1's table, a 64-hex `template_sha256`, and an `hf_base_url` pinned to a 40-hex revision; the two ids above are present.
- [ ] **Step 2–4:** fail, implement, pass; `python3 scripts/check-coherence.py` and `tests/test_catalog_licenses.py` green (each entry carries its licence row, so the single-`.gguf` rule of Task 2 is what matters). Add the new module to the Python test list in `AGENTS.md` and the CI workflow.
- [ ] **Step 5:** commit `feat: catalogue the LFM2.5 MoE checkpoints for experimental streaming`.

### Task 4: tiny MoE fixtures per architecture

**Files:** Create `scripts/make-moe-fixture.py`, `tests/test_make_moe_fixture.py`.

`make-moe-fixture.py --arch <name> --out <file> [--seed N]` writes a random-weight GGUF with `n_layer = 2`, `n_embd = 64`, `n_expert = 8`, `n_expert_used = 2`, `n_ff_exp = 32`, vocabulary of 256, Q4_K expert matrices (Q8_0 for shapes Q4_K cannot hold), using gguf-py from the submodule. The metadata keys per architecture are those `src/llama-model.cpp` reads at the pin; the script refuses an architecture outside Task 1's table.

- [ ] **Step 1: failing tests** — `test_each_arch_writes` (seven files, each reopened with gguf-py, `general.architecture` matches, every routed tensor of Task 1 present with `n_expert` in its last dimension), `test_unknown_arch_refused`, `test_seed_reproducible` (same seed → same SHA-256).
- [ ] **Step 2–4:** fail, implement, pass. Then `./build/linux-test/bin/xllama-cli -m <fixture> -p hi -n 4` loads and decodes each fixture resident (proves the metadata is complete).
- [ ] **Step 5:** commit `test: generate tiny MoE fixtures for every streaming architecture`.

### Task 5: generic loader predicate and per-architecture parity (PR 2)

**Files:** `patches/moe/` (the PR 2 native patch), `tests/test_moe_native.cpp`.

**Interfaces — Consumes:** `moe_routed_role`, `moe_arch`, fixtures of Task 4.

- [ ] **Step 1:** in the provider patch, replace `arch_name == "lfm2moe" && tensor ∈ {GATE,UP,DOWN}_EXPS` and the `owned input requires LFM2MoE` check with calls to `moe_arch` / `moe_routed_role`. Refuse at load when a tensor whose base ends in `_exps` is not routed for that architecture (Review Focus 1).
- [ ] **Step 2: failing tests** — for each fixture: resident vs streamed logits over 16 teacher-forced tokens, cache of 2 slots and of all slots, equal within the comparator tolerances frozen in `scripts/compare-logits.py` for exact-reference kernels; `moe_native_gemma4_dense_resident`; `moe_native_gptoss_bias`; `moe_native_unknown_exps_refused`.
- [ ] **Step 3–4:** implement until all pass under ASan/UBSan.
- [ ] **Step 5:** record per-architecture status (fixture PASS/FAIL) in `moe-streaming-candidates.md`; commit with the PR 2 boundary.

### Task 6: console activation for any admitted pin (PR 3)

**Files:** the PR 3 activation files listed in the upstream plan.

- [ ] Replace the compiled-in 24B constants (`cpu_pilot_policy.h`, the model id/SHA literals in the dev-run config and runner) with the `MoePin` resolved from the bundled catalogue by the requested id. The model folder is `models/<id>/`.
- [ ] Replace the exact embedded-template comparison with `template_sha256` of Task 2 and render through the model's own GGUF template.
- [ ] Console ladder per pin, in the candidate-list order: CPU lifecycle with 2 and 512 slots, then the resident comparison where the checkpoint also fits resident (`lfm25-8b-a1b` `UD-IQ3_S` resident against `Q4_K_M` streamed), then benchmark rows through `bench/README.md`'s schema and `generate-benchmark-summary.py`.

### Task 7: further architectures

For each of `bailingmoe2` (Ling-mini-2.0), `ernie4_5-moe`, `gpt-oss`, `smallthinker`, `gemma4`, `afmoe`, in that order: Task 5 fixture PASS → catalogue entry (Task 3 shape) → Task 6 console ladder. A FAIL at any step is recorded in the candidate list with its evidence and stops that model, not the others.
