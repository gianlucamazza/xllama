# M0 / E2 — Mamba2 hybrid `can_shift=1` investigation

**Disposition:** mechanism **confirmed in code**; product impact
**NOT_RUN** (no cold vs post-shift continuation measurement on Granite).
Narrow follow-up proposed; **no default change** in this slice.

**Date:** 2026-11-24  
**Hypothesis (from [#527](https://github.com/gianlucamazza/xllama/pull/527)):**
session logs `can_shift=1` on `granitehybrid`; context shift on recurrent
state may be unsound. Refs #512.

Source follow-up (host-testable predicate + opt-in overflow case) is a
**separate PR** so this docs slice stays docs-only.

## What #527 actually logged

`bench/results/2026-11-24-granite40-h-tiny/hybrid-snapshot.log` (host,
`XLLAMA_GRANITE_H_TINY`):

```text
[xllama] session: can_shift=1 (memory_can_shift=1 n_swa=0) prefix_snapshot=1
```

That line is from `LlamaSession::ensure_ctx` in `src/bridge/session.cpp`:

```text
m_can_shift = llama_memory_can_shift(...) && llama_model_n_swa(...) == 0;
```

The existing #512 host test uses `n_ctx = 512` and short prompts. It
never overflows, so it never takes the #169 shift path. T1
(`t1-host-smoke.log`) goes through `run_inference`, not `Session`, and
does not print `can_shift`.

## llama.cpp pin `d812350` (read, not copied into product)

`granitehybrid` is `LLM_ARCH_GRANITE_HYBRID`, classified hybrid
(`llm_arch_is_hybrid` in `llama.cpp/src/llama-arch.cpp`) and uses SSM
hyperparameters (`ssm_d_conv` / `ssm_d_inner` / `ssm_d_state` in
`llama-model.cpp`). Memory is `llama_memory_hybrid` (attention KV +
recurrent/SSM).

Relevant pin behavior:

| Call                                    | Pin implementation                                                                                                                                                  | Meaning for Granite                                                                                                                                                       |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `llama_memory_hybrid::get_can_shift`    | returns `mem_attn->get_can_shift()` with comment "Shifting is trivially supported for recurrent"                                                                    | Attention half decides the flag. For this arch, `llama_kv_cache::get_can_shift` is true (not Step35, `n_pos_per_embd == 1`).                                              |
| `llama_memory_recurrent::get_can_shift` | always `true` ("shifting the pos is trivial")                                                                                                                       | Not consulted by hybrid `get_can_shift`.                                                                                                                                  |
| `llama_memory_hybrid::seq_rm`           | recurrent first, then attention                                                                                                                                     | Front-drop of `[keep, keep+n_discard)` is the xllama #169 path.                                                                                                           |
| `llama_memory_recurrent::seq_rm`        | Tail rollback in `(p0, end]` can fail; a range that does **not** cover the tail cell is a no-op on SSM tensors (no per-token recurrent cells in the evicted window) | Matches the xllama comment: absorbed history survives.                                                                                                                    |
| `llama_memory_hybrid::seq_add`          | both halves                                                                                                                                                         | Recurrent `seq_add` only does `cell.pos += shift` on the tail cell ("for Mamba-like or RWKV models, only the pos needs to be shifted"). SSM tensors are not RoPE-shifted. |

xllama then calls:

```text
llama_memory_seq_rm(mem, 0, keep, keep + n_discard);
llama_memory_seq_add(mem, 0, keep + n_discard, -1, -n_discard);
```

So on overflow, attention KV is front-dropped and K-shifted; the Mamba2
state keeps the evicted tokens in its compressed history and only the
recurrent cell's **position** moves. That is llama.cpp's documented
approximation, not an xllama-only bug.

`include/xllama/session.h` already says hybrid front-drop is "an
approximation the quality gates measure." Those gates have not been run
for Granite overflow.

## Why this is not a default-off for all hybrids

LFM2 / LFM2.5 catalogue models are also `llm_arch_is_hybrid`
(`LLM_ARCH_LFM2`). `tests/test_session.cpp` treats "LFM2 and
pure-attention archs" as the shift-success contract. Folding
`is_hybrid` or `is_recurrent` into `m_can_shift` would change **shipping**
long-chat behavior. M0 does not do that.

Granite is Mamba2-class SSM plus MoE; LFM2 is a different recurrent cell.
A later fail-fast should be architecture-narrow (`granitehybrid` / Mamba2),
and only after a measured cold vs post-shift mismatch — or an explicit
owner decision to fail-fast without that measurement.

## What was measured in this slice

| Check                                                         | Result                        | Evidence                                                                      |
| ------------------------------------------------------------- | ----------------------------- | ----------------------------------------------------------------------------- |
| Log `can_shift=1` on Granite                                  | Observed (prior host receipt) | `hybrid-snapshot.log`                                                         |
| Pin reports hybrid can_shift from attention only              | Confirmed                     | `llama-memory-hybrid.cpp` `get_can_shift`                                     |
| Recurrent `seq_add` shifts pos only                           | Confirmed                     | `llama-memory-recurrent.cpp` `seq_add`                                        |
| Recurrent front-drop leaves SSM history                       | Confirmed                     | `seq_rm` (no cells in evicted range; tail rollback is a different path)       |
| xllama #169 uses rm+add on that memory                        | Confirmed                     | `src/bridge/session.cpp` context-shift block                                  |
| Cold vs continued/restore after an actual shift               | **NOT_RUN**                   | needs Granite GGUF + overflow turns; existing snapshot test does not overflow |
| Console long-chat / `validate-console.sh longchat` on Granite | **NOT_RUN**                   | no console access                                                             |

**E2 disposition: DEFERRED** on product verdict (approximation confirmed;
correctness vs cold continuation unmeasured). Mechanism is not BLOCKED.

## Narrow follow-up proposal (not executed here)

1. **Host test (opt-in `XLLAMA_GRANITE_H_TINY`):** small `n_ctx`, reuse
   turns until #169 shift fires, require `can_context_shift() == true`,
   then compare greedy continuation after shift to a cold run of the
   surviving prompt. Skip in CI when the env var is unset (same pattern
   as `tests/test_granite_hybrid_snapshot.cpp`).
2. **Host-testable predicate:** extract the current
   `memory_can_shift && n_swa == 0` formula so CI covers SWA / cannot-shift
   cases without a 3 GB GGUF. Do **not** add `is_hybrid` to that
   predicate without an LFM2 regression plan.
3. **If (1) mismatches:** fail-fast (#173) for `granitehybrid` only
   (`can_shift=0` even when llama.cpp reports 1), behind a small Session
   change, with a log line. Re-test catalogue LFM2 separately; do not
   piggy-back.
4. **Do not** change catalogue defaults, n_ctx, or long-chat UI in the
   same PR.

Suggested issue home: a comment on #525 / #512, not a new umbrella. Code
for (1)–(2) belongs in a source PR, not this docs PR.
