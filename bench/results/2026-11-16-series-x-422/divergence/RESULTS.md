# First-divergence / logit-parity comparison for #422 (the #420 protocol), 2026-11-16 evening

This is a **re-run**, performed 2026-11-16 ~17:45–18:05 CST. This morning's
original runs (the #434 figures) saved per-run counts, timings, file hashes
and log excerpts only — no token sequences, no logit data — so the comparison
Gianluca asked for on #422 could not be read back from them. The re-run used
the identical conditions and captured the token sequences and logit dumps
this time. All divergence numbers below are from the re-run. The morning
bundle is referenced at the end, as offered in #434.

## Fixed conditions (identical to the morning study)

Console: Xbox Series X in Dev Mode, 192.168.4.54, Game designation
(`[resources] app_memory_limit_mb=5120` in the app log on every launch).
Package: xllama 1.7.0.1626, PFN
`GianlucaMazza.xllama_1.7.0.1626_x64__pj67f1fcj4n14`. Model:
lfm25-1.2b-instruct (LFM2.5-1.2B-Instruct-QAD-Q4_0.gguf). Generation road:
LAN API `/v1/chat/completions`, one user message "Write a short paragraph
about the sea at dawn.", temperature 0, max_tokens 128. Logit road: the
project's own headless probe (`logits.flag` → `run_logits`, the #420
instrument), backends selected by `logits_gpu_layers.txt` (0 / 99, resolved
value read back from the device log on every run), per-position dumps via
`logits_stride.txt = 1`. Comparator: `scripts/compare-logits.py`.

## Reproduction of the morning result

| Run | prompt_tokens | completion_tokens | finish | content SHA-256 |
|---|---|---|---|---|
| CPU 1 (knob "0") | 31 | 121 | stop | 923d4b39866d782a… |
| CPU 2 (knob "0") | 31 | 121 | stop | 923d4b39866d782a… (identical) |
| GPU 1 (knob deleted → catalogue 99) | 31 | 125 | stop | 658eda0ee24271b6… |
| GPU 2 (knob deleted → catalogue 99) | 31 | 125 | stop | 658eda0ee24271b6… (identical) |
| Final check (knob restored "0") | 31 | 121 | stop | 923d4b39866d782a… |

**The re-run reproduced the 121 vs 125 split exactly**, with each backend
deterministic across repeats (byte-identical content within each arm).

## Item 1 — temperature 0 took the greedy path: CONFIRMED

- Source: `include/xllama/sampling.h` — `SamplingConfig::is_greedy()` returns
  true when `temperature <= 0.0f`; `src/bridge/sampler_chain.h` then builds a
  chain containing only `llama_sampler_init_greedy()`. The LAN API Session
  path uses this same chain builder.
- Empirical: two generations per backend are byte-identical (table above),
  and every probe dump sidecar records `"greedy": true` (run_logits sets
  greedy explicitly).

## Item 2 — index of the first different token: generated token 35 (index 34)

Token sequences were segmented from the output texts with the model's own
GGUF tokenizer data (tokens + merges extracted from the console model's
GGUF; Llama-3-style pre-tokenizer pattern, which is what this build's
`lfm2` pre type implements). Validation, three independent oracles:

- Segmented counts equal the API usage figures exactly: 121 (CPU), 125 (GPU).
- The device probe's own logged token counts for the standalone strings
  match the segmentation everywhere: CPU text 122 incl. BOS, GPU text 126
  incl. BOS, raw prompt 11 incl. BOS, CPU-text prefixes of 120/240/360/480
  chars → 27/56/81/109.
- The probe sidecars for the two teacher-forced strings report
  n_prompt_tokens 131 and 135, matching the segmentation exactly.
- Detokenize(segment(text)) == text for both outputs.

Result: the two sequences share a **34-token common prefix**. The first
difference is at **generated index 34 (the 35th generated token)**:

- CPU: token **9557**, piece " motion" → "...their rhythmic motion echoing
  the soft whispers of the morning."
- D3D12: token **64051**, piece " sigh" → "...their rhythmic sighs echoing
  the stillness of the morning."

(The shared text prefix ends "...their rhythmic".) Full sequences:
`raw/generated_token_ids.json`, `raw/generated_cpu_tokens.txt`,
`raw/generated_gpu_tokens.txt`.

## Item 3 — top-k and logit gap at that index: NAMED GAP in the generation context; measured in the probe's raw context

**The gap, with evidence.** No road in the installed package dumps the
decode-time logit row for the chat-templated context:

- The only logit-dump road is the headless probe, and it tokenizes its input
  raw (run_inference with parse_special=false; the Session chat path uses
  parse_special=true — see the comment at src/bridge/session.cpp:604).
  Measured: the rendered ChatML template string that the chat path tokenizes
  to **31 tokens** tokenizes to **51 tokens** in the probe (device log,
  "prompt tokens: 51"; the dump is kept as `raw/probe_tmpl_raw.bin`).
- The LAN API returns `logprobs: null` by design (docs/api-endpoint.md;
  uwp/api-server.cpp).

So the generation-context top-k is recorded as a **named gap**, not
approximated. What the #420 instrument does produce — its native mode — is
the same comparison on an identical raw context fed to both backends:
the user prompt concatenated with the generated continuation, teacher-forced.
At the position corresponding to the split (position 43 of the raw-context
tokenization, predicting the split token), the two arms' rows are:

| Rank | CPU logit | D3D12 logit | Token |
|---|---|---|---|
| 1 | 15.7324 | 15.7449 | " mur" (7843) |
| 2 | 15.4889 | 15.4407 | " sigh" (64051) |
| 3 | 15.2915 | 15.3083 | " pulse" (24037) |
| 4 | 14.9317 | 14.9582 | " dance" (14105) |
| 5 | 14.8275 | 14.6245 | " wh" (901) |
| 6 | 14.3945 | 14.4497 | " motion" (9557) |
| 7 | 14.1802 | 14.1401 | " breath" (8311) |
| 8 | 14.0881 | 14.0646 | " embrace" (32000) |
| 9 | 14.0666 | 14.2119 | " e" (821) |
| 10 | 13.7123 | 13.6573 | " whisper" (64035) |

Top-10 sets are identical (overlap 1.00); top-1 agrees (" mur" on both).
**Logit gap (top-1 − top-2) at the split position: 0.2434 on CPU, 0.3041 on
D3D12.** Honest context note: in the raw context neither backend puts its
generation token first — the templated context is what produced " motion"
and " sigh" — and the GPU generation's token " sigh" ranks second on both
backends there. As a measure of how much the raw context differs from the
generation context: the CPU arm's own greedy argmax in the raw context
agrees with the teacher-forced text at 93/130 positions (CPU continuation
string) and 98/134 (GPU continuation string).

## Item 4 — max absolute difference of the logit row, teacher-forced after the split

From the stride-1 per-position dumps (both backends, identical input per
pair; rows compared position by position, compare-logits conventions):

- **At the split row:** max_abs_diff **0.248774**, NMSE **3.389e-04**,
  top-1 match, top-10 overlap 1.00. The project's comparator on the
  extracted rows (`raw/splitrow_scpu_cpu.bin` vs `raw/splitrow_scpu_gpu.bin`)
  prints **PASS** under its default thresholds (max-abs ≤ 0.5, NMSE ≤ 1e-3,
  top-10 overlap ≥ 0.8).
- **After the split, teacher-forced along the CPU continuation** (87 rows):
  largest per-row max_abs_diff **0.3614** (at position 111); mean of per-row
  maxima 0.2518.
- **After the split, teacher-forced along the GPU continuation** (91 rows):
  largest per-row max_abs_diff **0.3809** (at position 134, the final row);
  mean of per-row maxima 0.2487.
- Determinism check: the at-split rows extracted independently from the two
  continuation probe pairs are byte-identical per arm (SHA-256
  f558e59c… CPU, 0174e186… GPU), as expected — the split row is a property
  of the shared prefix, measured twice.

All of item 4's rows are from the raw context described in item 3 (the
generation-context rows are the named gap there).

## Console state discipline

- On arrival: `gguf_gpu_layers.txt` = "0" (1 byte, SHA-256
  5feceb66ffc86f38d952786c6d696c79c2dbc239dd4e91b46729d73a27fb57e9),
  api.flag present, model.txt = lfm25-1.2b-instruct, health ok.
- GPU generation arm: the knob file was DELETED through the portal
  (morning's road), verified absent two ways (download 404; absent from the
  LocalState listing); the app was restarted and the log shows
  "offloaded 17/17 layers to GPU".
- Probe cycles: api.flag / model.txt / prompt.txt stashed and restored
  byte-identical after every cycle (readback match verified each time);
  probe knob files (logits_gpu_layers.txt, logits_stride.txt, logits.flag)
  deleted after each cycle.
- Restore: knob re-uploaded as "0" with `path=\LocalState`, read back b'0',
  SHA-256 matching the arrival value; app restarted; health ok; a fresh
  generation reproduced the CPU control exactly (121 tokens, content hash
  identical to the CPU arm).
- Final verification (after all probes): LocalState listing is identical to
  the arrival listing (the last probe's logits.bin / logits.bin.json /
  logits.done outputs were deleted in this final pass), knob "0",
  api.flag present, original prompt.txt and model.txt in place, app running
  in UI/API mode, health ok. The console stands exactly as found.

## Deviations and incidents

1. **Raw-context logit measurement (items 3–4)** — see the named gap in
   item 3. The generation-context top-k/logit rows cannot be produced by
   the installed package; nothing was approximated or relabelled.
2. **First probe cycle's cleanup was interrupted.** That cycle ran as a
   background job whose completion summary misreported its save path and
   the template token count (it claimed 31; the device log says 51). The
   discrepancy was caught from the console's own state (api.flag missing,
   health down, probe files present), the kept dump's hash matched the
   summary's hash so the dump itself was sound, and the state was repaired
   and re-verified from portal readbacks before any further work. Every
   figure in this file was taken from files and device readbacks, not from
   run summaries.
3. The probe's whole-log tail mixes earlier launches; per-run claims here
   rest on the saved sidecars, the positions sidecars' token counts, and
   the portal readbacks, which are unambiguous.

## Morning bundle (from the original runs, as offered in #434)

From `~/workspace/scorpio_bench/series_x_untested_paths/gpu/` (SHA-256,
recorded in that study's RESULTS.md):

- `runs_cpu_control.csv`: 5309b40ef68ef0d963b9ef3ce699944b62004561cc590ee117c530108c69f600
- `runs_gpu.csv`: e785c52bca45da6754d8a9e27a9cf11cc17a600b76a7bbf1287c2beae950a44c
- `runs_350m_witness.csv`: f18b8847e74c188d0cfd7012c876b6a8d5cade122cfc43a855fae17f8404aee1
- `xllama_log_excerpt.txt`: d24a49b1d9aa52caf62e0f17147dcc54e6572021bb923fd1fbc500adbe3645a4

## Files produced by the re-run (SHA-256)

In this folder's `raw/` (pulled from the tower work dir
`~/xllama_divergence_work/out/`):

- `gen_cpu_1.json` a54ce2b322fc5ae4845ee036ee9dcc8b518fe3b4fee7f2cec28796607105ca1e
- `gen_cpu_2.json` ee6d29fbbb1ac352cf3f4ddb5bf4015045d8039c33646a17ba0a5824c711c889
- `gen_gpu_1.json` c30fad51b88bf28bc06fd37b6c6a5dea4486c18f147b73ccf48c416791c60ada
- `gen_gpu_2.json` b70fc7a958258327b63295ba8897c3328ba25d38f6bd18c91199d230fb7d2bac
- `gen_final_1.json` 77f6f57b6c53cff101dd29f82bd91b7d6c4db5bd9c487146aadb70c7edf741d8
- `generated_token_ids.json` 22f04cf081d5eff0d3246a88386cb6c4759a4f17bf0c27fa5898899c309c1802
- `generated_cpu_tokens.txt` c4f09150cb727824fe9c6fd01c2b4e90a5ffda17be058fce17a02764c4bca4e3
- `generated_gpu_tokens.txt` 8e8705f5d80585debfec3804d78b98b32adaef78642504dfb33cb8c9e3e11896
- `splitrow_scpu_cpu.bin` f558e59cabc0f2723bebc3ecee879b485d83e6f47b8a54f9bfbf0b04469a43b7
- `splitrow_scpu_gpu.bin` 0174e1864fcbf84f789623c00a712611ca33aebfb82dc0000e6fba237946373a
- `probe_tmpl_raw.bin` 7975c740c7ffcecac257a86b95813521d78992d880efc5d994d5de14506cd0aa (the 51-token template probe dump, gap evidence)
- `probe_tmpl_raw.bin.json` cb98a838c1611fefcdc43d48600cbdb43bac3f1f349385e47ed7806089a521d0
- `analysis.json` 3653b782172967142a902697f2c218b68d94a71291a371f02a5fd383e2fd6bbb (full computed record: top-10s, per-string stats)
- Probe sidecars and logs: `probe_s{cpu,gpu}_{cpu,gpu}.bin.json`,
  `probe_s{cpu,gpu}_{cpu,gpu}.bin.positions.json`, `probe_s*_*.log`,
  `probe_tmpl_cpu.log` (hashes in the tower manifest; sidecar hashes:
  scpu pair d23279fb…, sgpu pair 89a0aec2…; positions sidecars 3fe4bd3f… /
  c664a946…).

Kept on the tower (`~/xllama_divergence_work/out/`, hashed; large files
follow the morning bundle's precedent of tower-side retention):

- `probe_scpu_cpu.bin.positions.bin` e6d1e831a5e80b76737d70c7aa4a679f835c059a09ef69a0b905b2dabc0d131a (34,340,864 bytes, 131 rows)
- `probe_scpu_gpu.bin.positions.bin` 2c6b5279c987d9fc9ec78b7194616cf2a6c83ec735618f3407f4ee392c453fd7
- `probe_sgpu_cpu.bin.positions.bin` f5b3b69483167d328130e4e99ac734d4593bb186db216df8eaef83cb3510273b (35,389,440 bytes, 135 rows)
- `probe_sgpu_gpu.bin.positions.bin` 74d4a612edd5b4abf28ffea28ffdc19440ef07c8d83b4c2fa8017b6e365d4dfd
- Last-token dumps `probe_s{cpu,gpu}_{cpu,gpu}.bin`: 9e6f5b5c…, a02b6cb7…, 33f12c3f…, d388029c…
- Probe input strings: `s_cpu.txt` b7a053f9cfaddbe5de506ee962d29ac7261cc6039c8757ddf17b35461740977b, `s_gpu.txt` c7bbaa00326dc06ed2421fd7ae84bb888c1343240ae9a8f8bd53f542212a82ee

Nothing was posted anywhere. The post-ready comment is in
`FOLLOWUP_COMMENT_DRAFT.md` (SHA-256 ae6220e3e9008d3762a74df74e563e22e3bb96cc494018373d2de32ba0d08068).
