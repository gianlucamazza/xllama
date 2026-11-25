# Changelog

All notable changes to xllama are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

- **Context shift evicts whole turns.** A long chat past `n_ctx` dropped a raw token count after the system prompt, which could cut a message or a chat-template marker in half. The shift now ends right after the format's turn terminator (`GenerateParams::turn_close`, set by the chat UI from `ChatFormat::turn_close`); with no boundary that frees enough room it fails fast and the full-prompt retry drops whole turns. A host recall probe found no evicted-turn fact surviving in LFM2.5 or Granite-4.0-H recurrent state.

- **Hybrid conversations restored from disk keep their prefix checkpoint.** The `LocalState\kv\<id>.kv` snapshot now ends with an optional `xllama-ckpt` trailer holding the partial-state checkpoint. After switching back to a conversation, an edited reply re-prefills only the tail instead of the whole history. Files without the trailer still load, and older builds ignore it.

- **Hybrid prefix reuse keeps a partial checkpoint instead of a full copy.** LFM2 and Granite-4.0-H sessions copied the whole context state (`llama_state_get_data`, attention KV included) after every prefill. They now keep a partial-state checkpoint (`llama_state_seq_*_ext`, `PARTIAL_ONLY`, as llama-server's context checkpoints): 160 KiB for LFM2.5-1.2B, independent of `n_ctx`. A restore truncates the still-resident attention KV to the checkpoint, and an exact regenerate now restores too, with output identical to a cold run, instead of re-prefilling the whole prompt. A context shift drops the checkpoint.

- **The chat UI no longer continues on another conversation's KV.** A LAN API request on the same model rewrote the resident KV without changing `hub.generation`, so the GUI's next turn appended its message to the API conversation, and a conversation switch could save the API's KV into the GUI's snapshot file. `Session::kv_epoch()` now identifies the resident KV: a continuation (`GenerateParams::expected_kv_epoch`) on a rewritten KV is refused before anything is touched and the turn re-prefills its own context; the snapshot save checks the same epoch. A full prompt past `n_ctx` is refused before the KV rewind, so it no longer destroys the resident conversation, and the GUI shows the same "prompt too long" message as the API instead of generating a silently shortened reply. One wording for every surface lives in `prompt_budget.h`.

- **Hybrid prefix snapshots may start a few tokens before the prompt end.** On a long hybrid/recurrent full-prompt prefill the session captures the head, then prefills the tail, so a later prompt that diverges only in those last tokens can restore instead of prefilling from scratch. Exact regenerate of the same prompt still Clears and prefills the whole prompt so the last token reseeds logits. The opt-out knob and non-hybrid rewind are unchanged. On host (LFM2.5-1.2B, CPU) the head/tail split produces the same greedy output as `hybrid-prefix-snapshots=0`; an opt-in `XLLAMA_TEST_MODEL` case guards it. Watchdog prompt progress stays absolute across the split.

- **D3D12 shared-activation cache keyed by tensor identity.** Hardening of the d3d12 shared-activation cache: lookups use tensor owner, data pointer, column stride, type and row size, not address alone. Latent hazard; no known failure on main.

## [1.8.1]

User-facing notes: [docs/releases/v1.8.1.md](docs/releases/v1.8.1.md).

- **Device Portal credentials off argv.** Console scripts pass `XBOX_USER`/`XBOX_PASS` to curl through a 0600 `-K` config file instead of `-u` on the command line.

- **Opt-in GPU ROPE for d3d12 GGUF layers.** With
  `LocalState\d3d12-rope.txt` set to `1`, plain F32 NORMAL/NEOX `ROPE` (no
  `freq_factors`, no YaRN) runs on the d3d12 backend, so q/k/v of a layer form
  one split with the bias ADD knob on (5 → 4 submissions per layer). The
  cos/sin table is computed on the CPU. The Series X comparisons reported in
  #511 (operator-declared, Series X) matched logits bit for bit for the three
  models and two ubatch sizes tried. Series S self-test at `d599486`
  (correctness and bit-identity only, no speed claim):
  [`d3d12-selftest-d599486-rope-20261124.csv`](bench/results/d3d12-selftest-d599486-rope-20261124.csv)
  (`--require-rope`, GATE=PASS 34/34, all ten `rope_*` rows bit-identical).
  Details: [GGUF GPU decode](docs/gguf-gpu-decode.md).

- **Granite 4.x chat renderer (#525).** `chat_format_for` selects
  `<|start_of_role|>…<|end_of_role|>…<|end_of_text|>` for ids and filenames
  whose basename contains `granite`, shared with #521 (Granite 4.2-3B). Host
  T1 and Series S T3 receipts are under
  `bench/results/2026-11-24-granite40-h-tiny/`. Not a catalogue entry. The
  #273 row is [model-matrix.md](docs/model-matrix.md) §I1.

- **D3D12 MMV for IQ3_XXS, IQ4_XS, Q5_K.** The ggml `d3d12` backend accepts
  those weight types for MUL_MAT once the matching compute pipelines are
  present (same q8 activation path and selftest contract as Q4_0 / Q4_K /
  Q6_K). A failed Q5_K / IQ4_XS / IQ3_XXS pipeline creation leaves that type
  on the CPU and does not take down the backend. IQ4_XS / IQ3_XXS decode
  kernels keep the codebook/grid in groupshared. The 3×256 IQ3_XXS selftest
  row is a 98-byte unaligned-row correctness case and is excluded from the
  decode speed gate. IQ4_XS decode uses the common ≥100 GB/s packed floor.
  IQ3_XXS decode uses a measured 55 GB/s packed floor (receipt
  `bench/results/d3d12-selftest-151804d-20261124.csv`, about 11% below the
  2048×2048 row; not a product speed claim). Catalogue Q4_0 / Q6_K models
  are unchanged. See
  [GGUF GPU decode](docs/gguf-gpu-decode.md).

- **Hybrid-cache prefix snapshots.** GGUF sessions already reuse a supported
  KV prefix and fall back to a full prefill when a hybrid cache (LFM2) refuses
  a tail rewind. After each successful prefill the session now keeps an
  in-memory prefix state (logits + memory) for hybrid/recurrent models, and
  restores it when the next full prompt is a strict extension of that saved
  prefix, so API and multi-turn turns re-prefill only the suffix. An exact
  regenerate Clears and re-prefills so greedy output matches a cold run.
  `LocalState\hybrid-prefix-snapshots.txt` = `0` opts out. Non-hybrid #170a
  rewind is unchanged. Idea from bjoern2088, rebuilt in-tree. See
  [session.md](docs/sdk/session.md) and
  [architecture.md](docs/architecture.md#kv-cache-reuse-both-backends).

- **Opt-in GGUF thread count for interactive sessions.**
  `LocalState\gguf_n_threads.txt`, when present and positive, sets llama.cpp
  threads for GUI and LAN API GGUF chat through the shared session resolver.
  Absent, 0, or an invalid value keeps auto-detect. GPU-layer knobs and
  d3d12 placement are unchanged. Headless bench still uses `bench_threads.txt`.
  Idea from Chorss, rebuilt in-tree.

- **Opt-in LCM diffusion scheduler.** `LocalState\diffuse-lcm.txt` set to `1`
  uses the Latent Consistency Model step table instead of Euler. Absent, the
  SD-Turbo Euler path is unchanged. See
  [using-the-app.md](docs/using-the-app.md).

## [1.8.0]

User-facing notes: [docs/releases/v1.8.0.md](docs/releases/v1.8.0.md).

- **Opt-in GPU bias ADD for d3d12 GGUF layers (#504, contributed by Chorss).** With
  `LocalState\d3d12-bias-add.txt` set to `1`, the F32 bias ADD after a
  projection runs on the d3d12 backend, so it no longer ends the GPU split.
  This affects models with biases, such as Qwen2.5 and Bielik. The default stays off. See
  [GGUF GPU decode](docs/gguf-gpu-decode.md).

- **SSE for LAN tool proposals.** Requests with `stream: true` receive the
  parsed assistant output and indexed tool calls as SSE, followed by the
  finish reason and usage. The tool path emits its first event after parsing
  the complete generation.

- **Actionable tool-proposal errors (#497).** LAN API overflow returns
  `context_length_exceeded` with token usage and an optional field when the
  budget failure is attributable to tools or messages. An ONNX tool proposal
  returns `tools_unsupported_for_model` and suggests a tool-capable GGUF or plain
  chat. Generation uses the grammar and stop tokens from the selected transcript
  after trimming.

- **Series S App cap after the launch window (#457).** On package 1.7.0.1667
  the default CPU chat (`lfm25-350m`) was measured in the UI process with the
  LAN API, the request starting more than 20 seconds after launch. Both
  designations generated the same 103 tokens. After that window, App is slower
  than Game on prefill and on the logged decode checkpoint. The 2026-08-21
  headless result and the Game-metadata verdict are unchanged
  ([release-readiness.md](docs/release-readiness.md) §6,
  `bench/results/store-app-vs-game-2026-11-18.csv`).

- **Series X App designation explained (#457).** An independent tester found
  that with the App designation the app runs at full speed for about ten
  seconds after launch, then at about 40 % of the Game speed, and that fewer
  threads make it slower. It is a CPU cap of the App designation; the remedy
  is the Game designation. The headless bench cannot show it, so the release
  gate's App measurement now uses the UI process more than 20 seconds after
  launch.

- **Per-layer D3D12 dispatch attribution (#421).** The same opt-in
  (`d3d12-dispatch-timing.txt` = `1`) now tags every timed dispatch with a
  layer index from the weight or graph-node name, and logs one
  `d3d12-layer` line per phase × layer × op. `gpu_ms` is the GPU timestamp
  sum; `wait_ms` is `exec_wait_ms − submission GPU time, divided by n`.
  That remainder can come out slightly negative because the CPU and GPU
  clocks are different domains. Missing timestamps print as `missing`.
  Nodes with no layer go in `unattributed`. Off, the backend path is
  unchanged. `scripts/parse_d3d12_layer_log.py --csv` writes those rows to
  a sidecar, one row per data line, and refuses to overwrite the log it
  read. A missing timestamp stays the word `missing`. The bench CSV
  schema is unchanged. One Series S run of package 1.7.0.1667
  (`lfm25-350m`, knob on) produced the lines for layers 0–15 and
  `unattributed`, with no timestamp left missing
  ([evidence](docs/research/evidence/2026-11-18-d3d12-layer/README.md)).
  On the same package the shared opt-in (dispatch timing and layer lines
  are one file) costs about the same as that per-dispatch knob: prefill
  unchanged and about 4 % of decode on
  LFM2.5-1.2B and Coder-3B. LFM2.5-350M loses about 6 % of decode and about
  3 % of prefill, and that prefill cell is wider. Peak memory on the GPU
  arms is unchanged. A paired CPU arm shows the host working set rising
  when layers move to the GPU (42 MB, 50 MB and 122 MB). Diagnostic only
  ([evidence](docs/research/evidence/2026-11-18-d3d12-layer-overhead/README.md)).

- **Per-dispatch GPU timing (#421).** With `d3d12-dispatch-timing.txt` = `1`
  in LocalState, the d3d12 backend times every dispatch and logs, after each
  prefill and decode phase line, the GPU time per kind of dispatch (operation,
  weight type, kernel variant, thread width, column class). Off by default;
  the barriers it adds serialise dispatches, so it is a diagnostic, not a
  benchmark setting. Measured on the Series S, it leaves prefill unchanged and
  costs about 4 % of decode; its first use showed that the four-column kernel
  of #414 lost on LFM2.5-1.2B in its Q4_0 shader (1.14 to 1.25 times slower
  per dispatch) while its K-quant shaders took half the time.

- **GPU correctness controls (ADR 0010, #420) and the four-column kernel
  (#313, #414).**
  - ADR 0010 was accepted, run once, found to fail an arm whose kernels are
    correct, and amended. A GPU control now checks every weight product
    against a float64 reference built from the model file and the arm's own
    quantized operand (`control-parity.py reference`, q8_K and q8_0 operands),
    and compares the output distribution over thousands of prompt positions
    with what a second CPU build shows (`control-parity.py distribution`).
    `--dump-logits-stride N`, or `logits_stride.txt` on the console, dumps
    those positions; the tensor capture gains `matmul_inputs=<node>`.
    `check-matmul-reference.py` runs the reference check on its own.
  - On the Series S both gate models pass with the shipping kernels:
    Coder-3B (all 36 layers within 4.3e-14 NMSE, KL-divergence ratio 0.989
    against a second CPU build) and LFM2.5-1.2B (all 16 layers within
    1.7e-14, ratio 1.035 against llama.cpp's plain CPU kernels). The first
    control, under the original text, failed and stays recorded as failed.
  - The four-column prefill kernel passes the same control, with output
    byte-identical to the single-column kernel's, and Gate A (50/50). Its
    preregistered paired performance campaign (6 cells, 240 runs) failed:
    prefill 1.50 times on Coder-3B and 0.84 times on LFM2.5-1.2B, the model
    that runs on the GPU by default, with decode and memory unchanged. The
    kernel is not promoted: the draft (#414) is closed without merge and its
    branch is kept.
  - Evidence: `docs/research/evidence/2026-11-15-*` and `2026-11-16-*`.

- **Paired bench campaigns.** `scripts/bench-paired-knob.sh` runs matched
  A/B bench pairs that differ by one LocalState knob, alternating the order,
  checking from the app log that each arm ran the intended code path, and
  never replacing a failed arm. `scripts/paired-bench-summary.py` gives each
  cell's verdict with a paired bootstrap interval against preregistered
  margins. The first use is the #313 campaign of the four-column kernel.

- **Startup log:** a `[resources]` line states whether the title has the
  expanded (Game) resources, how many exclusive cores it was given, the
  logical core count and the app memory limit. It is there to explain runs
  like the 2.4 times slower App-designation runs seen on a Series X (#457).

- **LFM2.5 figures remeasured on 1.7.0.** `lfm25-230m` is 127.1 tok/s decode
  (237 MiB peak) and `lfm25-350m` is 97.9 tok/s on package 1.7.0.1563, each a
  median of three runs of exactly 512 generated tokens. The previous 132.4 and
  101.6 tok/s came from the 1.6.0 package and from runs that stopped after 33
  to 273 tokens, where decode is faster; the one earlier 512-token run of the
  230M was 128.5. The summary, README and model matrix use the new rows.

- **Scripts:** the deploy, bench and validation scripts no longer call
  `base64 -w0`, a GNU-only flag that fails on macOS. They strip the line
  breaks with `tr` instead; the encoded value is the same.

- **Series X is supported.** Accepted by the owner on 2026-11-15 from an
  independent tester's runs of package 1.7.0.1495 on a Series X: text
  generation with `lfm25-350m` and `lfm25-1.2b-instruct` through the LAN API
  (143 timed runs, imported as `bench/results/series-x-2026-11-13.csv` and
  `series-x-end-to-end-2026-11-13.csv` with the `xbox-series-x-declared`
  label; credit Joshua Henne / Eidon Labs Inc.). Image generation, training
  and the GPU path have not been run on a Series X, and every published
  figure remains a Series S figure. With the App designation the tester's
  runs were about 2.4 times slower than with the Game designation; the cause
  is not known. The public wording changes from "Not yet tested on Series X"
  to "Supported on Series S and Series X", with the scope in
  `docs/positioning.md`.

- **Model download:** a download now checks the free space on the LocalState
  volume before the first request and fails at once with both numbers ("needs
  N MB, M MB available") when the files would not fit. A write error deletes
  the partial `.part` file, which used to stay behind and hold the space the
  next attempt needed. The same message reaches the UI status and the
  `/api/pull` error record.

- **Storage diagnostics:** the boot log states the capacity and free space of
  the volume that holds LocalState (`[storage] LocalState volume=...`). On the
  console that volume is not the one the Dev Home storage allocation sizes.

- **Model download:** the downloader no longer writes model bodies to the
  system HTTP cache while it streams them to LocalState. On a Series S, a
  2.45 GB download failed repeatedly with "There is not enough space on the
  disk" after 0.6 to 1.45 GB, with more free space than that reported.

- **Gemma 4 chat template:** `gemma4-e2b` and other Gemma 4 ids now get the
  Gemma 4 turn tokens (`<|turn>`, `<turn|>`) and a dedicated system turn. They
  were rendered with the Gemma 3 template, whose `<start_of_turn>` and
  `<end_of_turn>` markers are plain text in the Gemma 4 vocabulary, so the
  model saw no turn structure and could write the marker back as text. Chat
  UI, LAN API, CLI and device training share the one format. Gemma 3 ids are
  unchanged. Found in a community fork (EgbertJiang/xllama). The Series S
  prefill, decode and peak figures for `gemma4-e2b` were remeasured with this
  template on 2026-11-15. Its H9 rerun is 6/8, the same total as before on
  a different set of tasks (`constrained_summary` now passes, `json_extraction`
  now fails on a Markdown fence); see `docs/model-matrix.md`.

- **Startup model check:** `EnsureModelAsync` now ends with `co_return`. It
  was declared as a coroutine and contained no coroutine statement, so it was
  an ordinary function that returned no value (undefined behaviour). The MSVC
  package was not seen to fail; a community fork (EgbertJiang/xllama) reported
  a trap in a Clang cross-build after model provisioning.

## [1.7.0] - 2026-11-10

User-facing notes: [docs/releases/v1.7.0.md](docs/releases/v1.7.0.md).
Released as `v1.7.0.0` from CI package 1.7.0.1495, which completed Series S
acceptance; receipts and limitations are linked from the notes. Host-side
validation tooling merged after the candidate (#423 strict numerical parity
comparisons, #424 attributable ORT profile reports) and the #425 GPU
acceleration baseline are in the tag; they do not change package inputs.

- **GPU validation:** the standard backend selftest exits nonzero when its
  gate fails and rejects empty, non-finite or negative measurements. Native
  matmul and get-rows error oracles reject non-finite values instead of
  allowing NaNs to disappear in a maximum reduction.

- **API validation:** parse the complete training-status response before
  applying the bounded display preview in the HTTP framing gate. Longer
  progress payloads no longer cause a false failure; missing or invalid
  counters still fail closed.

- **Console tooling:** reject Device Portal file-upload failures consistently
  in deploy, benchmark and validation scripts, including HTTP 200 rejection
  payloads and transport errors previously mistaken for empty-body success.

- **Diffusion:** add a separate fixed deployment profile for first-party
  exports. Reshape targets must be proven constant and free of literal zeros
  before normalization for DirectML; retain the dynamic export for host parity.
- **Diffusion export:** preserve the checkpoint's native CLIP vocabulary and
  merges alongside the unified tokenizer metadata, so first-party exports
  remain self-contained for the console tokenizer with Transformers 5.

- **Host training:** the CPU recipe uses an isolated Python 3.12 environment
  synchronized from a hash-locked dependency graph. System-package inheritance
  and an unpinned Torch fallback are rejected. Operator setup and qualification
  scope are documented in [training/README.md](training/README.md).

- **Docs:** Current prose uses Microsoft's term Developer Mode in place of
  "Dev Mode". Peak working-set receipts that were written in MB now use MiB;
  the digits are unchanged. Historical changelog entries, code, and app strings
  are left as they are.

- **Fixed:** `lfm25-1.2b-thinking` downloads again. LiquidAI reconverted the GGUF on
  2026-10-29, so `resolve/main` no longer matched the catalogue's SHA-256. The
  entry now points at the revision holding the measured file; the bytes are unchanged.

- **Fixed:** A tool proposal on `/v1/chat/completions` no longer returns a thinking
  model's `<think>` reasoning in `content`, matching plain chat. A reply cut off
  mid-thought has empty `content` and `finish_reason` `length`.

- **`scripts/quantize.sh` takes an optional `--imatrix FILE`,** passed through to
  `llama-quantize` for self-converted GGUFs. Calls without the flag keep the same
  arguments, and no catalogue entry changes.

- **Fixed:** Replaying a tool call through `/v1/chat/completions` no longer
  returns a false `400` `model chat template does not render tools`. The check
  counted the tool's name in transcript fields the template may not print: `name`
  on a `tool` message, or the assistant's calls on templates without a `tool_calls`
  branch (LFM2-2.6B, the catalogue's LFM2.5-1.2B-Thinking file). It now compares
  the prompt rendered with and without `tools`. Phi-4-mini is still refused.

- **Catalogue sidecar edits no longer re-download verified weights, and GPU
  routing ignores a still-downloading DML tree.** Fetch planning is per file
  (`classify_file_pin`): a missing LICENSE/NOTICE is fetched in the background
  after the chat model is marked ready. GPU routing requires every expected
  catalogue file, a LocalState `.complete` marker, and an idle writer;
  otherwise it stays on CPU and logs the reason. A GPU folder that has every
  filename but no marker is hashed again (Ok weights are skipped) so the
  marker can be published; a sidecar-only write does not publish `.complete`
  over unhashed weights. New conversations stay on CPU while any write —
  including that hash pass — holds the gate, and the choice sticks for the
  conversation. Background provisioning now catches and logs exceptions
  instead of letting them terminate the process.

- **Fixed:** After an in-place upgrade, adding or changing a licence or notice
  row in the catalogue no longer makes already-verified weight files look
  stale. Which pins count as weights comes from `files[].role` (`weight` /
  `data`, or `licence` / `notice` / `sidecar`); when `role` is omitted the
  documented filename default in `model_file_plan.h` applies. Missing licence
  or notice files download on their own. Routing classifies those weight pins
  by size only and also requires the LocalState `.complete` marker, which
  attests every pin was hashed at download or promotion; otherwise it stays
  on the CPU model or reports not-ready, and it does not load a partial
  download.

- **GUI and LAN API share compute admission** (proposed [ADR 0008](docs/adr/0008-process-compute-admission.md),
  audit F08). A process-wide RAII permit serializes chat, embeddings, diffusion,
  device training, startup pre-load and post-pull model loading. API requests
  receive the existing busy response while the slot is occupied (streaming pulls
  emit an error event); a streamed chat reply keeps the permit until the last
  SSE event, including when the client disconnects mid-stream. The LAN listener
  dispatches each connection off `ConnectionReceived` so a second LAN request
  can observe 503 while the first is still streaming. Startup pre-load retains
  its bounded first-request wait for the API and the first GUI send.

- **Image dialog:** a static "Powered by Stability AI" caption at the bottom
  of the dialog's settings when the image model is Stability-licensed (today:
  SD-Turbo, the only image model). The helper checks a licence string first
  (the catalogue has no licence field yet, so it is unused today), otherwise
  the `sd-turbo` model-id family. The line is a licence credit, not a
  partnership or endorsement claim.

- **Qwen2.5-Coder-3B withdrawn from the catalogue.** The 3B Instruct weights
  are under the
  [Qwen Research License (non-commercial)](https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct/blob/main/LICENSE)
  and are no longer downloadable in-app or via `/api/pull`
  (`api_pull_model_allowed` rejects `qwen25-coder-3b`; LoadModelManifest
  drops the id from LocalState overrides). Its benchmark results stay in the
  docs, labelled as measured on a model under a non-commercial research
  licence. **Qwen2.5-Coder-1.5B-Instruct** (Apache-2.0) is now the coding
  option and runs on the CPU. LFM2.5-1.2B is now the only GGUF model on the
  GPU by default.

- **Docs:** Series S / Series X wording swept across docs, templates and the
  social card: "Tested and measured on Series S. Not yet tested on Series X."
  (README first screen in #365; release-readiness texts in #366).

- **README first screen** rewritten for public launch: Developer Mode warning,
  "Tested and measured on Series S. Not yet tested on Series X.", LAN API
  marked off by default with no authentication, SSE streaming note, receipt
  links for every benchmark figure, the unreceipted 10 GB RAM and
  chars-per-token figures removed, and the Microsoft / Meta disclaimer.

- **Docs: install from a GitHub Release in the browser.**
  `docs/install-release.md` now starts with Device Portal steps that need
  no repo checkout or `xbox-env` file. The `deploy.sh` path remains as
  the command-line install. The browser steps follow Microsoft's
  documentation and have not yet been walked end to end on a console.
- Public docs no longer include a Partner Center reservation or Store ID.

- **Docs:** "Store catalogue" wording replaced with "signed model catalogue" /
  "built-in model list"; `/api/pull` described as a download from the built-in
  model list (endpoint unchanged); new _Model catalogue design limits_ section
  in CONTRIBUTING; licence and no-selling note in model-selection and
  using-the-app.

- **Licence copies for re-hosted `models-v1` builds.** SmolLM2, Qwen3.5-0.8B,
  and SD-Turbo now download a pinned upstream licence file next to the weights,
  plus a NOTICE sidecar (SD-Turbo's is the §IV.a line and the ONNX fp16
  derivative note). TAESD's MIT file travels with the optional VAE overlay.
  Texts live in `LICENSES/`; the matching `models-v1` assets are uploaded
  before merge.

- **Optional debug dashboard** (default off). A Settings toggle and/or
  `LocalState\debug-dashboard.flag` shows real resident-model, LAN, and
  inference readings above the prompt. Missing sources render as
  `unavailable`; the chat UI is unchanged while the view is off. Aggregation
  lives in `include/xllama/debug_dashboard.h` (host-tested). Original idea by
  [@maxmorrison74](https://github.com/maxmorrison74) (xllama-on-steroids fork);
  rebuilt in-tree without the rebrand, background, soundtrack, or footer credit.

- **`diskbw` compiles on macOS.** The buffered cache-drop hint used
  `posix_fadvise(POSIX_FADV_DONTNEED)`, which Darwin does not provide.
  Hosts without that feature (including macOS) now leave the page cache
  untouched, matching the existing Win32 no-op. Linux and UWP behaviour
  is unchanged. Thanks [@maxmorrison74](https://github.com/maxmorrison74).
- **`POST /v1/chat/completions` streams with `stream: true`.** Token pieces travel as
  `chat.completion.chunk` events over chunked `text/event-stream` (assistant role first,
  finish chunk with `usage`, then `data: [DONE]`). The text is raw: post-processing needs
  the whole reply, so it stays on the non-streaming path. A client disconnect aborts the
  generation, a failure after the head becomes an SSE error event, and stop sequences
  are withheld from the stream instead of reaching the client first.
- **GUI and LAN API load a model the same way** (audit F03, [ADR 0007](docs/adr/0007-session-spec.md)).
  The API ignored the catalogue `lora`, and the hub reused a session by model id,
  so the first surface to load a LoRA model decided whether its adapter applied.
  `session_spec.h` is now the one resolver (context, backend, GPU layers, adapter)
  for GUI, chat, embeddings and pull; the hub reuses the resident session only
  for the same load, and a pull of the resident model reloads it.
- **The LAN API frames requests exactly** (audit F02, [ADR 0006](docs/adr/0006-http-framing.md)).
  A negative `Content-Length` or a body shorter than declared used to return 200
  (and append a preference sample); a stalled body held a thread with no answer.
  Parsing moved to `include/xllama/http_framing.h` (host-tested): one decimal
  length, 413 above 8 MiB instead of a clamp, 400 for a short body, 408 past
  10 s for the head or 15 s for the body.
- **Tool proposals follow the template's grammar** ([ADR 0005](docs/adr/0005-tool-proposal-grammar.md)).
  The grammar `common/chat` already derives from the template's parser is now
  applied as llama-server does: lazy for `auto`, full for `required` or a named
  function, checked one token at a time with a full-vocabulary pass only on a
  rejection. `lfm25-1.2b-instruct` stops at the end of its call instead of
  narrating past it (which the parser rejected); speculative drafting is off
  while a grammar is active.

- **Generation numbers are validated before the model loads** (audit F01).
  `max_tokens` `-1`, `1.5` and `2147483647` used to return 200 (the negative one
  an empty reply), and an `INT_MAX` reply overflowed the shared budget
  arithmetic (UBSan). Token counts must be whole and below `n_ctx`; `temperature`,
  `top_p` and `seed` have ranges; image `steps`/`seed` clamp on the number before
  converting it. `budget_fits` is now the one 64-bit admission test for both
  trimmers, and `resolve_max_length` saturates instead of overflowing.
- **`usage.prompt_tokens` is the prompt's length** (#335). It reported the
  tokens prefilled, so a repeated prompt on a session that reused its KV prefix
  said `1`. `InferenceResult::n_prompt` carries the length; `n_p_eval` keeps
  meaning prefill work for tok/s.

- **The console shows what the LAN API is doing.** While the endpoint is on,
  a panel in the main header shows the base URL with the console's IP, the
  request being served (endpoint, model, elapsed), request and `busy` counts,
  the last request's time and tokens, and the last failure for 10 minutes.
  Before, an agent could drive the console for hours while the screen said
  "Ready" and named the chat model, not the one the API had loaded. A pad turn
  that waits for a LAN request now says so instead of stalling silently, and
  Settings → LAN API shows the full base URL. Text for the 10-foot view is
  larger (status 18pt, metrics 16pt) and the toolbar uses Segoe MDL2 icons
  instead of `[S]`/`[=]`/`[*]`. Counters: `include/xllama/api_activity.h`
  (header-only, host-tested).
- **History rows fit the dialog.** The 580 px list was wider than a
  ContentDialog's content area, so a long title wrapped and pushed its delete
  button past the edge. Rows are now a fixed-width grid (title on one line,
  then "N messages • date"), with the delete button always aligned; "1 msgs"
  became "1 message". Generation metrics sit on their own row above the
  toolbar instead of being truncated next to it.
- **Unknown models get `404`, not `500`.** A chat or embeddings request naming
  a model that is not on the console used to fail with `500` "session create
  failed" and an internal LocalState path; it now returns `404` "model '<id>'
  is not on this console; see GET /v1/models" without attempting a load.

- **Inference hangs are diagnosable and self-healing.** After the 2026-10-11
  LFM2-2.6B CPU re-measure hung with the log stuck at `prompt tokens: 298`,
  both backends now log `prefill starting` / `prefill done` (tokens + ms) and
  a periodic decode progress line (every 64 tokens or 15 s by default). An
  optional in-app watchdog aborts if a prefill chunk or decode token makes no
  progress for 180 s (comfortably above a normal run; 0 disables). Config:
  `--decode-progress-every`, `--decode-progress-ms`, `--infer-stall-timeout`,
  `XLLAMA_DECODE_PROGRESS_EVERY`, `XLLAMA_DECODE_PROGRESS_MS`,
  `XLLAMA_INFER_STALL_TIMEOUT_S`, and the matching `bench_*.txt` files.
  `bench-xbox-ort.sh` records a hung/timeout/stall row (`host` suffix
  `-timeout` / `-stall` / `-hung`) and exits non-zero instead of writing 2 of
  3 rows and returning OK. Harness wait: `--timeout` /
  `XLLAMA_BENCH_TIMEOUT_S` (default 300).

- **Docs: Series S remeasure on 1.6.0.1163**. Five published comparison rows
  now come from `bench/results/2026-10-11-remeasure-1163/` (3 recorded runs,
  median). New data points, not reproductions of the July rows. July figures
  stay as dated history in `docs/benchmarks.md`.

- **LAN chat accepts a client-side tool proposal** ([ADR 0004](docs/adr/0004-tool-proposals.md)).
  A non-empty `tools` or legacy `functions` schema is rendered with the GGUF
  Jinja template and may come back as `tool_calls`. xllama does not execute
  the tool. `tool_choice: none` and `tools: []` stay on the plain chat path.
  An ONNX model, a GGUF with no chat template, or a template that does not
  render the schema (phi4-mini) is a 400. The prompt budget
  for this path is `fit_tool_transcript`. `scripts/validate-api.sh tools` checks the contract on
  the console.

- **Docs: GPU decode in context**. A new section in `docs/benchmarks.md` sets
  the Series S and xllama beside known hardware (RX 6600, Steam Deck, Iris Xe
  and others, from the llama.cpp Vulkan scoreboard), measured as effective
  bandwidth over peak.
  - Decode: our kernels reach 63% of peak, end to end 23%; RDNA2 cards with
    llama.cpp Vulkan reach 74–86%.
  - Prefill: ~3–4% of the int8 peak (#313).

- **GGUF GPU path keeps Flash Attention** (#310). With GPU layers,
  llama.cpp's AUTO check disabled FA, because the KV cache stays on the CPU,
  so attention ran unfused, unlike on the CPU-only path. `apply_gguf_gpu_context`
  now requests FA explicitly. The q8-KV fallback restores the requested mode
  instead of AUTO.
  - Series S (CI 1.6.0.1163), default configuration:
    - Coder-3B decodes 26.6 tok/s (was 22.6), 1.90× the CPU;
    - LFM2.5-1.2B decodes 74.8 tok/s (was 65.4), 1.88×;
    - prefill is 1.65× / 2.35× the CPU, and peak RAM sits +88 / +77 MB over it;
    - H9 6/8 on both, and `validate-console.sh all` passes.

- **GGUF GPU decode D3: on by default for Qwen2.5-Coder-3B and LFM2.5-1.2B**
  (#228).
  - A catalogue entry can set `gpu_layers`; both models now run their
    matmuls on the d3d12 backend, and their display names say GPU.
    `resolve_gguf_gpu_layers` is the one decision point for the GUI and the
    LAN API; the operator file `gguf_gpu_layers.txt` still overrides it (`0`
    forces the CPU).
  - Series S (CI 1.6.0.1159):
    - Coder-3B at its catalogue `n_ctx` 4096 decodes 1.60×, peak +150 MB;
    - default-configuration `validate-console.sh all` passes;
    - H9 is unchanged.
  - Other GGUF models stay on the CPU.

- **GGUF GPU decode D2-r2 = PASS** (#228). Series S, CI 1.6.0.1156, opt-in
  backend `d3d12`, every layer on the GPU:
  - Coder-3B: decode 1.60×, prefill 1.27×, peak +159 / +125 MB;
  - LFM2.5-1.2B: decode 1.64×, prefill 1.82×, peak +163 / +156 MB;
  - H9 6/8 = 6/8 on both, and `validate-console.sh all` passes with the knob
    and without it.
  - A fix found on the way: zero-size ops (ubatches without output rows) are
    now no-ops on the backend. Before, the scheduler copied weights to the
    CPU, which timed out `validate genroom` and pushed Coder-3B to +220 MB.
  - The default stays CPU until the D3 per-model decision.

- **d3d12 GGUF layers: lower peak RAM** (#309).
  - A tied Q6_K `token_embd` now lives once, in `D3D12_Weights`: `GET_ROWS`
    runs on the GPU and placement goes through `tensor_buft_overrides`.
  - The GPU compute reserve is sized for one logits row (`n_outputs_max`).
  - Every GGUF load logs its working set per stage (`[xllama] ws …`).
  - Series S (CI 1.6.0.1148): peak with all layers on the GPU drops on
    Coder-3B from 2583 to 2203 MB (CPU 2044), and on LFM2.5-1.2B from 1048 to
    946 MB (CPU 783).
  - For D2-r2 and D3 the RAM criterion is now a product one, decided before
    the run: peak ≤ 3584 MB and ≤ 200 MB over the CPU run.
  - The output cap follows what the context requests: 1 row for
    prefill/decode, 1 + draft with prompt lookup, llama.cpp's default for
    embeddings. A blanket cap of 1 aborted those batches; an independent
    review caught it before merge.
  - The selftest times a median of 5 runs and covers GET_ROWS.

- **d3d12 GGUF layers: q8 activations** (#312). The backend now quantizes each
  matmul input on the CPU with ggml's own `from_float` (q8_0 / q8_K), exactly
  as the CPU backend does, and the kernels sum integers (`dot4add_i8packed`,
  cs_6_4). Gate A on Series S (CI 1.6.0.1144): the selftest is within ≤ 2.9e-7
  of the CPU backend, decode shapes run at 109–164 GB/s, and H9 is 6/8 = 6/8
  on Coder-3B and LFM2.5-1.2B (D2: Coder-3B 5/8).

- **GGUF GPU decode D2b: opt-in d3d12 layers** (#228). `src/bridge/llama_gpu.h`
  applies a GPU-layer request in one place for the bench, the CLI
  (`--gpu-layers`), GUI sessions and the LAN API
  (`LocalState\gguf_gpu_layers.txt`, default off). KV and attention stay on
  the CPU. The device's host buffer type holds the CPU compute buffer, and
  persistent CPU threadpools serve the split graph. Bench knobs:
  `--gpu-layers` and `--ignore-eog` (exactly `n_predict` tokens).
  - Series S (CI 1.6.0.1138): decode **1.59×** on Coder-3B and **1.58×** on
    LFM2.5-1.2B; prefill 1.32× / 1.74×; `validate-console.sh all` PASS.
  - The predeclared D2 gate **fails** on peak RAM (+539 / +265 MB) and on
    Coder-3B H9 (5/8 vs 6/8), so GGUF decode stays on the CPU by default.
- Fix: dropped a loader patch that wrote repacked CPU tensors in chunks.
  ggml's repack buffer requires whole-tensor writes, so the app aborted when
  switching to a GGUF model with a repacked tensor over 8 MiB (development
  packages only, never released).
- `scripts/check-win-syntax.sh` now defines `XLLAMA_USE_LLAMA` and adds the
  llama.cpp include dirs. The GGUF bridge files used to pass without being
  compiled.

- **GGUF GPU decode D2a: ggml backend `d3d12`** (#228). A GPU-type ggml device
  registered at runtime (`include/xllama/ggml_d3d12.h`,
  `src/bridge/ggml_d3d12.cpp`): MUL_MAT with Q4_0 / Q4_K / Q6_K weights in a
  DEFAULT-heap buffer, activations in a CPU-visible default buffer, everything
  else on the CPU. Kernels `shaders/ggml_d3d12_mmv_*.hlsl`; host tests emulate
  them against ggml's dequantizers; console selftest `d3d12be.flag`
  (`scripts/bench-d3d12-selftest.sh`). Series S: **D2a PASS** on the third run
  (CI package 1.6.0.1125) — 12/12 correct, every decode shape ≥ 102 GB/s with
  the kernel width picked by K; runs 1–2 failed the speed gate and stay
  recorded. Wired into sessions by D2b (entry above).

- **GGUF GPU decode design + probe D1** (#228). Design SSOT
  [`docs/gguf-gpu-decode.md`](docs/gguf-gpu-decode.md): a ggml backend `d3d12`
  owned by xllama (matmul-only first, opt-in), cost model and predeclared D1/D2
  gates. New `gpustep.flag` / `gpustep-inproc.flag` probe (round trip,
  simulated token per model, heap bandwidth, in-XAML run),
  `scripts/bench-gpustep.sh`, `xllama-cli --gpustep` / `--gpustep-verdict`.
  D3D12 plumbing shared in `src/bridge/d3d12_compute.{h,cpp}`. Series S (CI
  package 1.6.0.1117): **D1 = `D2-matmul-only`** — round trip 49.7 µs,
  simulated Coder-3B token 21.16 ms with a CPU sync per split, in-XAML PASS,
  weights belong in DEFAULT heaps. CSV `bench/results/phase15-gpustep-d1.csv`.
  No backend, no tok/s claim. `bench-gpugemv.sh` now deletes a stale `.done`
  before the run.

- **H6.3 multi-row Q4_K GEMV density probe — K3** (#228). New `rows`
  (cs_6_0) and `dot4` (cs_6_4, q8 activations) kernels in `gpugemv.flag`:
  4 rows × 64 threads per group, X in registers, no LDS transpose. Series S
  CI package 1.6.0.1113: `rows` median **143.06 GB/s packed**, `dot4` 136.94,
  G1 PASS 3/3; `wave32` reproduces H6.2 at 25.12. Measure-only: no Session
  GPU backend, no tok/s claim. CSV `bench/results/phase15-gpugemv-h63.csv`.

- **FloppyLM experiment (docs only).** Extreme-compression Tiny GPT
  (`floppy_4mb` class) is not an xllama engine. SSOT
  [`docs/floppylm.md`](docs/floppylm.md); checklist in `ROADMAP.md`.
  [PR #301](https://github.com/gianlucamazza/xllama/pull/301) closed without
  merge on 2026-10-01; this repository keeps no FloppyLM engine. No quality
  or speed claim.
- **llama.cpp `b29c606` → `7fe450e`** (#295, upstream 0.5.0). Host: greedy
  output and final-prefill logits of QAD `lfm25-350m` and `lfm25-1.2b-instruct`
  are bit-identical to the old pin; 264/264 doctest cases. Series S (CI package
  1.6.0.1088): `gguf`, `longchat` and `kvsnap` gates PASS; 1.2B decode 40.03
  vs 40.07 and prefill 109.4 vs 109.6 tok/s on 1.6.0.1072 (sequential A/B, same
  config; `bench/results/2026-09-30-llama-7fe450e.csv`). No headline change.
  The pin adds the `hrm_text` arch (DFM Mimir), which the arch-watch (#275)
  evaluates separately.
- **WS-F microphone PASS** (#241). With an Xbox Wireless headset attached,
  `scripts/probe-mic.sh` captured real audio inside the AppContainer (3 s,
  48 kHz stereo, RMS 0.019). The ASR surface is unblocked; no model ships yet.
  Evidence `bench/results/2026-09-30-mic.json`, `docs/uwp-constraints.md` §10d.
- **Series S catalogue gates** (2026-09-30, CI package 1.6.0.1072,
  `bench/results/2026-09-30-catalogue-gates*`). Bars were fixed before the run.
  - **Phi-4-mini** (`phi4-mini`, #268) joins the catalogue:
    [unsloth/Phi-4-mini-instruct-GGUF](https://huggingface.co/unsloth/Phi-4-mini-instruct-GGUF)
    Q4_K_M (MIT, ~2.49 GB, HF only), Phi-3 template. H9 6/8 against
    `llama32-3b`'s 5/8, 11.2 tok/s decode, 2765 MB peak.
  - **LFM2.5 QAD Q4_0** (#270): `lfm25-230m` (132.4 tok/s, 236 MB, H9 1/8),
    `lfm25-350m` (101.6 tok/s, 311 MB, H9 4/8) and `lfm25-1.2b-instruct`
    (40.0 tok/s, 783 MB, H9 6/8) now pin Liquid
    [QAD Q4_0](https://www.liquid.ai/blog/qad), with no H9 regression and
    lower peak. `lfm2-2.6b` is unchanged (no QAD published).
- **H9 `grounded_qa` scorer fix** (#243). The task now rejects distributive
  readings ("each", "ciascuno", "per core", …). One full Series S re-run
  (`bench/results/2026-09-30-h9-rescore.jsonl`) is the new H9 source. Only
  `lfm25-230m` Q4_K_M changed (2/8 → 1/8), which ties its QAD Q4_0 and
  returns the 230M to QAD.
  - **Rejected, measured**: MiniCPM5-1B (#267, H9 3/8 against a 6/8 bar) and
    Gemma-3-1B (#269, dominated by `lfm25-1.2b-instruct`) were catalogued for
    the trial and removed again. Neither id ships.
- **Positioning** (#280). Public docs lead with the local-inference stack
  (games + assistants). Chat UI is the demo shell. Honest limits:
  [docs/positioning.md](docs/positioning.md).
- **Main branch governance** (#264, ADR 0002). Ruleset: PR required, four CI
  checks strict, squash-only merges.

## [1.6.0.0] - 2026-09-28

Local embeddings and catalogue-backed Ollama pulls, with one resident session
and shared GUI/API model-write ownership. The unified CI MSVC package is the
Dev Mode release path; upgrades from the current identity preserve LocalState.

- Update llama.cpp to `b29c606` and repair the AppContainer patch and MSVC source
  inventory for the updated dependency.

- Add GGUF embedding inference through the shared resident session and opt-in LAN API:
  Ollama `/api/embed`, legacy `/api/embeddings`, and OpenAI `/v1/embeddings` with float
  or base64 vectors. BGE-M3 Q8_0 and Nomic v2 MoE Q8_0 passed the host 3584 MiB
  memory gate and are catalogued; Qwen3-Embedding-4B Q4_K_M measured 4411 MiB and is
  not catalogued. Series S protocol/memory validation now passes; see the
  [catalogue status and evidence](docs/model-matrix.md#g-embedding-api-catalogue-host-and-series-s-validated). Dedicated embedding throughput and retrieval-quality benchmarks remain open.
- Add catalogue-only Ollama `POST /api/pull` with NDJSON download progress, SHA-256
  verification, serialized pulls, and a load into the shared resident session on success.
- Serialize GUI/API network writes, USB import and rollback with the accepted
  [ADR 0001](docs/adr/0001-model-download-writer.md) atomic RAII permit. Rejected
  writers leave files/markers untouched; preserve adapters and release before
  completion callbacks. Settings and autopilot share model provisioning policy.
- Validate divergent hybrid KV restoration against a cold reference; retain safe
  full-prefill when tail rewind is unsupported and control append-only reuse
  separately. Source-matched console and LAN suites pass; the main CI package
  1.5.6.1041 also passes the shipping API/embedding smoke. These pre-release results are bound to their recorded source/package;
  the exact tagged package gate is recorded in the GitHub release notes.
  Microsoft Store publication remains separate.
- Keep embedding batches serial on the single-slot llama.cpp context; preserve model
  pooling metadata and return normalized vectors with optional dimensions. Fix truncation
  direction for LAST pooling (keep suffix not prefix), add EOS token append when GGUF
  requests it, reject invalid Matryoshka dimensions before inference, and filter embedding
  models from chat UI and POST /v1/chat/completions (400). Ollama library name aliases
  (`bge-m3`, `nomic-embed-text-v2-moe`, `qwen3-embedding:4b`) map to catalogue ids.

- **Research package and XAB baseline.** Added a citable Series S Dev Mode
  report with claim-level provenance, generated evidence tables/figures and a
  release runbook. Added `scripts/run-xab.sh` to compose text, KV, H9 and
  diffusion harnesses. Historical measurements without thermal/power sidecars
  remain explicitly non-production evidence.
- **Demo presentation edit.** Added a reproducible 30-fps hard-cut presentation
  edit for concise README walkthroughs. The raw Device Portal recording and
  its measured capture rate remain the only performance evidence.
- **Catalogue signing and reproducibility evidence.** Store CI now signs the
  bundled model catalogue with the GitHub Actions-held RSA key and the Store
  verifier imports the pinned CNG public-key blob explicitly. The Xbox Series S
  Dev capture in `bench/results/phase17-console-2026-08-26.csv` records three
  post-warm-up runs with model hash, context, TTFT, throughput and peak memory;
  external power and thermal-equilibrium evidence remain separate gates.
- **CI benchmark validation fix.** Closed the CSV-validation loop in
  `build-linux.yml` so the shell `for`/`if` block is syntactically complete and
  all v2 benchmark rows are validated in Linux CI.

## [1.5.6.0] - 2026-08-22

Phase 16 catalogue win plus the pin and probes that landed with it, plus the
SDK transition (SessionHub factory, probe optionals, RoutingPolicy configurability,
Doxygen docs). Product ship path remains **CI MSVC**. Suite is still **10**
console gates.

**Upgrading from 1.5.x is a normal in-place update** (same package identity
`GianlucaMazza.xllama`).

### SDK

- **Modular SDK documentation.** Replaced the monolithic 594-line `docs/SDK.md`
  with 11 focused `.md` files under `docs/sdk/` (`overview`, `session`,
  `inference`, `routing`, `training`, `chat`, `kvstore`, `path`, `platform`,
  `sampling`, `misc`) — 742 lines total, each covering a single public-header
  group.

- **SessionHub factory pattern.** `SessionHub` now has an explicit constructor
  (`SessionHub::ensure_locked` / `reset_locked`) while keeping the
  backward-compatible `session_hub()` singleton. Callers that need explicit
  lifecycle can construct directly; the rest keeps working unchanged.

- **Probe optionals.** `CMakeLists.txt` now gates all probe sources under
  `option(XLLAMA_BUILD_PROBES "Build benchmarking probes" ON)`. CLI `--membw`,
  `--diskbw`, `--ramceil`, `--gpubw`, `--gpugemv` and the headless `.flag`
  files are absent when built with `-DXLLAMA_BUILD_PROBES=OFF`.

- **RoutingPolicy configurability.** All five routing gates (`dml_text_model_ok`,
  `decide`, `allow_kind`, `reuse_kv_kind`, `reuse_kv_model`) are now
  `std::function` callbacks on `RoutingPolicy`, with member wrappers dispatching
  through them. `default_policy()` returns a fully-configurable instance. Free
  functions (`dml_text_model_ok()`, `decide_routing()`, etc.) wrap the
  lazy-static default for backward compat.

- **Doxygen comments.** All 34 public headers in `include/xllama/` now carry
  `///` Doxygen documentation on every struct, class, method, and field.
  `InferenceParams` / `InferenceResult` fields rewritten to fix duplicate
  `abort_flag` declaration and add `///` on every member.

### Added

- **H6.2 wave32 Q4_K GEMV density probe** (`gpugemv.flag`,
  `shaders/gpugemv_q4k_wave32.hlsl`). Measure-only, not a Session GPU backend.
  Series S CI MSVC `1.5.5.922`: G1 PASS, `wave32` median **25.4 GB/s packed**
  (retimed naive 1.96). **K2 park** — G2 stays 40. CSV
  `bench/results/phase15-gpugemv-h62.csv`.
- **Store D1 App vs Game spike** on Series S (`lfm25-350m`, same package).
  App GPU budget **691 vs 3801 MB**; long-gen decode ~87 vs ~95 tok/s; peak
  320 MB both. Listing should request **Game** metadata. CSV
  `bench/results/store-app-vs-game-2026-08-21.csv`.
- **`validate-console.sh store`** — Store SKU smoke (catalogue download + GGUF
  chat + `set_api` reject). Not one of the 10 Dev Mode hardware gates. **PASS**
  2026-08-21 on CI Store SKU `1.5.5.928` (Game); Dev SKU restored after.
- **Store Phase 2 pack:** EN listing copy, age notes, and operator
  steps (`docs/release-readiness.md`); GitHub Pages privacy URL
  `https://gianlucamazza.github.io/xllama/privacy.html`; Settings link + Issues
  template to report inappropriate generated content (Store policy 11.16).
  The Partner Center reservation and Store ID were removed from public docs. Dev Mode identity unchanged. Public description and age / generative-AI notes in `docs/release-readiness.md` §9/§11.
- **`include/xllama/json_utils.h`** — header-only JSON string helpers:
  `json_escape()` (canonical) and `json_read_string()` with full `\uXXXX`
  decode (including surrogate pairs → UTF-8). Replaces 7 duplicated
  implementations across the tree.
- **`src/bridge/decode_loop_ort.h`** — consolidated ORT GenAI decode loop.
  Mirrors the structure of `src/bridge/decode_loop.h` (llama path).
- **`src/bridge/ort_common.h`** — shared ORT setup helpers (SEH translator,
  log callback registration).
- **CI ASan lane** (`build-linux.yml` job `asan`): workflow_dispatch-only.
  Builds with the `linux-asan` CMake preset (Debug + ASan + UBSan) and runs
  the full test suite under sanitizers.
- **`tests/test_json_utils.cpp`** — 11 test cases covering json_escape
  (quotes, backslash, control chars → `\uXXXX`, mixed content),
  json_read_string (all escapes, `\uXXXX`, surrogate pairs, lone surrogates,
  unterminated strings, unknown escapes lenient), and round-trip
  (escape → read_string → original).
- **`include/xllama/cancel_policy.h`** — `CancelTarget` enum +
  `cancel_target()` precedence function (image > training > text). The generic
  "a job is running" flag is set by all three, so the specific flags are the
  only ones that identify the job. Exhaustively tested on the host
  (`tests/test_cancel_policy.cpp`, all eight combinations of the three flags).
- **Autopilot op `mark`** — a rendez-vous for screenshot capture. The app writes
  a label to `LocalState\autopilot-mark.txt` and blocks; the host polls for the
  file, takes its Device Portal screenshot, and deletes the file to release the
  script.
- **Autopilot op `show_pane`, and a guard against the crash designing it
  exposed.** Settings, History and the image viewer are all `ContentDialog`s and
  none was reachable from the autopilot. The guard is product code: nothing
  checked whether a dialog was already open, and a second one throws inside a
  `fire_and_forget`, whose `unhandled_exception()` calls `std::terminate()`.
  `AutopilotAction` gained a dedicated `label` field. Ops 15 → 17.

### Fixed

- **json_escape triplicated across 7 TUs with divergent behavior.** Consolidated
  into `include/xllama/json_utils.h` (single canonical implementation, full
  `\uXXXX` control-char escaping). All callers now use `xllama::json_escape`.
  The training job parser (`src/bridge/training.cpp`) was also upgraded to
  decode `\uXXXX` (including surrogate pairs → UTF-8) via the new
  `json_read_string` helper — previously `\u0041` was passed through verbatim
  as `u0041`, a correctness gap for non-ASCII dataset paths/names.
  `device_train.cpp`'s `result.json` writer (which passed control chars raw,
  producing invalid JSON) is now fixed. Full round-trip tested
  (`tests/test_json_utils.cpp`, 11 cases).

- **ORT stateless path silently ignored stop_sequences.**
  `run_inference_ort` (inference.cpp) never checked `params.stop_sequences`,
  while `OrtSession::run_decode` did. Both paths now share the consolidated
  `decode_loop_ort.h` loop which applies stop sequences after each iteration
  (same semantics as llama.cpp: the stop-triggering token IS counted).

- **ORT decode loop duplicated** between `run_inference_ort` (inference.cpp)
  and `OrtSession::run_decode` (session.cpp) — the exact pattern the project
  had already fixed on the llama side. Extracted into
  `src/bridge/decode_loop_ort.h` (header-only, same structure as
  `decode_loop.h` for llama).

- **SEH translator and OgaSetLogCallback duplicated** between inference.cpp
  and session.cpp. Extracted into `src/bridge/ort_common.h`
  (`install_se_translator()`, `register_oga_logging()`).

- **The B button no longer drops out of the app.** On Xbox an unhandled
  `BackRequested` is the shell's cue to suspend and return to Home, and the
  handler only marked the event handled while an inference was running — so a
  B press on an idle chat closed xllama with no warning. Completed turns are
  already on disk, so this cost the typed-but-unsent prompt, the KV cache and a
  full model reload, not the conversation. Every `BackRequested` is now
  handled: B cancels a running reply and otherwise does nothing; leaving the
  app stays on the Xbox (Guide) button (`uwp/MainPage.cpp`,
  `docs/uwp-constraints.md` §10e, `docs/using-the-app.md`).

## [1.5.5.0] - 2026-08-20

Phase 16 catalogue win plus the pin and probes that landed with it. Product
ship path remains **CI MSVC**. Suite is still **10** console gates.

**Upgrading from 1.5.x is a normal in-place update** (same package identity
`GianlucaMazza.xllama`).

### Added

- **Phase 16 model-scouting campaign** (`docs/phase16-model-scouting.md`) and
  its first shipped result: catalogue entry **`lfm25-230m`** (LFM2.5-230M
  Q4_K_M, direct from LiquidAI so the LFM Open License travels with the
  weights). Console-measured on Series S at **119.2 tok/s decode / 241 MB peak
  / H9 2/8** — it becomes the **floor** tier, displacing `gemma3-270m` at
  1.55× its decode and 127 MB less peak, at one H9 task below it. The
  first-launch default stays `lfm25-350m`, which is slower and heavier but
  scores 4/8: the default trades throughput for capability deliberately.
  Evidence `bench/results/phase16-gguf.csv`, `bench/results/phase7-h9.jsonl`.
- **`gemma3-270m` H9 measured for the first time (3/8).** Its `model-matrix`
  §A1 cell had carried `—` since the model was catalogued, which is what made
  the floor-tier comparison uncomputable until now.
- **MiniCPM5 chat renderer** (`model_is_minicpm5`: template `<s>` BOS +
  no-think prefill). Host T1 passes both halves; T3 console bench was not
  booked this campaign, so `minicpm5-1b` is **not** in the catalogue.
- **`diskbw` NVMe probe** (`diskbw.flag` / `scripts/bench-diskbw.sh`). Series S
  unbuffered read ~2.0 GB/s sequential / 1.55–1.76 GB/s random 2 MiB. SSD
  streaming is real, narrow, and not a product path
  (`docs/ssd-inference-assessment.md`).
- **WS-F microphone probe** (`mic.flag` / `scripts/probe-mic.sh`).
  `AudioGraph` opens under AppContainer; `AccessDenied` did not fire; no
  headset was attached (`DeviceNotAvailable`). Not a verdict — #241 stays
  open. `docs/uwp-constraints.md` §10d.
- **Markdown formatting is gated in CI**, pinned to `prettier@3.9.6` alongside
  the existing `clang-format==22.1.5`, over every tracked `*.md`
  (`.prettierrc`, `.prettierignore`). The repo had no Markdown formatter and
  had drifted; the whole corpus was formatted once in the same change.

### Changed

- **`llama.cpp` pin `6d5a910` → `0865990` (b10333, #244)** with UWP glue
  (`llama-kv-cache-msa.cpp`, `llama_model_params.load_mode`, sampler
  `n_vocab` on penalties). Phase 16 WS-B had closed without a bump — both
  desk `arch:not-in-pin` flags were refuted at the GGUF header — and this
  Dependabot bump is independent of that kill.
- **H5 BitNet/low-bit survey closed NO-GO** (2026-08-10). The pin carries the
  `bitnet` arch; no sub-4B model trained at ≤2 bits publishes weights.
- **Phase 16 WS-E (embeddings) closed** — S-gate FAIL, no named consumer.
  Capability exists in the pin (`nomic-bert`); the product has no surface.
  Tracked as #242.

### Fixed

- **Documentation drift found by the Phase 16 audit**: `model-matrix` §D
  claimed `H2 open` while §A3 of the same file recorded H2 FAIL; its "Last
  updated" stamp predated its own newest content; §G still listed a shipped
  MSIX as an open gap; `benchmarks.md` cited a stale `llama.cpp` pin;
  `recommended-config.md` enumerated a four-gate console suite that has been
  ten gates since v1.5.4.0; and the Phase 16 campaign doc had wrongly listed
  H5 (BitNet) as closed when it is an open desk survey.
- **`deploy.sh` stop/start** actually stop and start; an unreachable console
  and a missing CSRF token now fail instead of claiming success.
- **Wine 11.15**: `ensure-cppwinrt-pin.sh` prefers the native cppwinrt at the
  pin. `quantize.sh` pointed at a tree we never build.
- **CI**: `check-coherence.py` asks git which Markdown we own; shellcheck is
  pinned; the clang-cl ggml patch is versioned instead of carried untracked.
