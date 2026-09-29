# Console validation runbook

Current on-device release gates for Xbox Series S Dev Mode. Historical experiment
procedures and measured results live in `CHANGELOG.md` and `bench/results/`.

## Prerequisites

- Install the latest `xllama-appx` unified artifact.
- Set Device Portal credentials in `~/.config/xllama/xbox-env`.
- Confirm the package is designated **Game** in Dev Home.
- Provision models after any uninstall, because uninstall removes `LocalState`.

```bash
source ~/.config/xllama/xbox-env
./scripts/provision-models.sh --all-test
```

**After installing a NEW package identity** (first install, or an identity
rename like the 1.5.0.0 VenereLabs → GianlucaMazza migration): launch the app
once (`./scripts/deploy.sh start-app`) before provisioning. The `LocalState`
folder does not exist until first launch, and every WDP file upload fails with
`"File move failed" / "The system cannot find the path specified"` until it
does (hit during the 2026-07-25 migration).

For a controlled trial, set `XLLAMA_EXPECTED_PFN` to the installed package full
name verified against the selected CI run and commit. Console/API suites reject
a different installed package before writing test files; the benchmark runner
also checks it before every run. Keep the expected value fixed until the next
intentional deployment.

## Official automated suite

```bash
./scripts/validate-console.sh all
```

The suite drives the live UI through autopilot and fails unless all current
hardware gates pass:

- **routing** — auto A/B with the parity-validated `-v2` DML asset: the long
  (>1550 tok) turn routes to GPU, short turns to CPU (#91 lifted for that asset);
- **settings** — the `set_routing` / `set_sampling` / `set_kv_reuse` /
  `set_taesd` / `set_system_prompt` autopilot ops are dispatched and every
  resulting value is asserted against the persisted `settings.json`. The
  baseline is seeded with the opposite of each target, so an op that silently
  does nothing fails rather than inheriting a value that already matched (needs
  `smollm2-360m-cpu-int4` in LocalState; `set_taesd` / `set_system_prompt` need
  an app build >= 1.4.0.632, the rest >= 1.4.0.606);
- **GGUF chat** — the default LFM model loads through llama.cpp and generates,
  **and the conversation is read back from `chats/index.json` with a non-empty
  title**. That second half was added after the gate passed for months over a
  real defect: `NewChat()` assigns the id, so the branch that also derived the
  title could never run, and every saved conversation had `"title":""` (14 of 14
  on the console). The turn generated correctly and the log said so — what was
  wrong was the record on disk, which no gate had ever looked at;
- **longchat** (#169) — a chat whose history exceeds the trimmer ceiling is
  injected, then run with KV reuse. Trimmed rounds must stay in the reuse regime
  (prefill = the delta, not the ~1800 tokens of the whole history), and when the
  resident KV overflows `n_ctx` the session must front-drop-evict and carry on.
  Three log assertions, and two of them are negative: a `context shift — evicted`
  line must appear, and neither `retrying with full prefill` nor `context full`
  may. A continuation that silently falls back to a full re-prefill still answers
  correctly, which is why the gate reads the log rather than the reply;
- **kvsnap** (#170b) — leave a conversation, come back, and the history must not
  be re-read: the snapshot written on the way out is restored on the way back and
  the #170a prefix diff turns it into a delta. Beyond the two log lines
  (`KV state saved`, `KV snapshot restored`) the gate compares **prompt-token
  counts**: the returning turn must prefill under a quarter of the cold turn. A
  snapshot that is written, restored and then ignored passes both log checks and
  fails this one. Seed replies are one-token greedy replies to keep the saved
  prefix append-only. This measures the supported optimization, not hybrid tail
  rewind: completed/re-rendered replies can diverge inside the saved tail, where
  LFM must safely full-prefill. `validate-kv-fallback.py` forces that divergence
  and compares its output and token count with a cold session. The injected history is deliberately kept **under** the trimmer
  budget, so a trim cannot muddy the signal with #169's shift;
- **coderpaste** (#193) — a long paste on a coding session (`n_ctx` 4096), in two
  regimes and two runs, on `qwen25-coder-0.5b`. **A**: past the 2048 logical
  batch but inside `n_ctx` → the prefill must chunk and answer normally.
  **B**: past `n_ctx` → the app must refuse with `prompt too long` and stay
  alive. What this pins is an abort, not an error: `llama_decode` asserts
  `n_tokens <= n_batch` with `GGML_ABORT` in Release too, so the old failure
  killed the process. A dead app never writes `autopilot-done.txt`, so the
  missing marker is itself the assertion;
- **thinkdone** (#223) — a short prompt on `lfm25-1.2b-thinking` at **n_predict
  1024** (catalogue default) completes with a non-empty answer after strip —
  the happy path that `thinkcut` deliberately does not cover.
- **thinkcut** (#193) — a thinking model (`lfm25-1.2b-thinking`) that spends its
  whole budget reasoning postprocesses to an empty answer, and the turn used to
  vanish: nothing saved, the streamed chain of thought orphaned on screen, status
  `Done`. The gate asserts both halves — `postprocess left no answer` in the log,
  **and** the conversation on disk still ending in an assistant turn carrying the
  explicit `reasoning only` stand-in. If the reasoning block is never truncated
  the gate fails on purpose: either `n_predict` is too generous or the model
  emits no `<think>` at all, which would make the whole strip a no-op. Investigate
  it; do not relax it;
- **genroom** (#193) — a full context must still leave room for the whole reply.
  The trimmer ceiling used to reserve a flat 250 tokens while the UI default
  `n_predict` is 512, and the generation loop clamps `n_predict` to what the
  context has left, so a prompt on the old ceiling got a reply cut at ~248 tokens
  in silence. The gate asserts the **arithmetic** — `prefill + n_predict <=
n_ctx` — rather than the answer's length, which keeps the verdict independent
  of whether a small model feels talkative. Two preconditions guard against a
  vacuous pass: the `prompt budget:` line must be present (or the exact fit never
  ran) and something must actually have been dropped (or the payload never filled
  the context);
- **TAESD** — image generation completes through DirectML with the fast VAE.
  This gate swaps `vae_decoder/model.onnx` from a local cache rather than
  flipping the in-app toggle: the toggle makes the console download the asset
  from the models-v1 release, which would make the gate slower and dependent on
  console-side network. The toggle's own writer is covered by the `settings`
  gate instead;
- **API** — separate LAN gate (`scripts/validate-api.sh`): `spike|chat|prefs|train|all`
  (chat + #118 preference/training-status probes). Not wired into
  `validate-console.sh all` today — run it from another host on the LAN.

### When a gate fails, look at the screen

Verdicts come from grepping `xllama.log`, which cannot describe a failure that
never reached the log — the app dying at launch behind a system dialog being the
case that cost the most time (`dml-metacommands-runbook.md`). So every run keeps
the last two Device Portal screenshots, and a failing gate writes them out:

```text
  Screenshots of the failing run: /tmp/xllama-gate-shots/longchat-*.png
```

`-1.png` is the later frame. What they are worth depends on how the gate failed,
and the first console run of this feature corrected the original claim:

- **autopilot `error:`** — `ApRun` writes its marker _without_ exiting, so the
  app is still on screen showing the broken state. This is what the frames are
  for, and they deliver it.
- **timeout** — the app is alive and most likely stuck. Same.
- **marker `ok`, log grep rejects** — every gate script ends with `quit`, and
  that path does exit. Keeping two frames was meant to cover this; measured, it
  often does not. Polling is every 10 s and a gate can finish in ~20 s, so both
  frames can land after the app is gone — observed, both showing Dev Home. **A
  Dev Home frame is not a UI fault**; for this class the log is the evidence.

Frames are taken at every poll, except during `taesd` — the one gate that
asserts a duration (VAE decode under 1000 ms), and a screenshot is GPU work on
the same SoC. That is a list, not a sampling rate, because sampling less often
everywhere would only make the collision rarer while halving the evidence for
the nine gates that time nothing. `taesd` still gets its end-of-run frame, and
loses nothing it needed: its image-generation failure takes the autopilot
`error:` path where that frame is the right one, and its `vae_ms` failure is a
number already in the log that no screenshot improves.

`XLLAMA_GATE_SHOTS=0` disables capture entirely; `XLLAMA_GATE_SHOTS_DIR` chooses
where the frames land.

Run an individual gate while debugging:

```bash
./scripts/validate-console.sh routing
./scripts/validate-console.sh settings
./scripts/validate-console.sh gguf
./scripts/validate-console.sh longchat
./scripts/validate-console.sh kvsnap
./scripts/validate-console.sh coderpaste
./scripts/validate-console.sh thinkcut
./scripts/validate-console.sh thinkdone
./scripts/validate-console.sh genroom
./scripts/validate-console.sh taesd
./scripts/validate-api.sh all          # spike + chat + prefs + train
./scripts/validate-console-training.sh rate   # preference UI path
```

## Store SKU smoke (not in `all`)

Retail SKU: no LAN, no USB, no headless flags. Autopilot remains. Procedure and
restore: `docs/store-readiness.md` §10.

```bash
./scripts/validate-console.sh store
```

Asserts (1) first-run catalogue download of `lfm25-350m` or an already-provisioned
line after `install-latest-build.sh --store`, (2) GGUF chat saved with a title,
(3) `set_api` is rejected with `LAN API not available in Store SKU`.

## Benchmark evidence

Use `scripts/bench-xbox-ort.sh` and the fixed prompts described in
[`bench/README.md`](../bench/README.md). A comparison row must come from one
atomic CSV record; never combine the best prefill, decode and RAM values from
different runs.

**Preflight — verify the on-device config is pristine.** `--threads` swaps
`models\<name>\genai_config.json` on the device and restores it on exit, but a
crashed or pre-restore-era sweep can leave the swap in place (observed twice: a
t8 residue, and a t4 residue found 2026-07-25 — the shipped CPU asset sets no
`intra_op_num_threads` at all). Before any measurement session:

```bash
./scripts/deploy.sh fetch-file "$(./scripts/deploy.sh pfn)" genai_config.json \
    /tmp/cfg.json 'models\smollm2-360m-cpu-int4'
grep intra_op_num_threads /tmp/cfg.json && echo "DRIFT: restore the pristine config first"
```

After adding or changing committed evidence:

```bash
python3 scripts/generate-benchmark-summary.py
python3 scripts/generate-benchmark-summary.py --check
```

Record raw CSV/JSONL output under `bench/results/`, update the appropriate
research verdict or changelog entry, and let the generated summary own the
comparison table.

## Campaign T3 — MiniCPM5-1B (#267)

Host T1 and the H16.1d renderer already shipped. Series S T3 needs the CI MSVC
unified package (`xllama-appx`) and `xbox-env`. The catalogue id is
`minicpm5-1b` (official openbmb Q4_K_M). Do not treat host tok/s as a Series S
result.

```bash
source ~/.config/xllama/xbox-env
./scripts/install-latest-build.sh --provision   # clean LocalState, no stale GGUFs
./scripts/provision-models.sh minicpm5-1b lfm25-1.2b-instruct
export XLLAMA_EXPECTED_PFN="$(./scripts/deploy.sh pfn)"
./scripts/bench-xbox-ort.sh minicpm5-1b --runs 4 --n-predict 96 \
  --out bench/results/<date>-catalogue-gates.csv
# H9 goes through the LAN API: bench mode never binds the port.
./scripts/validate-api.sh spike
./scripts/eval-xbox-models.sh --models minicpm5-1b,lfm25-1.2b-instruct \
  --out bench/results/<date>-catalogue-gates-h9.jsonl
```

`eval-xbox-models.sh` truncates `--out` before writing, so never point it at
`phase7-h9.jsonl`. Only that file feeds the published H9 column; new JSONL
files are raw evidence until `bench/benchmark-summary.json` says otherwise.

Campaign PASS bars (H16.1d, [phase16-model-scouting.md](phase16-model-scouting.md)):
H9 ≥ 6/8, median decode ≥ 34.1 tok/s (0.9× the `lfm25-1.2b-instruct` 37.9),
`peak_ws_mb` ≤ 811, and **zero** `<think>` / `</think>` in visible H9 output
(any leaked chain of thought fails the card).

Record the package next to the CSV as a JSON sidecar with `run_id`,
`head_sha`, `sha256` and `expected_pfn` (precedent:
`bench/results/2026-09-28-console-api-validation.json`). Then update
[model-matrix.md](model-matrix.md) §A4 / §F: PASS moves the row to §A1 with
the measured values; FAIL becomes `reject — measured` in §F with the numbers.
Regenerate `docs/benchmarks.md` only after the console CSV is committed.

LAN pull smoke (optional, while the API is up):

```bash
curl -N -X POST "http://${XBOX_IP}:11434/api/pull" \
  -H 'Content-Type: application/json' \
  -d '{"model":"minicpm5-1b"}'
```

## QAD ripin regression — LFM2.5 (#270)

Catalogue ids `lfm25-230m`, `lfm25-350m`, and `lfm25-1.2b-instruct` now pin
Liquid QAD Q4_0 GGUFs (filenames changed). `lfm2-2.6b` is unchanged (LFM2; no
QAD published). Do not treat host tok/s as a Series S result.

`provision-models.sh --force` uploads the new file but leaves the old
`*-Q4_K_M.gguf` in place, and the app loads the first `.gguf` in directory
order (`Q4_K_M` sorts before `QAD`). Start from a clean install, or delete the
old file with `./scripts/deploy.sh delete-file <pfn> <old>.gguf 'models\<id>'`.
Every bench row must report `quant` = `Q4_0`.

```bash
source ~/.config/xllama/xbox-env
./scripts/install-latest-build.sh --provision
./scripts/provision-models.sh lfm25-230m lfm25-350m lfm25-1.2b-instruct
export XLLAMA_EXPECTED_PFN="$(./scripts/deploy.sh pfn)"
for m in lfm25-230m lfm25-350m lfm25-1.2b-instruct; do
  ./scripts/bench-xbox-ort.sh "$m" --runs 4 \
    --out bench/results/<date>-catalogue-gates.csv
done
./scripts/validate-console.sh gguf
./scripts/validate-api.sh spike
./scripts/eval-xbox-models.sh --models lfm25-230m,lfm25-350m,lfm25-1.2b-instruct \
  --out bench/results/<date>-catalogue-gates-h9.jsonl
```

The default prompt (`n_prompt_tok` 298) matches the same-build baseline
`2026-09-28-main-lfm.csv`. No-regression bar, fixed before the run, per id
against the recorded Q4_K_M rows in [model-matrix.md](model-matrix.md) §A1:

| Id                    | H9 ≥ | Median decode ≥ (0.9×) | `peak_ws_mb` ≤ (+10%) |
| --------------------- | ---: | ---------------------: | --------------------: |
| `lfm25-230m`          |  2/8 |       107.3 (of 119.2) |          265 (of 241) |
| `lfm25-350m`          |  4/8 |         80.7 (of 89.7) |          352 (of 320) |
| `lfm25-1.2b-instruct` |  6/8 |         34.1 (of 37.9) |          892 (of 811) |

PASS moves the §A1 rows (quant, numbers, evidence) and the
`benchmark-summary.json` sources to the new CSV. FAIL on any id reverts that
id's manifest pin to the previous Q4_K_M file.

## Troubleshooting

- `./scripts/deploy.sh diagnose-startup` — process state, log and crash dumps.
- `./scripts/deploy.sh get-log` — current `xllama.log`.
- Remove a stale `bench.flag` before UI/autopilot runs. It is uploaded only when
  `install-latest-build.sh --bench` is used.
- Reinstall with a higher package revision for an in-place update; a same-version
  package with different contents is rejected.
- Detailed Device Portal behavior: [device-portal.md](device-portal.md).

## Release acceptance

A hardware-sensitive change is complete only when the relevant automated gate
passes on the target console and the package version, raw evidence and outcome
are recorded. Do not revive closed historical experiments unless new evidence
changes a documented constraint.

## Recorded 2026-09-28 validation

The [source/package-bound record](../bench/results/2026-09-28-console-api-validation.json)
separates the controlled candidate console/LAN campaign from the exact main
shipping-package smoke. Full console gates ran against production-identical
candidate sources; the main package additionally passed health/chat/pull, both
embedding contract suites and maximum BGE input. It records failed/inconclusive
trials and avoids assigning cumulative process high-water marks to one model.
Automated doc/coherence checks do not replace those runtime observations.

## Controlled model-writer and KV fallback probes

Run these against the expected CI MSVC package, with the console credentials
loaded and `XLLAMA_EXPECTED_PFN` set to its full package identity. Use a private
output directory: the harness backs up settings, conversations and KV metadata
and restores every tracked original byte before restarting the normal app.

```bash
python scripts/validate-model-writer.py --out /private/writer-proof \
  --model-backup /private/LFM2.5-350M-QAD-Q4_0.gguf \
  --embedding-backup /private/bge-m3-Q8_0.gguf
python scripts/validate-kv-fallback.py --out /private/kv-proof
```

The writer probe verifies independently pinned weight backups before deleting
console weights to force real remote transfers. Both GUI/API ownership directions
must reject the competing writer without changing its files, preserve adapters,
verify downloaded hashes and recover after completion. The reverse trial releases
the GUI selection when streamed API download progress arrives; a transfer that
finishes before admission is inconclusive, not a concurrency PASS. API/API requests
may serialize and both succeed; that result alone does not prove overlap.

The KV probe forces a divergent saved assistant reply and requires the hybrid
fallback's complete prompt count and greedy output to match a fresh cold session.
It also sends native and OpenAI embedding requests while the GUI is generating
and requires their respective 503 error envelopes. A missing overlap fails the
probe; do not reinterpret a sequential 200 response as a busy-path validation.
