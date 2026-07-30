# Store readiness — Xbox Store retail path

> **SSOT for the public Xbox Store workstream.** Product code and Dev Mode
> install docs stay on the research path until a submission is accepted.
> Numbers and platform constraints still live in `uwp-constraints.md` /
> `benchmarks.md`; this page owns **go-to-market gates**, dual-SKU policy, and
> the licence matrix for a retail listing.

**Status (2026-07-29):** Phase 0 discovery + Phase 1 **engineering foundation**
landed. **Not Store-ready** for retail: no Partner Center product, no
Store-signed package, no privacy URL, no age rating. Dev Mode remains the only
supported install path.

**Audience for a eventual listing:** hobbyist local-LLM / homebrew Xbox users
who should not need Dev Mode. Contributors keep the Dev Mode sideload path.

**Related:** plan in session / product intent; `docs/launch-copy.md` (claims
that must not be made); `docs/api-endpoint.md` (LAN: not for the Store);
`docs/install-release.md` (Dev Mode only today).

---

## 1. Baseline (what ships today)

| Area               | Current                                           | Store needs                          |
| ------------------ | ------------------------------------------------- | ------------------------------------ |
| Distribution       | Device Portal sideload + CI / GitHub Release      | Partner Center submission            |
| Signing            | `Publisher="CN=xllama-dev"`, self-signed test PFX | Partner Center publisher identity    |
| Package identity   | `GianlucaMazza.xllama`                            | Reserved Store identity (may differ) |
| LAN API            | Opt-in, unauthenticated                           | **Absent** from Store SKU            |
| USB models         | `removableStorage` + `E:\xllama\models`           | Prefer **absent** from Store SKU     |
| Content / AI       | No age gate, no generative-AI disclaimer          | Age rating + first-run disclosure    |
| Privacy            | No public policy                                  | HTTPS privacy URL + support contact  |
| Docs product claim | “Dev Mode only — no retail path”                  | Dual path only **after** Store live  |

Manifest capabilities today (`uwp/AppxManifest.xml`): `internetClient`,
`privateNetworkClientServer`, `removableStorage`. Store SKU target:
**`internetClient` only** (unless review forces another justified capability).

---

## 2. Product decisions (D1–D5)

### D1 — Game vs App designation

Benchmark and GPU-budget figures assume **Game** OS resources
(`uwp-constraints.md` §5 / App vs Game lever): **3801 MB** GPU budget, full
Game scheduling. In Dev Mode the designation is set in Dev Home (tile → View
details → App type) and can reset on reinstall.

| Designation                      | Implication                                                         |
| -------------------------------- | ------------------------------------------------------------------- |
| **Game** (current measured path) | Required for published perf claims; listing metadata must match     |
| **App** (shared resources)       | Likely starves large models / diffusion; **must measure before go** |

**Gate (Phase 0 spike):** on a Series S, with the shipping default chat model
(`lfm25-350m`), run the same short decode bench under **App** and **Game**.
Record tok/s + peak working set. Procedure: §6 below.

- If App mode is unusable → Store listing **must** obtain Game metadata, or do
  not ship.
- If App mode is “good enough” for hobbyist default chat → document the delta
  and avoid advertising Game-only numbers on the listing.

### D2 — Dual-track SKU

| SKU                     | Channel           | Surfaces                                                            |
| ----------------------- | ----------------- | ------------------------------------------------------------------- |
| **dev** (default today) | Dev Mode sideload | LAN API, bench flags, USB path, full catalogue                      |
| **store**               | Xbox Store retail | No LAN, no bench/USB research paths; curated catalogue; disclaimers |

Compile-time flag (planned Phase 1): `XLLAMA_STORE_SKU`. Dev package identity
(`GianlucaMazza.xllama` / `CN=xllama-dev`) must **not** be broken for existing
Dev Mode installs when the Store identity is reserved.

### D3 — Store SKU surface cuts

| Feature                                | Store SKU action                                                 |
| -------------------------------------- | ---------------------------------------------------------------- |
| LAN OpenAI endpoint                    | Remove (code + capability + Settings)                            |
| Headless `bench.flag` / operator flags | Remove or strip                                                  |
| USB model path + `removableStorage`    | Remove                                                           |
| On-device training (Lane B)            | Allowed behind advanced UI if privacy-local; optional v1 cut     |
| In-app model download                  | Keep (`internetClient` + licence matrix)                         |
| Stable Diffusion                       | Riskiest for review — text-only v1 Store is an accepted fallback |
| Patched ORT/GenAI in-package           | Keep; pin lifecycle stays in `vendor-lifecycle-plan.md`          |

### D4 — Generative AI policy bar (minimum)

1. Honest **IARC / age rating** (open chat + image gen ≠ Everyone).
2. **First-run disclosure:** local AI; may be wrong or inappropriate; not
   affiliated with Microsoft.
3. **Diffuse opt-in** + short NSFW risk note (or ship text-only first).
4. No false claims (`docs/launch-copy.md`: not “first LLM on Xbox”).
5. **Privacy policy** URL (what stays on device; outbound = model downloads only
   unless opt-in telemetry is added later — default off).
6. **Support:** GitHub Issues and/or email.

### D5 — Microsoft program path

1. Partner Center **individual** developer account (one-time fee).
2. Create product, reserve package identity / publisher.
3. UWP package targeting `Windows.Xbox` (Desktop optional later).
4. **ID@Xbox not required** for day-1 (no Xbox Live multiplayer/achievements).
5. Submit → certification loop → Available.

Out of scope day-1: paid IAP, XCloud, Microsoft marketing, LAN-API Store rewrite.

---

## 3. Phases (summary)

| Phase                 | Goal                                                  | Exit                                    |
| --------------------- | ----------------------------------------------------- | --------------------------------------- |
| **0 Discovery** (now) | Go/no-go, this doc, App vs Game spike, licence matrix | Written go or stop                      |
| **1 Store SKU code**  | `XLLAMA_STORE_SKU`, capabilities, CI variant          | Installable store MSIX smoke on console |
| **2 Compliance pack** | Privacy URL, age rating, listing assets, NOTICE       | Partner Center listing complete         |
| **3 Submission**      | Upload, cert iteration, Game metadata if needed       | Live on Store                           |
| **4 Post-launch**     | README dual path, stable vs research cadence          | Hobbyist retail install works           |

Calendar order-of-magnitude: **~2–4 months**, dominated by review and policy,
not pure coding.

---

## 4. Model licence matrix (catalogue)

Authoritative download layout: `uwp/models/manifest.json`. Narrative redistribution
notes: `model-selection.md` (LFM / Qwen / Gemma). This table is the **Store
allowlist draft** — not legal advice; re-verify before submission.

| Catalogue id                        | Family                 | Typical licence (upstream)                                                                                  | Hosting today                          | Store SKU draft                                                     |
| ----------------------------------- | ---------------------- | ----------------------------------------------------------------------------------------------------------- | -------------------------------------- | ------------------------------------------------------------------- |
| `lfm25-350m`                        | Liquid LFM2.5          | LFM Open License v1.0 (LICENSE next to weights; commercial use limited by entity revenue — see upstream §5) | `models-v1` + LICENSE                  | **Allow** default chat if non-commercial listing OK under LFM terms |
| `lfm25-1.2b-instruct`               | Liquid LFM2.5          | LFM Open License                                                                                            | HF LiquidAI + LICENSE                  | **Allow** if same                                                   |
| `lfm2-2.6b`                         | Liquid LFM2            | LFM Open License                                                                                            | HF LiquidAI + LICENSE                  | **Allow** if same                                                   |
| `lfm25-1.2b-thinking`               | Liquid LFM2.5          | LFM Open License                                                                                            | HF LiquidAI + LICENSE                  | **Allow** (advanced)                                                |
| `qwen35-0.8b`                       | Qwen                   | Apache-2.0 (unsloth GGUF)                                                                                   | `models-v1`                            | **Allow**                                                           |
| `qwen3-1.7b`                        | Qwen                   | Apache-2.0                                                                                                  | HF unsloth                             | **Allow**                                                           |
| `qwen25-coder-0.5b` / `1.5b` / `3b` | Qwen2.5-Coder          | Apache-2.0                                                                                                  | HF unsloth                             | **Allow**                                                           |
| `smollm2-360m-cpu-int4`             | SmolLM2                | Apache-2.0 (HuggingFaceTB lineage — confirm pin)                                                            | `models-v1`                            | **Allow**                                                           |
| `smollm2-360m-dml-fp16-v2`          | SmolLM2                | same                                                                                                        | `models-v1`                            | **Allow** (GPU text)                                                |
| `smollm2-1.7b-cpu-int4`             | SmolLM2                | same                                                                                                        | `models-v1`                            | **Allow** / size check                                              |
| `gemma3-270m`                       | Gemma                  | Gemma Terms of Use (not Apache)                                                                             | HF only (no models-v1 mirror)          | **Review** before Store; may exclude v1                             |
| `gemma4-e2b`                        | Gemma                  | Gemma Terms                                                                                                 | HF only                                | **Review** / likely advanced-only or exclude                        |
| `llama32-3b`                        | Llama 3.2              | Meta Llama Community License                                                                                | HF only (do not mirror without review) | **Review** / likely exclude v1 Store                                |
| `sd-turbo-fp16`                     | Stable Diffusion Turbo | Stability / model card terms + OpenRAIL-class constraints                                                   | `models-v1`                            | **Highest risk** — text-only Store v1 if review hostile             |

**Rules of thumb for Store v1 catalogue**

1. Prefer **Apache-2.0 + LFM-with-LICENSE-file** defaults.
2. Do **not** re-host Gemma/Llama on `models-v1` without explicit licence OK.
3. Ship **LFM 350M + SmolLM2 + Qwen** as the safe core; treat Gemma/Llama/SD as
   optional advanced or Dev-only until counsel/policy pass.
4. In-app download must remain **allowlisted** (manifest), fail-closed.

---

## 5. Gap checklist (Phase 0 → 3)

### Account / process

- [ ] Partner Center individual account created (or explicit no-go)
- [ ] Product reserved; Store publisher CN known
- [ ] IARC age rating questionnaire drafted
- [ ] Privacy policy URL live
- [ ] Support contact published

### Engineering

- [x] `XLLAMA_STORE_SKU` / `XllamaStoreSku=true` strips LAN / USB / headless flags
- [x] `uwp/AppxManifest.store.xml` (`internetClient` only); identity still test CN
- [x] `build-uwp.ps1 -StoreSku` (+ `/p:XllamaStoreSku=true`)
- [x] CI store lane via `workflow_dispatch` `store_sku=true` → artifact
      `xllama-appx-store` (no VM; not on every PR)
- [x] `install-latest-build.sh --store` (Linux → Device Portal)
- [x] First-run generative-AI disclaimer (`LocalState\disclaimer.accepted`)
- [x] Privacy draft [`privacy.md`](./privacy.md)
- [ ] NOTICE / runtime attributions (ORT, llama.cpp, DirectML, models)
- [ ] App vs Game spike results filed under `bench/results/`
- [ ] Console smoke of store SKU (chat + model download)

### Listing

- [ ] EN title / description / screenshots / trailer (current numbers)
- [ ] “Not affiliated with Microsoft”
- [ ] Hardware: Series S\|X
- [ ] Category / Game metadata decision from D1

### Go/no-go

- [ ] **Go** only if: Partner Center ready, App-vs-Game acceptable, core
      catalogue licences OK, generative-AI bar accepted (or text-only fallback).
- [ ] **Stop / pivot** to “public Dev Mode only” if Game metadata is refused and
      App mode is unusable, or model/AI policy blocks listing.

---

## 6. Spike procedure — App vs Game resources

**Goal:** quantify whether a retail **App** designation can run the default
hobbyist path.

**Prerequisites:** Series S (or X) in Dev Mode, current shipping MSIX, default
chat model already provisioned (`lfm25-350m`), Device Portal access.

1. Install / upgrade to current unified package (`install-release.md`).
2. Dev Home → xllama tile → **View details** → set **App type = App**.
3. Cold-start app, load `lfm25-350m`, run a fixed prompt (same as a known bench
   row, e.g. short decode from `bench/README.md` methodology).
4. Capture: decode tok/s, prefill tok/s if available, `peak_working_set_mb` (log
   or headless bench if still available on dev SKU), GPU budget if logged.
5. Switch **App type = Game**, reboot/relaunch if needed, repeat **identical**
   prompt.
6. Write CSV under `bench/results/store-app-vs-game-<date>.csv` with columns:
   `designation,model,prefill_tps,decode_tps,peak_ws_mb,notes`.
7. Paste a one-line verdict into this section when done.

**Verdict (fill after measurement):** _TBD — not yet run._

Historical note: GPU budget **3801 MB** and published benchmarks are **Game**
numbers (`uwp-constraints.md` §5). Do not quote them as App-mode until measured.

---

## 7. Risks (short)

| Risk                                 | Mitigation                                      |
| ------------------------------------ | ----------------------------------------------- |
| Certification rejects open image gen | Store v1 text-only; SD stays Dev / later update |
| App mode too slow / OOM              | Require Game listing metadata or no-go          |
| LFM / Gemma / Llama licence friction | Curated catalogue; default LFM+Qwen+Smol        |
| Dual identity confuses users         | Clear README: Store vs Dev Mode packages        |
| Research pace vs Store freeze        | Store SKU thin; `main` stays Dev Mode           |

---

## 8. Explicit non-goals (day-1)

- Full ID@Xbox Live services
- Authenticated LAN API on Store SKU
- Desktop Store as primary (optional later if `Windows.Desktop` stays)
- Monetisation
- Claiming Microsoft affiliation or “first LLM on Xbox”

---

## 9. No Windows VM — CI is the only UWP builder

Local packaging still needs Windows (see `windows-dev-vm.md`), but the **supported
Store path does not**: use GitHub Actions `windows-2022` from Linux.

### Build Store SKU (from Linux) — verified

```bash
# On any branch that contains the Store SKU code:
gh workflow run build-uwp.yml -f store_sku=true --ref "$(git branch --show-current)"
gh run watch   # wait for success

# Artifact name: xllama-appx-store
gh run list --workflow build-uwp --limit 5
gh run download <run-id> -n xllama-appx-store
```

PR/push builds keep **only** the usual `xllama-appx` + `xllama-appx-llamacpp`
lanes (no extra cost). The **`store` job** runs only when `workflow_dispatch`
sets `store_sku=true` (composite action `.github/actions/build-uwp-package`).

**Smoke (2026-07-29, branch `research/h2-ram-ceiling`):** run
[30446583341](https://github.com/gianlucamazza/xllama/actions/runs/30446583341)
— `store` + unified + llamacpp all green; artifact `xllama-appx-store`
downloaded on Linux as `xllama_1.5.1.771_x64.msix` (~19 MB) + `xllama-test.cer`.

### Install on Xbox Dev Mode (from Linux)

Same identity as dev today → **replaces** the Dev Mode package on the console.

```bash
source ~/.config/xllama/xbox-env
./scripts/install-latest-build.sh --store          # this branch
./scripts/install-latest-build.sh main --store     # explicit branch
# --bench is rejected on --store (headless flags compiled out)
```

### Optional local Windows (not required)

```powershell
.\scripts\build-uwp.ps1 -Configuration Release -Platform x64 `
  -Backend unified -PatchedGenAI -PatchedOrt -StoreSku
```

MSBuild: `/p:XllamaStoreSku=true` → `XLLAMA_STORE_SKU=1` + `AppxManifest.store.xml`.

What changes under Store SKU:

| Surface                             | Dev SKU           | Store SKU                           |
| ----------------------------------- | ----------------- | ----------------------------------- |
| LAN API + Settings                  | yes               | compiled out                        |
| Headless flags (`bench.flag`, …)    | yes               | compiled out                        |
| USB model path + `removableStorage` | yes               | compiled out                        |
| First-run AI disclaimer             | yes (both)        | yes                                 |
| Publisher CN                        | `xllama-dev` test | still test until Partner Center     |
| CI artifact                         | `xllama-appx`     | `xllama-appx-store` (dispatch only) |

### Privacy policy draft

In-repo: [`privacy.md`](./privacy.md). For Partner Center, host a stable HTTPS
URL (GitHub Pages or equivalent) pointing at this content.

### Listing draft (EN, not submitted)

| Field             | Draft                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Title             | xllama                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| Short description | Local LLM chat on Xbox Series S\|X — models run on the console.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| Long description  | xllama turns an Xbox Series S\|X into a local inference box: gamepad chat UI, on-demand model catalogue, optional image generation. Nothing is sent to a cloud LLM API. Research-grade hobby project; not affiliated with Microsoft. Requires enough free storage for model downloads.                                                                                                                                                                                                                                                                                                 |
| Category          | (TBD — Game metadata preferred if resource measurements require it; see §2 D1)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| Age rating        | (TBD — IARC; generative text + optional image gen)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| Screenshots       | **2 captured** on 2026-07-30 (`docs/screenshots/store/`, 1920×1080 straight off the console, inside the accepted 1366×768 – 3840×2160 range with no resampling) via `scripts/capture-store-screenshots.sh`. Chat and multi-turn only. Settings, History **and the image viewer** are all `ContentDialog`s: `generate_image` completes but the chat view only reports `> Image ready — open [*] Image to view`, so the image is behind that dialog and a frame taken there shows chat text. All three need one `show_pane` autopilot op that opens and closes in the same action (#214) |
| Support           | GitHub Issues on gianlucamazza/xllama                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| Privacy           | Link to published `privacy.md`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |

## 10. Next steps (no VM)

1. Push branch → `gh workflow run build-uwp.yml -f store_sku=true` → download
   `xllama-appx-store` (validates packaging without a VM).
2. **Human:** Partner Center account (browser); App vs Game spike on console (§6).
3. Publish privacy URL; finish listing + age rating.
4. NOTICE / attributions pack; then submission after smoke on console.
