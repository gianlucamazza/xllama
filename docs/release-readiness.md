# Release readiness

> **Single source of truth for the dual-SKU policy, the catalogue licence matrix, and the App vs Game resource gate.**
> Product code and Dev Mode install docs stay on the
> research path. Numbers and platform constraints still live in
> `uwp-constraints.md` / `benchmarks.md`.

**Status (2026-08-26):** Phase 0 discovery + Phase 1 **engineering foundation**
landed, **five screenshots** captured (§9), **D1 App vs Game measured**
(§6 — request Game metadata), and **reduced-SKU console smoke PASS** (§10 —
CI run `32963270335`, catalogue signature verification + download + GGUF chat +
`set_api` reject). The Dev benchmark evidence is now recorded in
`bench/results/phase17-console-2026-08-26.csv` (3 warm-state repetitions;
methodology and missing external power evidence are documented in
`bench/README.md`). The catalogue verifier uses the pinned `release-v1` RSA
public key; CI signs reduced-SKU bundles with the private key held in
`XLLAMA_CATALOGUE_PRIVATE_KEY`.

Tested and measured on Series S. Not yet tested on Series X.

The package is still test-signed (`CN=xllama-dev`). Privacy URL:
<https://gianlucamazza.github.io/xllama/privacy.html>.

**Audience:** hobbyist local-LLM / homebrew Xbox users on Dev Mode.
Contributors keep the Dev Mode sideload path.

**Related:** product intent in [positioning.md](positioning.md);
`docs/api-endpoint.md` (LAN: research SKU only);
`docs/install-release.md` (Dev Mode only today).

---

## 1. Baseline (what ships today)

| Area               | Current                                           | Reduced SKU                                             |
| ------------------ | ------------------------------------------------- | ------------------------------------------------------- |
| Distribution       | Device Portal sideload + CI / GitHub Release      | Not distributed; CI artifact only (`workflow_dispatch`) |
| Signing            | `Publisher="CN=xllama-dev"`, self-signed test PFX | Same test identity                                      |
| Package identity   | `GianlucaMazza.xllama`                            | Same (`GianlucaMazza.xllama`)                           |
| LAN API            | Opt-in, unauthenticated                           | **Absent**                                              |
| USB models         | `removableStorage` + `E:\xllama\models`           | Prefer **absent**                                       |
| Content / AI       | No age gate, no generative-AI disclaimer          | Age note + first-run disclosure                         |
| Privacy            | HTTPS GitHub Pages (`privacy.html`)               | Same URL                                                |
| Docs product claim | “Dev Mode only”                                   | Dual path only after a public cut                       |

Manifest capabilities today (`uwp/AppxManifest.xml`): `internetClient`,
`privateNetworkClientServer`, `removableStorage`. Reduced-SKU target:
**`internetClient` only** (unless review forces another justified capability).

---

## 2. Product decisions (D1–D4)

### D1 — Game vs App designation

Benchmark and GPU-budget figures assume **Game** OS resources
(`uwp-constraints.md` §5 / App vs Game lever): **3801 MB** GPU budget, full
Game scheduling. In Dev Mode the designation is set in Dev Home (tile → View
details → App type) and can reset on reinstall.

| Designation                      | Implication                                                         |
| -------------------------------- | ------------------------------------------------------------------- |
| **Game** (current measured path) | Required for published perf claims; package metadata must match     |
| **App** (shared resources)       | Likely starves large models / diffusion; **must measure before go** |

**Gate (Phase 0 spike):** on a Series S, with the shipping default chat model
(`lfm25-350m`), run the same short decode bench under **App** and **Game**.
Record tok/s + peak working set. Procedure: §6 below. Not yet tested on
Series X.

- If App mode is unusable → a public package **must** obtain Game metadata, or
  do not ship.
- If App mode is “good enough” for hobbyist default chat → document the delta
  and avoid advertising Game-only numbers as if they were App.

### D2 — Dual-track SKU

| SKU                              | Channel                                           | Surfaces                                                            |
| -------------------------------- | ------------------------------------------------- | ------------------------------------------------------------------- |
| **dev** (default today)          | Dev Mode sideload                                 | LAN API, bench flags, USB path, full catalogue                      |
| **reduced** (`XLLAMA_STORE_SKU`) | Not published; CI artifact for console smoke only | No LAN, no bench/USB research paths; curated catalogue; disclaimers |

Compile-time flag: `XLLAMA_STORE_SKU`. Dev package identity
(`GianlucaMazza.xllama` / `CN=xllama-dev`) must **not** be broken for existing
Dev Mode installs.

### D3 — Reduced-SKU surface cuts

| Feature                                | Reduced SKU action                                           |
| -------------------------------------- | ------------------------------------------------------------ |
| LAN OpenAI endpoint                    | Remove (code + capability + Settings)                        |
| Headless `bench.flag` / operator flags | Remove or strip                                              |
| USB model path + `removableStorage`    | Remove                                                       |
| On-device training (Lane B)            | Allowed behind advanced UI if privacy-local; optional v1 cut |
| In-app model download                  | Keep (`internetClient` + licence matrix)                     |
| Stable Diffusion                       | Riskiest for review — text-only v1 is an accepted fallback   |
| Patched ORT/GenAI in-package           | Keep; pin lifecycle stays in `vendor-lifecycle-plan.md`      |

### D4 — Generative AI bar (minimum)

1. Honest **age note** (open chat + image gen ≠ Everyone).
2. **First-run disclosure:** local AI; may be wrong or inappropriate; not
   affiliated with Microsoft.
3. **Diffuse opt-in** + short NSFW risk note (or ship text-only first).
4. No false claims (e.g. "first LLM on Xbox").
5. **Privacy policy** URL (what stays on device; outbound = model downloads only
   unless opt-in telemetry is added later — default off).
6. **Support:** GitHub Issues and/or email.

---

## 3. Phases (summary)

| Phase                      | Goal                                                  | Exit                                      |
| -------------------------- | ----------------------------------------------------- | ----------------------------------------- |
| **0 Discovery**            | Go/no-go, this doc, App vs Game spike, licence matrix | Written go or stop                        |
| **1 Reduced SKU code**     | `XLLAMA_STORE_SKU`, capabilities, CI variant          | Installable reduced MSIX smoke on console |
| **2 Compliance pack**      | Privacy URL, age notes, screenshots, NOTICE           | Public description complete               |
| **3 Public Dev Mode path** | GitHub Release + sideload remain the install path     | Hobbyist Dev Mode install works           |

---

## 4. Model licence matrix (catalogue)

Authoritative download layout: `uwp/models/manifest.json`. Narrative redistribution
notes: `model-selection.md` (LFM / Qwen / Gemma). This table is the **catalogue
allowlist draft** — not legal advice; re-verify before shipping.

| Catalogue id                 | Family                 | Typical licence (upstream)                                                                                                                                                                                                                              | Hosting today                                  | Reduced SKU draft                                          |
| ---------------------------- | ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------- | ---------------------------------------------------------- |
| `lfm25-230m`                 | Liquid LFM2.5          | LFM Open License v1.0 (LICENSE next to weights; commercial use limited by entity revenue — see upstream §5)                                                                                                                                             | HF LiquidAI + LICENSE (QAD Q4_0)               | **Allow** floor tier if same                               |
| `lfm25-350m`                 | Liquid LFM2.5          | LFM Open License v1.0 (LICENSE next to weights; commercial use limited by entity revenue — see upstream §5)                                                                                                                                             | HF LiquidAI + LICENSE (QAD Q4_0)               | **Allow** default chat within the LFM §5 revenue threshold |
| `lfm25-1.2b-instruct`        | Liquid LFM2.5          | LFM Open License                                                                                                                                                                                                                                        | HF LiquidAI + LICENSE (QAD Q4_0)               | **Allow** if same                                          |
| `lfm2-2.6b`                  | Liquid LFM2            | LFM Open License                                                                                                                                                                                                                                        | HF LiquidAI + LICENSE                          | **Allow** if same                                          |
| `lfm25-1.2b-thinking`        | Liquid LFM2.5          | LFM Open License                                                                                                                                                                                                                                        | HF LiquidAI + LICENSE                          | **Allow** (advanced)                                       |
| `phi4-mini`                  | Phi-4                  | MIT (unsloth GGUF of microsoft/Phi-4-mini-instruct)                                                                                                                                                                                                     | HF unsloth                                     | **Allow** (Series S T3 PASS; not yet tested on Series X)   |
| `qwen35-0.8b`                | Qwen                   | Apache-2.0 (unsloth GGUF)                                                                                                                                                                                                                               | `models-v1`; licence copy not yet on models-v1 | **Allow**                                                  |
| `qwen3-1.7b`                 | Qwen                   | Apache-2.0                                                                                                                                                                                                                                              | HF unsloth                                     | **Allow**                                                  |
| `qwen25-coder-0.5b` / `1.5b` | Qwen2.5-Coder          | Apache-2.0                                                                                                                                                                                                                                              | HF unsloth                                     | **Allow** (`1.5b` is the coding option)                    |
| `qwen25-coder-3b`            | Qwen2.5-Coder          | [Qwen Research License (non-commercial)](https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct/blob/main/LICENSE)                                                                                                                                       | not catalogued (receipts only)                 | **Exclude** — not downloadable                             |
| `smollm2-360m-cpu-int4`      | SmolLM2                | Apache-2.0 (HuggingFaceTB lineage — confirm pin)                                                                                                                                                                                                        | `models-v1`; licence copy not yet on models-v1 | **Allow**                                                  |
| `smollm2-360m-dml-fp16-v2`   | SmolLM2                | same                                                                                                                                                                                                                                                    | `models-v1`; licence copy not yet on models-v1 | **Allow** (GPU text)                                       |
| `smollm2-1.7b-cpu-int4`      | SmolLM2                | same                                                                                                                                                                                                                                                    | `models-v1`; licence copy not yet on models-v1 | **Allow** / size check                                     |
| `gemma3-270m`                | Gemma                  | Gemma Terms of Use (not Apache)                                                                                                                                                                                                                         | HF only (no models-v1 mirror)                  | **Review** before shipping; may exclude v1                 |
| `gemma4-e2b`                 | Gemma                  | Gemma Terms                                                                                                                                                                                                                                             | HF only                                        | **Review** / likely advanced-only or exclude               |
| `llama32-3b`                 | Llama 3.2              | Meta Llama Community License                                                                                                                                                                                                                            | HF only (do not mirror without review)         | **Review** / likely exclude v1                             |
| `sd-turbo-fp16`              | Stable Diffusion Turbo | Stability AI Community License (2024-07-05): redistribution needs a copy of the licence, a "Notice" file and "Powered by Stability AI"; commercial use needs registration and ends above US$1M annual revenue. Registration with Stability AI: pending. | `models-v1`; licence copy not yet on models-v1 | **Highest risk** — text-only v1 if review hostile          |

**Rules of thumb for the public catalogue**

1. Prefer **Apache-2.0 + LFM-with-LICENSE-file** defaults.
2. Do **not** re-host Gemma/Llama on `models-v1` without explicit licence OK.
3. Ship **LFM 350M + SmolLM2 + Apache Qwen** as the safe core. `qwen25-coder-3b`
   is not in the catalogue (Qwen Research License, non-commercial; receipts only).
   Treat Gemma/Llama/SD as optional advanced or Dev-only until counsel/policy
   pass.
4. In-app download must remain **allowlisted** (manifest), fail-closed.

---

## 5. Gap checklist

### Account / process

- [x] Privacy policy URL:
      <https://gianlucamazza.github.io/xllama/privacy.html>
- [x] Support contact published (GitHub Issues)
- [x] Age / generative-AI notes drafted (§11)

### Engineering

- [x] `XLLAMA_STORE_SKU` / `XllamaStoreSku=true` strips LAN / USB / headless flags
- [x] `uwp/AppxManifest.store.xml` (`internetClient` only); identity still
      test `CN=xllama-dev`
- [x] `build-uwp.ps1 -StoreSku` (+ `/p:XllamaStoreSku=true`)
- [x] CI reduced-SKU lane via `workflow_dispatch` `store_sku=true` → artifact
      `xllama-appx-store` (no VM; not on every PR)
- [x] Configure the repository secret `XLLAMA_CATALOGUE_PRIVATE_KEY` with the
      offline RSA private key whose public half is compiled into
      `uwp/model-downloader.cpp`; the reduced-SKU workflow signs the bundled
      catalogue before packaging and fails closed when the secret is absent.
      Verified in CI run `32963270335` and by the reduced-SKU console smoke on
      2026-08-26.
- [x] `install-latest-build.sh --store` (Linux → Device Portal)
- [x] First-run generative-AI disclaimer (`LocalState\disclaimer.accepted`)
- [x] Privacy draft [`privacy.md`](./privacy.md)
- [x] NOTICE / runtime attributions draft (`NOTICE` at repo root — verify per release)
- [x] App vs Game spike results filed under `bench/results/`
      (`store-app-vs-game-2026-08-21.csv`, CI `1.5.5.922`, `lfm25-350m`)
- [x] Console smoke of reduced SKU (chat + model download) —
      `validate-console.sh store` PASS 2026-08-26 on CI run `32963270335`
      (package revision `959`, Game). Install start-app downloaded
      `lfm25-350m` from the signed catalogue; gate: GGUF chat with a saved
      title + `set_api` rejected (`LAN API not available in Store SKU`). Dev
      SKU restored after.

### Public description

- [x] Screenshots (5, 1920×1080, `docs/screenshots/store/`, 2026-07-30)
- [x] EN title / description / features (§9; no trailer yet)
- [x] “Not affiliated with Microsoft” (description + first-run disclaimer)
- [x] Hardware: tested and measured on Series S (not yet tested on Series X)
      (§9 additional requirements)
- [x] Category / Game metadata decision from D1: **request Game**. App runs
      default CPU chat only; GPU budget 691 vs 3801 MB and ramceil
      `avail_phys` ~183 vs ~4983 MB at 128 MB committed. See §6.
- [x] In-app report path (Settings → GitHub Issues catalogue-report form)

### Go/no-go

- [ ] **Go** only if: App-vs-Game acceptable, core catalogue licences OK,
      generative-AI bar accepted (or text-only fallback).
- [ ] **Stop / pivot** to “public Dev Mode only” if Game metadata is refused and
      App mode is unusable, or model/AI policy blocks a public cut.

---

## 6. Spike procedure — App vs Game resources

**Goal:** quantify whether a public **App** designation can run the default
hobbyist path.

**Prerequisites:** Series S in Dev Mode (not yet tested on Series X), current
shipping MSIX, default chat model already provisioned (`lfm25-350m`), Device
Portal access.

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

**Verdict (2026-08-21, CI `1.5.5.922`, `lfm25-350m` Q4_K_M, t6, `standard-512`,
3 recorded runs):** App **runs the default CPU chat**, ~8% slower on
long generations (decode **86.6–87.0** vs Game **94.7** tok/s at n_gen ≥ 150;
peak **320 MB** both). It is **not** a substitute for Game: GPU budget **691 vs
3801 MB**; ramceil `avail_phys` after 128 MB commit **183 vs 4983 MB**. Balanced /
quality / coding / DML / diffusion do not fit App. A public package **must
request Game metadata**; do not quote Game benches as App. CSV:
`bench/results/store-app-vs-game-2026-08-21.csv`.

Historical note: GPU budget **3801 MB** and published benchmarks are **Game**
numbers (`uwp-constraints.md` §5). App-mode numbers live only in that CSV.

---

## 7. Risks (short)

| Risk                                 | Mitigation                                 |
| ------------------------------------ | ------------------------------------------ |
| Review rejects open image gen        | v1 text-only; SD stays Dev / later update  |
| App mode too slow / OOM              | Require Game package metadata or no-go     |
| LFM / Gemma / Llama licence friction | Curated catalogue; default LFM+Qwen+Smol   |
| Dual identity confuses users         | Clear README: reduced vs Dev Mode packages |
| Research pace vs public freeze       | Reduced SKU thin; `main` stays Dev Mode    |

---

## 8. Not planned for the first release

- Full ID@Xbox Live services
- Authenticated LAN API on the reduced SKU
- Desktop as primary (optional later if `Windows.Desktop` stays)
- Monetisation
- Claiming Microsoft affiliation or “first LLM on Xbox”

---

## 9. No Windows VM — CI is the only UWP builder

Local packaging still needs Windows (see `windows-dev-vm.md`), but the
**supported reduced-SKU path does not**: use GitHub Actions `windows-2022`
from Linux.

### Build reduced SKU (from Linux) — verified

```bash
# On any branch that contains the reduced-SKU code:
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

### Install on Xbox Series S in Dev Mode (from Linux)

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

What changes under the reduced SKU:

| Surface                             | Dev SKU           | Reduced SKU                         |
| ----------------------------------- | ----------------- | ----------------------------------- |
| LAN API + Settings                  | yes               | compiled out                        |
| Headless flags (`bench.flag`, …)    | yes               | compiled out                        |
| USB model path + `removableStorage` | yes               | compiled out                        |
| First-run AI disclaimer             | yes (both)        | yes                                 |
| Publisher CN                        | `xllama-dev` test | same test identity                  |
| CI artifact                         | `xllama-appx`     | `xllama-appx-store` (dispatch only) |

### Privacy policy

Source of record: [`privacy.md`](./privacy.md). Public URL (GitHub Pages from
`docs/` on `main`; Jekyll renders this file):

<https://gianlucamazza.github.io/xllama/privacy.html>

### Public description draft (EN)

Limits from measured Game-envelope Series S figures in `bench/results` — do
not quote App-mode tok/s. Do not claim "first LLM on Xbox". Tested and
measured on Series S. Not yet tested on Series X.

| Field                  | Draft                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Product name           | xllama                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| Short description      | Local generative AI on Xbox (tested and measured on Series S; not yet tested on Series X): chat and optional image generation run on the console. Not affiliated with Microsoft. Models download on demand.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| Description            | xllama uses **on-device generative AI**. Language models and optional image generation run on the console. Tested and measured on Series S. Not yet tested on Series X. Nothing is sent to a cloud LLM API. Generated text and images can be inaccurate, biased, or inappropriate. This project is **not affiliated with Microsoft**. First launch downloads the default chat model from an allowlisted catalogue. Keep free storage: a few hundred MB for the default (LFM2.5-350M); several GB for larger models or image generation. Chat is gamepad-driven. Report inappropriate generated content from Settings. Performance in this description is the **Game** resource envelope on Series S: default chat about **101.6 tokens/s median** decode (101.5–101.8 across three runs) and about **311 MB** peak working set (LFM2.5-350M QAD Q4_0, CPU; capture 2026-09-30). Larger models are slower and heavier. Research-grade hobby project. No account, no analytics, no in-app purchases. |
| Product features       | On-device language models (no cloud LLM). Gamepad chat UI. On-demand catalogue download. Optional on-device image generation. First-run generative-AI disclosure. Report inappropriate content from Settings. Not affiliated with Microsoft.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| Category / genre       | **Game** (D1: App GPU budget 691 vs 3801 MB Game; ramceil `avail_phys` ~183 vs ~4983 MB at 128 MB committed. Default CPU chat runs on App; 1.2B+/DML/diffusion do not.)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| Age rating             | Expect **not Everyone**. Open generative text + optional images. Text-only v1 is the fallback if image gen fails review. See §11.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| Hardware               | Tested and measured on Series S (not yet tested on Series X). Additional: free storage for model downloads (hundreds of MB default; several GB for larger catalogue / image gen). Gamepad.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| Search terms (≤7)      | local LLM; on-device AI; chatbot; generative AI; image generation; hobby; console AI                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| Screenshots            | **5 captured** 2026-10-12 on 1.6.0.1202 with the LAN API off, as in the reduced SKU (`docs/screenshots/store/`, 1920×1080) via `scripts/capture-store-screenshots.sh`. Trailer: **none** for v1 (optional).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| Support                | GitHub Issues: https://github.com/gianlucamazza/xllama/issues                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| Privacy                | https://gianlucamazza.github.io/xllama/privacy.html                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| Notes for a public cut | Reduced SKU (`XLLAMA_STORE_SKU`): no LAN, no USB, no headless flags; capability `internetClient` only. First-run disclaimer. Most models download from Hugging Face; a few converted builds from `models-v1`. Generative AI is local; the report path is Settings → GitHub Issues. Request **Game** resource envelope. No Xbox network / ID@Xbox (no multiplayer). No login. First launch needs network for the default model download. Package identity is the test publisher (`CN=xllama-dev`); current CI MSIX is test-signed.                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |

## 10. Next steps (no VM)

1. **Console smoke — PASS 2026-08-26.** Reduced SKU CI
   [32963270335](https://github.com/gianlucamazza/xllama/actions/runs/32963270335)
   (`xllama_1.5.5.959_x64.msix`), App type **Game**. The log accepted the
   pinned RSA catalogue key and downloaded `lfm25-350m`. Repeat:
   ```bash
   source ~/.config/xllama/xbox-env
   ./scripts/install-latest-build.sh main --store   # uninstalls Dev SKU; wipes LocalState
   # Confirm Dev Home App type is still Game (reinstall can reset it).
   ./scripts/deploy.sh stop-app
   ./scripts/validate-console.sh store              # chat + catalogue download + set_api reject
   ./scripts/install-latest-build.sh main --provision  # restore Dev SKU
   ```
   Same identity as Dev: do not leave the reduced SKU installed. `--bench` is rejected.
2. NOTICE / attributions: already drafted at repo root; re-verify on the
   shipping build.

## 11. Age and generative-AI notes

These are product facts for an honest age note. They do **not** replace a
live questionnaire if one is ever required.

**Expected outcome:** not Everyone. Open-ended generative text, and optional
image generation, are not a children’s product (`privacy.md`). Typical digital
ratings land **Teen / PEGI 12–16** or higher once “users can generate
unrestricted content” is declared. If image gen is rejected, cut SD-Turbo
from v1 (D3 fallback) and treat the package as text-only.

### Interactive elements

| Topic                           | Answer                   | Why                                                                                 |
| ------------------------------- | ------------------------ | ----------------------------------------------------------------------------------- |
| Users interact with the product | Yes                      | Gamepad chat / Settings / catalogue                                                 |
| Users interact with each other  | **No**                   | Local only; no Xbox network, no accounts                                            |
| Users share location            | No                       | No location API                                                                     |
| Digital goods / IAP             | No                       | No purchases                                                                        |
| Personal info collected         | No (beyond device-local) | No account; see privacy URL                                                         |
| Unrestricted internet           | **Partial**              | `internetClient` for catalogue downloads only; no general web browser               |
| User-generated content online   | **No**                   | Generated output stays on the device; other users cannot view it in an online state |
| Live generative AI              | **Yes**                  | Local LLMs + optional image models respond to user prompts                          |

### Content (app assets vs generated)

App **assets** (UI, screenshots, icons): no violence, sexual content, profanity,
substances, or gambling.

**Generated** output is unconstrained: models can emit violence, sexual content,
strong language, or other disallowed _online_ UGC. Declare that honestly. Do
not claim a content filter we do not ship.

| Public v1 shape                              | Extra note                                                                                                      |
| -------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| Text + optional images (current reduced SKU) | Highest image-gen / report-path risk. Keep the report path. Be ready to disable image gen if review rejects it. |
| Text-only fallback                           | Same generative-text answers; image-related questions **No**. D3 already allows this.                           |

We have neither an opt-in filter nor a sign-in gate, which is another reason
any assigned rating must already cover unrestricted generation.
