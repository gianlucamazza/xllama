# Store readiness — Xbox Store retail path

> **SSOT for the public Xbox Store workstream.** Product code and Dev Mode
> install docs stay on the research path until a submission is accepted.
> Numbers and platform constraints still live in `uwp-constraints.md` /
> `benchmarks.md`; this page owns **go-to-market gates**, dual-SKU policy, and
> the licence matrix for a retail listing.

**Status (2026-08-21):** Phase 0 discovery + Phase 1 **engineering foundation**
landed, **five listing screenshots** captured (§9), **D1 App vs Game measured**
(§6 — request Game metadata), and **Store SKU console smoke PASS** (§10 —
CI `1.5.5.928`, Game, catalogue download + GGUF chat + `set_api` reject).
Product Dev Mode cut is **v1.5.5.0** (10 console gates including `thinkdone`;
last fully gated evidence remains v1.5.4.0 / MSIX `1.5.4.887`). **Not
Store-ready** for retail: no Partner Center product, no Store-signed package, no
IARC certificate. Privacy HTML is published at
<https://gianlucamazza.github.io/xllama/privacy.html> (GitHub Pages). EN listing
copy and an IARC prep sheet are in this file. Partner Center account / product
reservation remains a **human gate** (ID verification in the browser).

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
| Privacy            | HTTPS GitHub Pages (`privacy.html`)               | Same URL in Partner Center           |
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
| `lfm25-230m`                        | Liquid LFM2.5          | LFM Open License v1.0 (LICENSE next to weights; commercial use limited by entity revenue — see upstream §5) | `models-v1` + LICENSE                  | **Allow** floor tier if same                                        |
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

- [ ] Partner Center individual account created (or explicit no-go) —
      free flow at <https://storedeveloper.microsoft.com> (ID + selfie)
- [ ] Product reserved; Store publisher CN known
- [x] IARC age rating questionnaire drafted (§11; complete live in Partner Center)
- [x] Privacy policy URL:
      <https://gianlucamazza.github.io/xllama/privacy.html>
- [x] Support contact published (GitHub Issues; listing copy below)

### Engineering

- [x] `XLLAMA_STORE_SKU` / `XllamaStoreSku=true` strips LAN / USB / headless flags
- [x] `uwp/AppxManifest.store.xml` (`internetClient` only); identity still test CN
- [x] `build-uwp.ps1 -StoreSku` (+ `/p:XllamaStoreSku=true`)
- [x] CI store lane via `workflow_dispatch` `store_sku=true` → artifact
      `xllama-appx-store` (no VM; not on every PR)
- [x] `install-latest-build.sh --store` (Linux → Device Portal)
- [x] First-run generative-AI disclaimer (`LocalState\disclaimer.accepted`)
- [x] Privacy draft [`privacy.md`](./privacy.md)
- [x] NOTICE / runtime attributions draft (`NOTICE` at repo root — verify per release)
- [x] App vs Game spike results filed under `bench/results/`
      (`store-app-vs-game-2026-08-21.csv`, CI `1.5.5.922`, `lfm25-350m`)
- [x] Console smoke of store SKU (chat + model download) —
      `validate-console.sh store` PASS 2026-08-21 on CI Store SKU `1.5.5.928`
      (Game). Install start-app downloaded `lfm25-350m` from the catalogue;
      gate: GGUF chat with a saved title + `set_api` rejected
      (`LAN API not available in Store SKU`). Dev SKU restored after.

### Listing

- [x] Screenshots (5, 1920×1080, `docs/screenshots/store/`, 2026-07-30)
- [x] EN title / description / features / search terms (§9; no trailer yet)
- [x] “Not affiliated with Microsoft” (listing + first-run disclaimer)
- [x] Hardware: Series S\|X (§9 additional requirements)
- [x] Category / Game metadata decision from D1: **request Game**. App runs
      default CPU chat only; GPU budget 691 vs 3801 MB and ramceil
      `avail_phys` ~183 vs ~4983 MB at 128 MB committed. See §6.
- [x] Store policy **11.16** disclosure in listing metadata; in-app report
      path (Settings → GitHub Issues `store-report`)

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

**Verdict (2026-08-21, CI `1.5.5.922`, `lfm25-350m` Q4_K_M, t6, `standard-512`,
3 recorded runs):** App **runs the default CPU chat**, ~8% slower on
long generations (decode **86.6–87.0** vs Game **94.7** tok/s at n_gen ≥ 150;
peak **320 MB** both). It is **not** a substitute for Game: GPU budget **691 vs
3801 MB**; ramceil `avail_phys` after 128 MB commit **183 vs 4983 MB**. Balanced /
quality / coding / DML / diffusion do not fit App. Store listing **must request
Game metadata**; do not quote Game benches as App. CSV:
`bench/results/store-app-vs-game-2026-08-21.csv`.

Historical note: GPU budget **3801 MB** and published benchmarks are **Game**
numbers (`uwp-constraints.md` §5). App-mode numbers live only in that CSV.

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

### Privacy policy

Source of record: [`privacy.md`](./privacy.md). Partner Center URL (GitHub
Pages from `docs/` on `main`; Jekyll renders this file):

<https://gianlucamazza.github.io/xllama/privacy.html>

### Listing draft (EN, not submitted)

Paste into Partner Center. Limits from
[Store listing info (MSIX)](https://learn.microsoft.com/en-us/windows/apps/publish/publish-your-app/msix/add-and-edit-store-listing-info).
Numbers are Game-envelope Series S figures from `docs/launch-copy.md` /
`bench/results` — do not quote App-mode tok/s. Do not claim “first LLM on Xbox”
(`docs/launch-copy.md`).

| Field                   | Draft                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Product name            | xllama                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| Short description       | Local generative AI on Xbox Series S\|X: chat and optional image generation run on the console. Not affiliated with Microsoft. Models download on demand.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| Description             | xllama uses **on-device generative AI**. Language models and optional image generation run on the Xbox Series S\|X. Nothing is sent to a cloud LLM API. Generated text and images can be inaccurate, biased, or inappropriate. This project is **not affiliated with Microsoft**. First launch downloads the default chat model from an allowlisted catalogue. Keep free storage: a few hundred MB for the default (LFM2.5-350M); several GB for larger models or image generation. Chat is gamepad-driven. Report inappropriate generated content from Settings. Performance on this listing is the **Game** resource envelope on Series S: default chat about **95 tokens/s** decode and about **320 MB** peak working set (LFM2.5-350M Q4_K_M, CPU). Larger models are slower and heavier. Research-grade hobby project. No account, no analytics, no in-app purchases. |
| Product features        | On-device language models (no cloud LLM). Gamepad chat UI. On-demand catalogue download. Optional on-device image generation. First-run generative-AI disclosure. Report inappropriate content from Settings. Not affiliated with Microsoft.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| Category / genre        | **Game** (D1: App GPU budget 691 vs 3801 MB Game; ramceil `avail_phys` ~183 vs ~4983 MB at 128 MB committed. Default CPU chat runs on App; 1.2B+/DML/diffusion do not.)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| Age rating              | IARC in Partner Center — prep sheet §11. Expect **not Everyone**. Open generative text + optional images. Text-only Store v1 is the fallback if image gen fails 11.7 / certification.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| Hardware                | Xbox Series S or Series X. Additional: free storage for model downloads (hundreds of MB default; several GB for larger catalogue / image gen). Gamepad.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| Search terms (≤7)       | local LLM; on-device AI; chatbot; generative AI; image generation; hobby; console AI                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| Screenshots             | **5 captured** 2026-07-30 (`docs/screenshots/store/`, 1920×1080) via `scripts/capture-store-screenshots.sh`. Trailer: **none** for v1 (optional).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| Support                 | GitHub Issues: https://github.com/gianlucamazza/xllama/issues                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| Privacy                 | https://gianlucamazza.github.io/xllama/privacy.html                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| Notes for certification | Store SKU (`XLLAMA_STORE_SKU`): no LAN, no USB, no headless flags; capability `internetClient` only. First-run disclaimer. Models from GitHub Releases `models-v1`. Generative AI is local; 11.16 report path is Settings → GitHub Issues. Request **Game** resource envelope. No Xbox network / ID@Xbox (no multiplayer). No login. First launch needs network for the default model download. Package identity is still the test publisher until this product’s Store CN is reserved and stamped — current CI MSIX is **not** Store-signed.                                                                                                                                                                                                                                                                                                                              |

## 10. Next steps (no VM)

1. **Console smoke — PASS 2026-08-21.** Store SKU CI
   [32473955046](https://github.com/gianlucamazza/xllama/actions/runs/32473955046)
   (`xllama_1.5.5.928_x64.msix`), App type **Game**. Repeat:
   ```bash
   source ~/.config/xllama/xbox-env
   ./scripts/install-latest-build.sh main --store   # uninstalls Dev SKU; wipes LocalState
   # Confirm Dev Home App type is still Game (reinstall can reset it).
   ./scripts/deploy.sh stop-app
   ./scripts/validate-console.sh store              # chat + catalogue download + set_api reject
   ./scripts/install-latest-build.sh main --provision  # restore Dev SKU
   ```
   Same identity as Dev: do not leave the Store SKU installed. `--bench` is rejected.
2. **Human (browser):** open
   [storedeveloper.microsoft.com](https://storedeveloper.microsoft.com) →
   **Get started for free** → **Individual developer**. Sign in with a personal
   Microsoft account, complete government-ID + selfie verification. **Do not**
   start from Partner Center / Visual Studio — those still show the legacy paid
   flow.
   Docs:
   [open a developer account](https://learn.microsoft.com/en-us/windows/apps/publish/partner-center/open-a-developer-account),
   [free individual registration](https://learn.microsoft.com/en-us/windows/apps/publish/whats-new-individual-developer).
3. After the Apps & Games tile appears: create a **Game** product targeting Xbox
   Series S\|X, reserve the name `xllama`, paste §9 listing copy, set the privacy
   URL, run the IARC questionnaire from §11, tick **live generative AI**
   (Store policy 11.16).
4. **Do not upload** the current test-signed CI MSIX. Partner Center must first
   reserve the Store publisher identity; then a Store SKU build stamps that CN.
5. NOTICE / attributions: already drafted at repo root; re-verify on the
   submission build.

## 11. IARC prep sheet (not a certificate)

IARC runs **inside Partner Center** after the product exists. Exact wording of
the live form is not published; this sheet is what to answer from product facts.
It does **not** replace completing the questionnaire. IARC shares the publisher
display name and email with the rating authorities
([age ratings](https://learn.microsoft.com/en-us/windows/apps/publish/publish-your-app/msix/age-ratings)).

**Expected outcome:** not Everyone. Open-ended generative text, and optional
image generation, are not a children’s product (`privacy.md`). Typical digital
ratings land **Teen / PEGI 12–16** or higher once “users can generate
unrestricted content” is declared. If image gen pushes Adult/pornographic under
Store policy **11.7**, cut SD-Turbo from Store v1 (D3 fallback) and re-run IARC
as text-only.

### Interactive elements (both Store v1 shapes)

| Topic                           | Answer                   | Why                                                                                 |
| ------------------------------- | ------------------------ | ----------------------------------------------------------------------------------- |
| Users interact with the product | Yes                      | Gamepad chat / Settings / catalogue                                                 |
| Users interact with each other  | **No**                   | Local only; no Xbox network, no accounts                                            |
| Users share location            | No                       | No location API                                                                     |
| Digital goods / IAP             | No                       | No purchases                                                                        |
| Personal info collected         | No (beyond device-local) | No account; see privacy URL                                                         |
| Unrestricted internet           | **Partial**              | `internetClient` for catalogue downloads only; no general web browser (10.13.4)     |
| User-generated content online   | **No** (11.12)           | Generated output stays on the device; other users cannot view it in an online state |
| Live generative AI              | **Yes** (11.16)          | Local LLMs + optional image models respond to user prompts                          |

### Content (app assets vs generated)

App **assets** (UI, screenshots, icons): no violence, sexual content, profanity,
substances, or gambling. Listing metadata must stay at PEGI 12 / ESRB E10+ or
lower (**11.1**).

**Generated** output is unconstrained: models can emit violence, sexual content,
strong language, or other disallowed _online_ UGC. Declare that honestly in IARC
(“can users create content that may include …”). Do not claim a content filter
we do not ship.

| Store v1 shape                             | Extra IARC / policy note                                                                                    |
| ------------------------------------------ | ----------------------------------------------------------------------------------------------------------- |
| Text + optional images (current Store SKU) | Highest 11.7 / 11.16 risk. Keep the report path. Be ready to disable image gen if certification rejects it. |
| Text-only fallback                         | Same generative-text answers; image-related questions **No**. D3 already allows this.                       |

### After the form

Save and generate. Paste the resulting rating table back into this section (do
not invent ESRB/PEGI numbers here). 11.11.3: if generated content can exceed the
assigned rating, the product must offer an opt-in filter or a sign-in gate — we
have neither, which is another reason the assigned rating must already cover
unrestricted generation.

## 12. Partner Center operator steps

Cannot be completed by CI. Identity verification is in the browser.

1. Open <https://storedeveloper.microsoft.com> (only supported free-flow entry).
2. **Get started for free** → **Individual developer** (hobby / personal project;
   Store policy 10.14). Company is wrong unless publishing as a legal entity.
3. Personal Microsoft account (MSA). Government-issued ID + selfie on the phone.
4. Wait for Apps & Games (up to a few minutes). Direct link:
   <https://aka.ms/submitwindowsapp>.
5. Create product: **Game**, Xbox console. Reserve `xllama`.
6. Listing: paste §9. Privacy URL: `https://gianlucamazza.github.io/xllama/privacy.html`.
7. Age ratings: IARC from §11. Check the Partner Center box for live generative AI
   (11.16).
8. Properties: Xbox Series S\|X; gamepad; Game resource envelope (D1).
9. Stop. Store-signed identity and package upload wait on the reserved publisher
   CN — that is Phase 3 engineering, not this pack.

**Game vs Xbox network (10.13.1).** Game products on Xbox must use Xbox network
via ID@Xbox **or** publish without those services through the Creators-style
path. xllama has **no** multiplayer, friends list, or Xbox sign-in (D5: ID@Xbox
out of scope). If Partner Center blocks a Game without Xbox network, that is a
go/no-go: either accept Creators constraints, list as App (D1 says App cannot
carry 1.2B+/DML/diffusion), or stop.
