# Phase 1 Runbook — CPU Baseline on Xbox Series S

End-to-end instructions for building, deploying, and benchmarking xllama on Xbox Series S with ORT GenAI / SmolLM2-360M-Instruct.

## Prerequisites

- Xbox Series S with Dev Mode activated and Device Portal enabled
- Xbox IP address and Device Portal credentials
- Linux dev machine with this repo cloned (`--recursive`)
- Credentials file at `~/.config/xllama/xbox-env` (see `.env.example`)

The bundled model (SmolLM2-360M-Instruct INT4 CPU, 403 MB) is included in the MSIX — no separate download or upload is required.

For local UWP builds from an Arch workstation, use the Windows VM workflow in
[windows-dev-vm.md](./windows-dev-vm.md). Linux cannot build the UWP/MSIX package because Windows SDK packaging tools require Windows.

## 1. Get or build the MSIX

**Option A — GitHub Actions CI (recommended):**

Push to `main`; the `build-uwp` workflow runs on `windows-2022` automatically.
Download the `xllama-appx` artifact from the Actions run. It contains:

- `xllama_*.msix`
- `Dependencies/x64/*.appx` (framework dependencies)
- `uwp/xllama-test.cer`

**Option B — Local Windows VM:**

```powershell
.\scripts\build-uwp.ps1 -Configuration Release -Platform x64
# Output: uwp\AppPackages\xllama\xllama_*.msix
```

The CI and the local build both run `scripts/merge_onnx_external_data.py` to merge the model's external data into a self-contained `model.onnx` before packaging (required for AppContainer compatibility — see `docs/uwp-constraints.md §8`).

## 2. Deploy to Xbox

```bash
source ~/.config/xllama/xbox-env

./scripts/deploy.sh path/to/xllama_*.msix
```

The script uploads the MSIX, installs the `.cer` (if present alongside the MSIX), polls installation status, and prints "Installation succeeded."

After deployment, the app appears in Dev Home. Launch it from there.

Without the certificate, Xbox rejects the package with `0x800B0100`.

## 3. Verify app starts

Launch xllama from Dev Home. The UI loads: a header shows the model name (`smollm2-360m-cpu-int4`), a text area for input, and a status bar.

Check the startup log via Device Portal:

```bash
source ~/.config/xllama/xbox-env
PFN=$(./scripts/deploy.sh pfn)
./scripts/deploy.sh get-log
```

Expected sequence:

```
HH:MM:SS.mmm [xllama] App::App()
HH:MM:SS.mmm [xllama] App::OnLaunched
HH:MM:SS.mmm [xllama] MainPageController::Init
HH:MM:SS.mmm [xllama] Model resolved: Q:\...\models\smollm2-360m-cpu-int4
HH:MM:SS.mmm [xllama] Window activated
```

These lines confirm UWP lifecycle completed, the model path was resolved, and the window is live.

URL to Device Portal File Explorer: `https://<ip>:11443/#fileExplorer`
Navigate to: LocalAppData → `VenereLabs.xllama_..._x64__<token>` → `LocalState` → `xllama.log`

## 4. Switching models

The default bundled model is `smollm2-360m-cpu-int4`. Three alternatives are available, in order of preference:

**Option A — Settings ComboBox (v0.3.0+, recommended):**
Open the ⚙ Settings dialog inside the app. The Model ComboBox exposes:

- SmolLM2-360M (bundled MSIX)
- SmolLM2-1.7B (USB `E:\xllama\models\`)
- SmolLM2-360M (HF download in LocalState)

The selection is persisted to `LocalState/settings.json` and takes effect on the next inference call (session rebuilt transparently).

**Option B — USB drive (Exp 3, for models too large to bundle):**
Place the ONNX GenAI directory on a NTFS USB stick under `E:\xllama\models\<model-name>\`. `resolve_model_path` probes `E:\` automatically when LocalState and InstalledPath entries are absent.

**Option C — Device Portal upload (dev/scripted use):**

1. Build the model as an ONNX GenAI directory and merge external data:
   ```bash
   python3 scripts/merge_onnx_external_data.py <model_dir>
   ```
2. Upload to the console:
   ```bash
   source ~/.config/xllama/xbox-env
   PFN=$(./scripts/deploy.sh pfn)
   ./scripts/deploy.sh upload-dir ./my-model-dir/ "$PFN" "models\\my-model-name"
   ```
3. The app reads `LocalState/settings.json` for the active model. Edit via the Settings dialog, or write `LocalState/model.txt` as a legacy fallback (0.2.x compat):
   ```bash
   echo -n "my-model-name" > /tmp/model.txt
   ./scripts/deploy.sh upload-file /tmp/model.txt "$PFN" ""
   ```

Models are resolved by `resolve_model_path()` in `src/bridge/path_utils.cpp` in this order: `LocalState\models\<name>` → `Package.InstalledPath\models\<name>` → `E:\xllama\models\<name>`.

**Note**: the model directory must contain `genai_config.json` at its root — ORT GenAI loads the model from the directory, not from `model.onnx` directly.

## 5. Run benchmark (automated)

```bash
source ~/.config/xllama/xbox-env

./scripts/bench-xbox.sh smollm2-360m-cpu-int4 bench/config/phase1-smollm2-360m.json
```

Each invocation:

1. Writes the model name to `LocalState/model.txt`
2. Uploads `bench/prompts/standard-512.txt` to `LocalState/prompt.txt`
3. Writes `bench.flag` to trigger bench mode on next launch
4. Runs 3 iterations (drops run 1 as cold start)
5. Fetches `bench-result.csv` from `LocalState` and appends median to `bench/results/phase1-cpu.csv`

## 6. Manual bench (fallback)

If `bench-xbox.sh` cannot control the app lifecycle automatically:

```bash
source ~/.config/xllama/xbox-env
PFN=$(./scripts/deploy.sh pfn)

# 1. Write model name
echo -n "smollm2-360m-cpu-int4" > /tmp/model.txt
./scripts/deploy.sh upload-file /tmp/model.txt "$PFN" ""

# 2. Upload prompt
./scripts/deploy.sh upload-file bench/prompts/standard-512.txt "$PFN" ""

# 3. Touch bench.flag
echo -n "1" > /tmp/bench.flag
./scripts/deploy.sh upload-file /tmp/bench.flag "$PFN" ""

# 4. Launch app from Xbox UI (Dev Home → xllama)

# 5. Wait ~30-60s, then fetch results
curl -sS --basic -u "${XBOX_USER}:${XBOX_PASS}" -k \
    "https://${XBOX_IP}:11443/api/filesystem/apps/file?\
knownfolderid=LocalAppData&packagefullname=${PFN}&path=\\LocalState&filename=bench-result.csv" \
    -o /tmp/bench-result.csv

cat /tmp/bench-result.csv
tail -n +2 /tmp/bench-result.csv >> bench/results/phase1-cpu.csv
```

## 7. Interpreting results

Expected ranges (Xbox Series S, Zen 2 CPU, CPU EP, SmolLM2-360M INT4):

| Model        | Quant    | Backend            | Expected decode tok/s |
| ------------ | -------- | ------------------ | --------------------- |
| SmolLM2-360M | INT4 CPU | CPU EP (ORT GenAI) | 60–73                 |

The `backend` field in the CSV will be `ort-genai-cpu` (compile-time label driven by `XLLAMA_USE_ORT`). The runtime execution provider on Xbox is CPU EP, matching the label.

Load time is ~1–2 s (model is small; no mmap needed).

## 7b. Xbox Device Portal API quirks

| Topic               | Desktop WDP                       | Xbox WDP                                                      |
| ------------------- | --------------------------------- | ------------------------------------------------------------- |
| Auth method         | HTTP Digest                       | **HTTP Basic** (`--basic`)                                    |
| Install URL         | `/api/app/packagemanager/package` | `/api/app/packagemanager/package?package=<filename>`          |
| Install cert        | n/a                               | `POST /api/app/packagemanager/certificate?package=<certname>` |
| Signing             | Optional                          | **Required** — `0x800B0100` if absent                         |
| File download param | `path=\\<file>`                   | `path=\\LocalState&filename=<name>` (lowercase `filename`)    |
| File upload path    | `path=\\<dir>`                    | `path=\\LocalState[\\subdir]`                                 |

## 8. Troubleshooting

**App crashes at startup**: check `xllama.log` via `./scripts/deploy.sh get-log` or Device Portal File Explorer.

**`OgaCreateModel failed`**: ORT GenAI could not load the model. Common causes:

- Model directory not found: verify `LocalState\models\smollm2-360m-cpu-int4\genai_config.json` exists.
- External data not merged: ensure the MSIX was built with `merge_onnx_external_data.py` applied. The CI does this automatically.
- `weakly_canonical: Access is denied`: same root cause as above — external data file present in the model directory triggers the AppContainer path bug. Re-merge and redeploy.

**DirectML OOM** (`SEH 0xC0000005` on `OgaCreateModel` with DML EP): the GPU pool is too small. Use CPU EP (`"provider_options": []` in `genai_config.json`). See `docs/uwp-constraints.md §5`.

**Model not found** (fallback to defaults): ensure `genai_config.json` is at the root of the model directory, not inside a subdirectory.

**Deploy script auth failure**: Xbox Device Portal uses **HTTP Basic** auth, not Digest. Use `--basic` with curl. Verify credentials in Dev Home → Settings → Device Portal credentials.

**`0x800B0100` on install**: signing certificate not trusted. Re-run `deploy.sh` with the `.cer` alongside the `.msix`, or manually install via `./scripts/deploy.sh install-cert`.

**`0x80270300` on launch** (framework dependency missing): `deploy.sh` installs `Dependencies/x64/*.appx` automatically alongside the MSIX. Re-run deploy with the full artifact set.

## 8a. Debug logging

### xllama.log

Startup log at `LocalState/xllama.log` (UTC timestamps, append mode).

Fetch:

```bash
source ~/.config/xllama/xbox-env
./scripts/deploy.sh get-log
```

### WDP crash dump (minidump)

If the process terminates without a log entry:

```bash
source ~/.config/xllama/xbox-env

# List available dumps
curl --basic -u "${XBOX_USER}:${XBOX_PASS}" -k -sS \
     "https://${XBOX_IP}:11443/api/debug/dump/usermode/dumps"

# Download most recent dump
curl --basic -u "${XBOX_USER}:${XBOX_PASS}" -k -sS \
     -o /tmp/xllama.dmp \
     "https://${XBOX_IP}:11443/api/debug/dump/usermode/dump?pid=<PID>&type=2"
```

Analyse with WinDbg: `.sympath srv*https://msdl.microsoft.com/download/symbols` then `!analyze -v`.

## 9. DML profiling and GPU telemetry (GPU truth)

Answers the Phase 2 question — does the DML EP execute on the RDNA 2 GPU or
silently fall back to CPU? — without PIX (GDK-only, unavailable in Dev Mode).
Background and caveats: `docs/uwp-constraints.md §11`.

### One profiled DML run

```bash
source ~/.config/xllama/xbox-env
./scripts/profile-dml-run.sh --model smollm2-360m-cpu-int4 --gpu-sample
```

The script swaps in `bench/configs/genai_config-dml-profile.json` (DML EP +
`enable_profiling` + verbose ORT logging), runs one bench inference, downloads
`ort_profile_*.json` + the new `xllama.log` tail + `bench-result.csv` into
`bench/results/profiles/<timestamp>/`, restores the original config, and prints:

```
VERDICT: GPU                      # DML kernel time >= 90%
VERDICT: MIXED (dml=X% cpu=Y%)    # both providers active
VERDICT: CPU-FALLBACK             # zero DML kernel events
```

If the profile is not found: rerun with `--absolute-prefix` (renders the
`.tpl.json` config with an absolute LocalState prefix); if still missing,
deploy a v0.3.2+ MSIX (CWD pinned to LocalState) and rerun.

**DML requires a v0.3.4+ MSIX** (headless bench mode): in earlier versions the
XAML compositor's D3D12 device makes the DML EP init throw `887A0036` — see
`docs/uwp-constraints.md §7`. The DML configs also need
`past_present_share_buffer: true` (already set in `bench/configs/`).

### Interpretation

| Signal                       | GPU execution               | CPU fallback        |
| ---------------------------- | --------------------------- | ------------------- |
| Profiler `VERDICT:`          | `GPU` / high-DML `MIXED`    | `CPU-FALLBACK`      |
| `gpu-sample` engines         | 3D/compute > ~0.3 sustained | flat on all engines |
| `[xllama] gpu-mem post-load` | `current` ≈ model size      | `current` ≈ 0       |
| CSV `gpu_mem_mb`             | ≈ model size                | ≈ 0                 |

Run a control pass with the stock CPU config (expected `CPU-FALLBACK` + flat
engines) to calibrate both probes — `systemperf` is system-wide and Dev Home
activity adds GPU noise.

### Bench with GPU telemetry (v0.3.2+ MSIX)

```bash
./scripts/bench-xbox-ort.sh smollm2-360m-cpu-int4 --runs 3 \
    --out bench/results/phase2-dml.csv --gpu-sample
```

`gpu_mem_mb`/`gpu_budget_mb` CSV columns come from per-process
`QueryVideoMemoryInfo` (LOCAL segment) sampled after model load.
