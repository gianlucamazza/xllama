# Phase 1 Runbook — CPU Baseline on Xbox Series S

End-to-end instructions for reproducing the Phase 1 benchmark from scratch.

## Prerequisites

- Xbox Series S with Dev Mode activated and Device Portal enabled
- Xbox IP address and Device Portal credentials
- Linux dev machine with this repo cloned (`--recursive`)
- Windows machine (or VM) with Visual Studio 2022 + UWP workload for building .msix
- GGUF model files (see "Download models" below)

For local UWP builds from the Arch workstation, use the Windows VM workflow in
[windows-dev-vm.md](./windows-dev-vm.md). Linux Docker cannot build the UWP/MSIX
package because Windows SDK packaging tools require Windows.

## 1. Build .msix

**Option A — Local Windows VM:**
```powershell
.\scripts\build-uwp.ps1
# Output: uwp\AppPackages\xllama\xllama_0.1.0.0_x64.msix
```

**Option B — GitHub Actions CI:**

Push to `main`; the `build-uwp` workflow runs automatically on `windows-2022` (pinned: only runner with VS2022 + UWP workload).
Download the `xllama-appx` artifact from the Actions run.

## 2. Deploy to Xbox

```bash
export XBOX_IP=192.168.1.42
export XBOX_USER=devuser
export XBOX_PASS=your_password

./scripts/deploy.sh path/to/xllama_0.1.0.0_x64.msix
```

The script uploads the package, polls installation status, and prints "Installation succeeded."

After deployment, the app appears in Dev Home. It can be launched from there.

The script also auto-installs the companion `.cer` if present alongside the `.msix` (included in the CI artifact zip). Without the trusted certificate, Xbox rejects the package with `0x800B0100`.

## 3. Verify app starts

Launch xllama from Dev Home. The XAML UI loads: header shows the model filename (from `LocalState/model.txt`), status reads "Ready", ProgressBar is hidden.

Verify in Device Portal File Explorer that `LocalState/xllama.log` exists and contains:
```
[xllama] App::App()
[xllama] App::OnLaunched
[xllama] Window activated
```

These three lines confirm the UWP lifecycle completed and the XAML frame was activated. If the UI is a black screen instead, `LoadComponent` failed to find `MainPage.xaml` — check that the MSIX contains `MainPage.xaml` (run `unzip -l *.msix | grep xaml`). `App.xaml` is build-time metadata only and is intentionally not shipped as loose runtime XAML.

URL: `https://<ip>:11443/#fileExplorer`
Navigate: LocalAppData → `VenereLabs.xllama_0.1.0.0_x64__<token>` → `LocalState` → `xllama.log`

## 4. Download models

Phase 1 models are standard GGUF files. Download from Hugging Face:

| Model | HF repo | Filename |
|-------|---------|----------|
| Qwen3 1.7B Q4_K_M | `Qwen/Qwen3-1.7B-GGUF` | `Qwen_Qwen3-1.7B-Q4_K_M.gguf` |
| Llama 3.2 3B Q4_K_M | `bartowski/Llama-3.2-3B-GGUF` | `Llama-3.2-3B-Q4_K_M.gguf` |
| Qwen3 8B Q4_K_M | `Qwen/Qwen3-8B-GGUF` | `Qwen_Qwen3-8B-Q4_K_M.gguf` |

```bash
# Example with huggingface-cli (pip install huggingface-hub)
huggingface-cli download Qwen/Qwen3-1.7B-GGUF Qwen_Qwen3-1.7B-Q4_K_M.gguf \
    --local-dir ~/models/
```

The app reads the active model filename from `LocalState/model.txt` (one line, no trailing newline). The hardcoded default is `Qwen_Qwen3-1.7B-Q4_K_M.gguf`. To switch models, upload a new `model.txt`:

```bash
echo -n "Llama-3.2-3B-Q4_K_M.gguf" > /tmp/model.txt
./scripts/deploy.sh upload-file /tmp/model.txt "$PFN" ""
```

Models are resolved under `LocalState\models\` by `resolve_model_path()` in `src/bridge/path_utils.cpp`. Upload them there:

```bash
./scripts/deploy.sh upload-file ~/models/Qwen_Qwen3-1.7B-Q4_K_M.gguf "$PFN" models
```

## 5. Run benchmark (automated)

```bash
export XBOX_IP=192.168.1.42 XBOX_USER=devuser XBOX_PASS=your_password

# Qwen3 1.7B — primary target
./scripts/bench-xbox.sh ~/models/qwen3-1.7b-Q4_K_M.gguf \
    bench/config/phase1-qwen3-1.7b.json

# Llama 3.2 3B
./scripts/bench-xbox.sh ~/models/llama-3.2-3b-Q4_K_M.gguf \
    bench/config/phase1-llama32-3b.json

# Qwen3 8B (takes ~30 min total for 3 runs)
./scripts/bench-xbox.sh ~/models/qwen3-8b-Q4_K_M.gguf \
    bench/config/phase1-qwen3-8b.json
```

Each invocation:
1. Uploads the model to `LocalFolder/models/` on the console
2. Runs 3 inference iterations (drops run 1 as warmup)
3. Computes median and appends one row to `bench/results/phase1-cpu.csv`

## 6. Manual bench (fallback)

If `bench-xbox.sh` can't control the app lifecycle automatically:

```bash
# 1. Get package full name
curl -sS --basic -u $XBOX_USER:$XBOX_PASS -k \
    "https://$XBOX_IP:11443/api/app/packagemanager/packages" \
    | python3 -c "import sys,json; [print(p['PackageFullName']) for p in json.load(sys.stdin)['InstalledPackages'] if 'VenereLabs.xllama' in p.get('PackageRelativeId','')]"
# → e.g. VenereLabs.xllama_0.1.0.0_x64__abc123

PFN="VenereLabs.xllama_0.1.0.0_x64__abc123"

# 2. Upload model
./scripts/deploy.sh upload-file ~/models/qwen3-1.7b-Q4_K_M.gguf "$PFN" models

# 3. Upload prompt
cp bench/prompts/standard-512.txt /tmp/prompt.txt
./scripts/deploy.sh upload-file /tmp/prompt.txt "$PFN" ""

# 4. Launch app from Xbox UI (Dev Home → xllama)

# 5. Wait ~2-3 min, then fetch results
curl -sS --basic -u $XBOX_USER:$XBOX_PASS -k \
    "https://$XBOX_IP:11443/api/filesystem/apps/file?knownfolderid=LocalAppData&packagefullname=${PFN}&path=\\LocalState&filename=bench-result.csv" \
    -o /tmp/bench-result.csv

cat /tmp/bench-result.csv
# Append data row to bench/results/phase1-cpu.csv manually
tail -n +2 /tmp/bench-result.csv >> bench/results/phase1-cpu.csv
```

## 7. Interpreting results

Expected ranges (Xbox Series S, Zen 2 CPU, no Vulkan):

| Model | Quant | Size | Expected decode tok/s |
|-------|-------|------|----------------------|
| Qwen3 1.7B | Q4_K_M | ~1.1 GB | 25–40 |
| Llama 3.2 3B | Q4_K_M | ~2.0 GB | 15–25 |
| Qwen3 8B | Q4_K_M | ~4.7 GB | 6–10 |

Load times will be slow until Stage 1E (`CreateFileMappingFromApp`) because the model
is fully `fread()`'d into heap memory instead of memory-mapped.

## 7b. Xbox Device Portal API quirks

Xbox WDP differs from Desktop WDP in several ways discovered during Phase 1:

| Topic | Desktop WDP | Xbox WDP |
|---|---|---|
| Auth method | HTTP Digest | **HTTP Basic** (`--basic`) |
| Install URL | `/api/app/packagemanager/package` | `/api/app/packagemanager/package?package=<filename>` (required query param) |
| Install cert | n/a | `POST /api/app/packagemanager/certificate?package=<certname>` |
| Signing | Optional | **Required** even in Dev Mode (`0x800B0100` if absent) |
| LocalFolder path | Varies | `ApplicationData.LocalFolder` → `LocalState/` subdir of the package |
| File download param | `path=\\<file>` | `path=\\LocalState&filename=<name>` (lowercase `filename`) |
| File upload path | `path=\\<dir>` | `path=\\LocalState[\\subdir]` |

## 8. Troubleshooting

**App crashes at startup**: check `xllama.log` in LocalFolder via Device Portal (see §8a).

**Model not found**: ensure the GGUF is in `LocalFolder/models/` (not `LocalFolder/`).

**Slow load time**: expected with `use_mmap=false`. Qwen3-1.7B takes ~10-30s to load.

**Deploy script auth failure**: Xbox Device Portal uses **HTTP Basic** auth (not Digest as on Desktop WDP). Use `--basic` with curl.
Verify credentials in Dev Home → Settings → Device Portal credentials.

**"Not ready yet" in dashboard / `0x80270300` on launch**: framework dependency missing.
`deploy.sh` installs `Dependencies/x64/*.appx` automatically alongside the `.msix`. Re-run deploy.

**`UnhandledException: 0x802B000A` (E_XAMLPARSEFAILED) before OnLaunched**: XAML runtime
called `IXamlMetadataProvider::GetXamlType(...)` while parsing loose runtime XAML with
`x:Class`; the stub returned null. `build-uwp.ps1` strips `x:Class` from `MainPage.xaml`
and removes loose `App.xaml` from the final MSIX layout.

## 8a. Debug logging

### xllama.log

Every startup writes `LocalState/xllama.log` (append, UTC timestamps `HH:MM:SS.mmm`).
Expected sequence on clean launch:

```
HH:MM:SS.mmm [xllama] App::App()
HH:MM:SS.mmm [xllama] App::OnLaunched
HH:MM:SS.mmm [xllama] Window activated
```

Retrieve via WDP:

```bash
source ~/.config/xllama/xbox-env
PFN="VenereLabs.xllama_0.1.0.0_x64__pj67f1fcj4n14"
curl --basic -u "${XBOX_USER}:${XBOX_PASS}" -k -sS \
     -o /tmp/xllama.log \
     "https://${XBOX_IP}:11443/api/filesystem/apps/file?\
knownfolderid=LocalAppData&packagefullname=${PFN}&path=\\LocalState&filename=xllama.log"
cat /tmp/xllama.log
```

Diagnostic entries logged automatically:
- `InitializeComponent FAILED 0x... <message>` — XAML init failure with full description
- `UnhandledException: 0x...` — any unhandled XAML-thread exception
- `wWinMain exception: 0x...` — exception escaping `Application::Start`

### WDP crash dump (minidump)

If the process terminates without leaving a log entry, collect a user-mode minidump:

```bash
# List available crash dumps
curl --basic -u "${XBOX_USER}:${XBOX_PASS}" -k -sS \
     "https://${XBOX_IP}:11443/api/debug/dump/usermode/dumps"

# Download the most recent dump for xllama.exe
curl --basic -u "${XBOX_USER}:${XBOX_PASS}" -k -sS \
     -o /tmp/xllama.dmp \
     "https://${XBOX_IP}:11443/api/debug/dump/usermode/dump?pid=<PID>&type=2"

# Analyse on desktop with WinDbg:
#   .sympath srv*https://msdl.microsoft.com/download/symbols
#   !analyze -v
```

`type=2` = MiniDumpWithFullMemory. Requires Dev Mode + WDP enabled.
