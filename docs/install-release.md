# Install a release build on your Xbox

How to install a tagged xllama release (e.g.
[v1.0.0](https://github.com/gianlucamazza/xllama/releases/tag/v1.0.0)) on an
Xbox Series S|X in Dev Mode, from a Linux/macOS host. For building from source
see the [README](../README.md#build); for Dev Mode activation and Device
Portal basics see [device-portal.md](./device-portal.md).

## Prerequisites

- Xbox in **Dev Mode** with the Device Portal enabled (note IP + credentials).
- A credentials file, e.g. `~/.config/xllama/xbox-env`:

  ```bash
  export XBOX_IP=192.168.1.44
  export XBOX_USER=...
  export XBOX_PASS=...
  ```

- Internet access on the console (first launch downloads the default model).

## 1. Download the release assets

From the GitHub Release page (or `gh release download vX.Y.Z`):

- `xllama_X.Y.Z.0_x64.msix` — the app (~19 MB, no model inside)
- `xllama-test.cer` — the signing test certificate
- `Microsoft.VCLibs.x64.14.00.appx` — runtime dependency

## 2. Install the certificate, dependency, and app

```bash
source ~/.config/xllama/xbox-env

# Trust cert (deploy.sh also auto-installs a .cer found next to the .msix)
./scripts/deploy.sh install-cert xllama-test.cer

# VCLibs dependency: install once via the Device Portal UI
#   https://<XBOX_IP>:11443 → My games & apps → Install → Microsoft.VCLibs.x64.14.00.appx
# (or add it as a dependency in the same WDP install dialog as the MSIX)

# App
./scripts/deploy.sh xllama_X.Y.Z.0_x64.msix
```

Notes:

- **Upgrades**: a higher version installs over the old one and **preserves**
  LocalState (models, settings, history). WDP refuses a same-version install
  with different contents.
- After (re)installs, check the **App type** in Dev Home (tile → View details):
  the measured performance figures assume the **Game** designation
  (`uwp-constraints.md §5`).

## 3. First launch

Launch xllama from Dev Home (or `./scripts/deploy.sh start-app`). The app
downloads the default chat model (~417 MB) with a progress bar, then opens the
chat. See [using-the-app.md](./using-the-app.md) from here.

For image generation, the SD-Turbo model (2.4 GB) is downloaded from the
catalogue on the first **Generate** — mind the Dev Mode disk budget (~2.2–2.5 GB
free by default, **expandable to 90 GB** via Dev Home → Manage Dev Storage; see
[uwp-constraints.md §9](uwp-constraints.md)). Device Portal provisioning remains available
([../diffusion/README.md](../diffusion/README.md), runbook §7).

## Troubleshooting

- `0x80070070` (disk full) while installing models: the Dev Mode partition has
  ~2.2–2.5 GB free by default — remove unused models from LocalState, or raise the
  Dev Mode allocation (up to 90 GB, `uwp-constraints.md §9`).
- Startup issues: `./scripts/deploy.sh diagnose-startup` prints the process
  state, the app log, and any crash dumps in one shot.
- Log at any time: `./scripts/deploy.sh get-log`; individual files:
  `./scripts/deploy.sh fetch-file "$(./scripts/deploy.sh pfn)" <name> <local-out>`.
