# Install a release build on your Xbox

Install a tagged app release or a specific CI MSVC package on an Xbox in
Developer Mode. Tested and measured on Series S. Not yet tested on Series X.
The 2026-09-28 app release is
[v1.6.0.0](https://github.com/gianlucamazza/xllama/releases/tag/v1.6.0.0).
Its release notes identify the exact CI package and release validation results.
The [releases page](https://github.com/gianlucamazza/xllama/releases) also contains
research tags, which can be newer without containing an app MSIX.

Two install paths: Device Portal in a browser (no repo checkout, no
credentials file), or the command-line `deploy.sh` path below.

For building from source see the [README](../README.md#quick-install); for Dev
Mode activation and Device Portal basics see
[device-portal.md](./device-portal.md).

## Install from a release in the browser

These Device Portal steps follow Microsoft's documentation and have not yet
been walked end to end on a console. If something does not match, please
[open an issue](https://github.com/gianlucamazza/xllama/issues).

This path uses the console's Device Portal in a browser and the assets on the
repo's latest **app** GitHub Release. It does not need a clone of this
repository or `~/.config/xllama/xbox-env`.

The latest app release
([v1.6.0.0](https://github.com/gianlucamazza/xllama/releases/tag/v1.6.0.0))
attaches these files (names from the GitHub Release API):

| Asset                             | What it is                                                                                                                                     |
| --------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `xllama_1.6.0.1046_x64.msix`      | The app package (~19 MB, no model inside). The last part of the version number is the CI run stamp and changes with each release.              |
| `xllama-test.cer`                 | The test signing certificate. **Required** — shipping GitHub releases are test-signed, and Device Portal will not install the MSIX without it. |
| `Microsoft.VCLibs.x64.14.00.appx` | Visual C++ runtime. **Required** — the MSIX declares `PackageDependency` `Microsoft.VCLibs.140.00`.                                            |
| `SHA256SUMS`                      | SHA-256 of the files above (and of `release-validation.json`).                                                                                 |
| `release-validation.json`         | Validation record for that tag. Not needed to install.                                                                                         |

Read the real MSIX name off the release you download. Later app tags keep the
`xllama_*_x64.msix` pattern and still attach the `.cer`, VCLibs `.appx`, and
`SHA256SUMS`. Skip `research-v*` tags: they have no app MSIX.

### 1. Enable Developer Mode

Xbox Developer Mode is a Microsoft program. You activate it with **your own**
Partner Center app developer account. xllama does not provide one.

Follow Microsoft's
[Xbox Developer Mode activation](https://learn.microsoft.com/en-us/windows/uwp/xbox-apps/devkit-activation):
install the Xbox Dev Mode app from the Microsoft Store, note the activation
code, enter it in Partner Center, then switch the console to Developer Mode.

### 2. Open Device Portal in a browser

In Developer Mode, enable Device Portal from Dev Home and set its username and
password. Steps: [device-portal.md](./device-portal.md) and Microsoft's
[Device Portal for Xbox](https://learn.microsoft.com/en-us/windows/uwp/xbox-apps/device-portal-xbox).

Open `https://<console-ip>:11443`. Accept the console's self-signed TLS
certificate on first visit, then sign in with those Device Portal credentials.
The IP is on the Dev Home screen, or under **Settings → Network → Advanced
settings**.

### 3. Download the release assets

From the [releases page](https://github.com/gianlucamazza/xllama/releases),
open the latest **app** tag (currently
[v1.6.0.0](https://github.com/gianlucamazza/xllama/releases/tag/v1.6.0.0))
and download:

- `xllama_*_x64.msix` (v1.6.0.0 ships `xllama_1.6.0.1046_x64.msix`)
- `xllama-test.cer`
- `Microsoft.VCLibs.x64.14.00.appx`
- `SHA256SUMS`

### 4. Verify SHA-256

This is the one step that uses a terminal.

Each line of `SHA256SUMS` is `<hash>  <filename>`. Check the MSIX, the
`.cer`, and the VCLibs `.appx` against that file before installing:

```bash
sha256sum -c SHA256SUMS
```

On macOS use `shasum -a 256 -c SHA256SUMS`. If you also downloaded
`release-validation.json`, it has a line in `SHA256SUMS` too.

### 5. Install via Device Portal

In Device Portal, open **Home** and choose **Install** under **My games & apps**.
Microsoft documents this dialog in
[Device Portal for Xbox](https://learn.microsoft.com/en-us/windows/uwp/xbox-apps/device-portal-xbox)
and the
[Windows Device Portal overview](https://learn.microsoft.com/en-us/windows/uwp/debug-test-perf/device-portal#install-sideload-an-app):
it takes the app package, a certificate, and optional framework packages.
xllama needs all three:

1. **Certificate**: `xllama-test.cer`. The console must trust it before the
   MSIX will install ([device-portal.md](./device-portal.md)). If the dialog
   has no certificate field, install the `.cer` first.
2. **Framework dependency**: `Microsoft.VCLibs.x64.14.00.appx`. Add it as a
   framework package in the same dialog, or install it first through
   **My games & apps → Install**.
3. **App** — the `xllama_*_x64.msix` you verified.

Then start the install.

### 6. Launch

Launch xllama from Dev Home. What happens on first launch is in [First launch](#first-launch).

A higher version with the same identity (`GianlucaMazza.xllama`) installs over
the old one and **preserves** LocalState. Device Portal refuses a same-version
install with different contents. After install, check **App type** in Dev Home
(tile → View details): the measured figures assume the **Game** designation
(`uwp-constraints.md §5`).

## Install from the command line

### Prerequisites

- Xbox in **Dev Mode** with the Device Portal enabled (note IP + credentials).
- A credentials file, e.g. `~/.config/xllama/xbox-env`:

  ```bash
  export XBOX_IP=192.168.1.x
  export XBOX_USER=...
  export XBOX_PASS=...
  ```

- Internet access on the console (first launch downloads the default model).

### 1. Download the release assets

From the GitHub Release page (or `gh release download vX.Y.Z`):

- `xllama_X.Y.Z.REV_x64.msix` — the app (~19 MB, no model inside). The fourth
  component `REV` is the CI run stamp and differs per release; local builds keep
  `.0`. Read the real name off the release page, or let the shell below derive
  it from the file you downloaded.
- `xllama-test.cer` — the signing test certificate
- `Microsoft.VCLibs.x64.14.00.appx` — runtime dependency

#### Use a specific CI MSVC package

Main CI packages can include changes not yet attached to a release tag. Inspect
the selected run's exact `headSha` and successful status before downloading:

```bash
# Example: verified 2026-09-28 run; choose the run for your target commit.
XLLAMA_RUN_ID=36422945110
gh run view "$XLLAMA_RUN_ID" --repo gianlucamazza/xllama --json headSha,conclusion
gh run download "$XLLAMA_RUN_ID" --repo gianlucamazza/xllama \
  --name xllama-appx --dir "/tmp/xllama-ci-${XLLAMA_RUN_ID}"
# Locate the one MSIX under AppPackages, then use deploy.sh as in §2.
```

The unified `xllama-appx` artifact is the shipping/measurement path; see
[crossbuild-console.md](crossbuild-console.md). A higher revision with the same
identity updates in place and preserves LocalState. Verify `deploy.sh pfn` and
the running process package after install; WDP can briefly retain an older
registration alongside the updated package.

`install-latest-build.sh` performs a **fresh install**: it uninstalls registered
versions first and wipes their LocalState. Use it with backups/re-provisioning
when a fresh installation is intended. For profile-preserving updates, download
the chosen artifact and pass its MSIX directly to `deploy.sh`.

### 2. Install the certificate, dependency, and app

```bash
source ~/.config/xllama/xbox-env

# Trust cert (deploy.sh also auto-installs a .cer found next to the .msix)
./scripts/deploy.sh install-cert xllama-test.cer

# VCLibs dependency: install once via the Device Portal UI
#   https://<XBOX_IP>:11443 → My games & apps → Install → Microsoft.VCLibs.x64.14.00.appx
# (or add it as a dependency in the same WDP install dialog as the MSIX)

# App — derive the name from what you downloaded rather than typing a
# revision. The fourth component is a CI run stamp, so hard-coding one here
# would be wrong for every release but the one it was written for.
MSIX=$(ls -1 xllama_*_x64.msix | head -1)
./scripts/deploy.sh "$MSIX"
```

Notes:

- **Upgrades within the same identity**: a higher version installs over the
  old one and **preserves** LocalState (models, settings, history). WDP
  refuses a same-version install with different contents.
- **Any 1.5.x → 1.5.y is a normal in-place update**: same identity, LocalState
  (models, settings, history, KV snapshots) is preserved.
- ⚠️ **Upgrading from ≤1.4.x to 1.5.x is NOT an in-place update.** 1.5.0.0
  changed the package identity (`VenereLabs.xllama` → `GianlucaMazza.xllama`):
  it installs as a **new app**, and the old app's LocalState — downloaded
  models, chat history, settings, training output — does **not** carry over.
  Migration steps:
  1. Install 1.5.0.0 (this section) — the old app can stay during the switch.
  2. **Launch the new app once** (`./scripts/deploy.sh start-app`) — its
     LocalState does not exist until first launch, and model provisioning
     fails without it (`"File move failed"`).
  3. Re-provision models: in-app download, or
     `./scripts/provision-models.sh --all-test` from a host.
  4. When the new app works, uninstall the old `VenereLabs.xllama` package
     (Dev Home, or WDP `DELETE /api/app/packagemanager/package`).
- After (re)installs, check the **App type** in Dev Home (tile → View details):
  the measured performance figures assume the **Game** designation
  (`uwp-constraints.md §5`).

## First launch

Launch xllama from Dev Home (or `./scripts/deploy.sh start-app`). The app
downloads the default chat model with a progress bar, then opens the chat.
On **unified** shipping builds (v1.1.8+) that default is
**LFM2.5-350M** (~219 MB catalogue download); older releases used SmolLM2-360M
INT4 (~421 MB). See
[using-the-app.md](./using-the-app.md) from here.

For image generation, the SD-Turbo model (2.4 GB) is downloaded from the
catalogue on the first **Generate** — mind the Dev Mode disk budget (~2.2–2.5 GB
free by default, **expandable to 90 GB** via Dev Home → Manage Dev Storage; see
[uwp-constraints.md §9](uwp-constraints.md)). Device Portal provisioning remains available
([../diffusion/README.md](../diffusion/README.md), runbook §7).

## Troubleshooting

Known issues:

- **No MSIX on the release you opened.** `research-v*` tags are research
  packages and do not attach an app MSIX. Use the latest **app** tag
  (`v1.6.0.0` as of this writing).
- **Browser warns about the Device Portal certificate.** Expected. The
  console presents a self-signed TLS certificate; accept it on first visit
  ([device-portal.md](./device-portal.md)).
- **MSIX install is refused or the package is not trusted.** Install
  `xllama-test.cer` first. Packages must be signed, and that certificate
  must be trusted on the console before deployment
  ([device-portal.md](./device-portal.md)).
- **Install or first-launch model download fails with `0x80070070` (disk
  full).** The Developer Mode partition has ~2.2–2.5 GB free by default —
  remove unused models from LocalState, or raise the Developer Mode
  allocation (up to 90 GB, `uwp-constraints.md §9`).
- **Same-version install rejected.** Device Portal refuses a same-version
  package with different contents. Use a higher revision for an in-place
  update ([console-validation-runbook.md](./console-validation-runbook.md)).
- **Device Portal times out.** Device Portal can't reach a console while a
  game is running. Go back to Dev Home and try again
  ([device-portal.md](./device-portal.md), lifecycle gotchas).
- **Provisioning fails with `"File move failed"`.** LocalState does not
  exist until the app has launched once after a clean install
  ([device-portal.md](./device-portal.md)).
- **Coming from 1.4.x or earlier.** 1.5.0.0 changed the package identity
  (`VenereLabs.xllama` → `GianlucaMazza.xllama`). It installs as a new app;
  the old LocalState does not carry over. See the command-line notes above.
- **Measured tok/s look low.** Check **App type** in Dev Home (tile → View
  details). The published figures assume the **Game** designation
  (`uwp-constraints.md §5`).
- **Startup / logs (command-line host).** `./scripts/deploy.sh diagnose-startup`
  prints the process state, the app log, and any crash dumps.
  `./scripts/deploy.sh get-log` reads `xllama.log`; individual files:
  `./scripts/deploy.sh fetch-file "$(./scripts/deploy.sh pfn)" <name> <local-out>`.
