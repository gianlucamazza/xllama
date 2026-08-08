# Cross-building and deploying xllama from Linux

End-to-end path using **uwp-crossbuild** + **openappx** (no Windows VM).
Companion SSOTs: [uwp-constraints.md](uwp-constraints.md), campaign notes in
[phase15-re-opt.md](phase15-re-opt.md).

**Currency:** 2026-08-08.

### One-page launch decision

| Goal | Path | Status on Series S |
| --- | --- | --- |
| Ship / measure product tok/s | CI MSVC `build-uwp` → `xllama-appx` → openappx pack/sign/deploy | **Launches** (proven `1.5.2.910`+; W2 A/B on `1.5.2.920`) |
| Compile-smoke from Linux | uwp-crossbuild `build-project` → openappx | **Installs, does not activate** (`0x8027025b`) |
| hello-uwp / non-filesystem samples | uwp-crossbuild `/MT` or store `/MD` | Launches (not a proof for xllama) |

## Prerequisites

- `uwp-crossbuild` installed or checkout with `scripts/` on `PATH`
- `openappx` ≥ 0.6.3 (`pipx install 'openappx>=0.6.3'`)
- xwin splat + SDK: see uwp-crossbuild README (`fetch-sdk.sh`)
- Device: `~/.config/xllama/xbox-env` and/or `~/.config/uwp-crossbuild/device-env`
- Signing: PFX whose subject equals `Identity/@Publisher` in
  `uwp/AppxManifest.xml` (`CN=xllama-dev` for the dev identity)

## Launchable package today (shipping path)

**CI MSVC (`build-uwp` → artifact `xllama-appx`) is the path that launches on
Series S.** Measured 2026-08-07: CI `1.5.2.910` activates and loads GGUF;
Linux crossbuilt packages install but fail activation with **`0x8027025b`**
(openappx/WDP “Failed to launch the application”).

```bash
# Prefer: push a PR / main and download the workflow artifact, or:
gh run download <run-id> -n xllama-appx -D /tmp/xllama-ci-art
# then openappx sign + deploy (or scripts/deploy.sh / install-latest-build.sh)
```

## Build from Linux (uwp-crossbuild) — links, does not yet launch xllama

```bash
# From the xllama repo root, with uwp-crossbuild available on PATH (or invoked
# from its checkout). The helper is named build-project and takes:
#   --project uwp/xllama.vcxproj
#   --out /tmp/xllama-layout
#   --property XllamaBackend=unified
build-project \
  --project uwp/xllama.vcxproj \
  --out /tmp/xllama-layout \
  --property XllamaBackend=unified
```

(`build-project` is provided by the **uwp-crossbuild** tooling tree, not by a
script under this repository.)

Default is **static `/MT`** inside the AppContainer. It **links** filesystem-heavy
code cleanly, and `hello-uwp` launches with that CRT. **xllama’s larger static
pull** (registry / KERNEL32 / affinity imports from the static STL) still dies at
activation on Xbox (`0x8027025b`). Treat crossbuild packages as compile smoke
until that CRT surface is fixed upstream.

### Store `/MD` (opt-in experiment)

```bash
export UWP_STORE_CRT=1   # requires fetch-vclibs.sh output
# Manifest must declare Microsoft.VCLibs.140.00 (already in AppxManifest).
# Same uwp-crossbuild build-project entry as above, with store CRT env set.
```

xllama needs `__std_fs_*`, `__std_find_*`, `_Thrd_sleep_for`, etc. that live in
`libcpmt` (MT) and are **not** exports of `msvcp140_app.dll`. Blind
`libcpmt` last → LLD `FAILIFMISMATCH`. A sanitized `libcpmt` (`.drectve` /
RuntimeLibrary directives blanked) + a tiny `std::_Facet_Register` stub **does
link** a store-MD image that imports `msvcp140_app` / `vcruntime140_app`, but the
same package still fails activation (`0x8027025b`) — residual static objects keep
`RegOpenKeyExA` / `SetThreadAffinityMask` / `KERNEL32` imports that CI’s MSVC APP
CRT image does not have.

**Never** package Wine desktop `MSVCP140.dll` / `VCRUNTIME140.dll` as a product
path. Small samples without `std::filesystem` (hello-uwp) are fine on store `/MD`.

## Pack, sign, deploy

```bash
# Bump Version if an equal-or-higher package is already installed
# (edit layout AppxManifest Identity/@Version, e.g. 1.5.2.902)

openappx pack --root /tmp/xllama-layout --out /tmp/xllama.msix
openappx sign --package /tmp/xllama.msix --pfx ~/.config/uwp-crossbuild/dev.pfx
# or mint: openappx sign --make-test-cert 'CN=xllama-dev' --cert-out /tmp/xllama-dev-cert

source ~/.config/xllama/xbox-env
openappx deploy --device "https://${XBOX_IP}:${XBOX_PORT}" \
  --user "$XBOX_USER" --password "$XBOX_PASS" --insecure \
  --install-cert /tmp/xllama-dev-cert.cer   # once per cert
openappx deploy --device "https://${XBOX_IP}:${XBOX_PORT}" \
  --user "$XBOX_USER" --password "$XBOX_PASS" --insecure \
  --package /tmp/xllama.msix
```

## Compile / link / launch matrix (xllama-specific)

| Issue | Symptom | Fix / status |
| --- | --- | --- |
| `CACHE_LINE_SIZE` under clang-cl | compile fail in ggml | `llama.cpp/ggml/src/ggml-cpu/ops.h` skips feature-test when clang-cl |
| VCLibs not declared | install OK, launch `0x80070002` | `PackageDependency` Microsoft.VCLibs.140.00 in AppxManifest |
| Store `/MD` + desktop CRT | launch fail (desktop `VCRUNTIME140`) | `UWP_STORE_CRT=1` + `fetch-vclibs` `*_app` libs (gotcha 19) |
| Store `/MD` + raw `libcpmt` | LLD `FAILIFMISMATCH` RuntimeLibrary | never mix MT archive into MD (gotcha 21) |
| Store `/MD` + sanitized libcpmt + Facet stub | links APP CRT but still `0x8027025b` | residual registry/KERNEL32 imports from static STL objects |
| App-container `/MT` (default crossbuild) | links; xllama `0x8027025b` | enough for hello-uwp; not product path for xllama |
| Product launch | — | **CI MSVC** until crossbuild PE passes `pe-import-audit` + device launch |
| Forbidden imports (registry / affinity) | `0x8027025b` after install | Apply `./scripts/apply-uwp-patches.sh`; `XLLAMA_UWP=1` on ggml-uwp; audit with `uwp-crossbuild/scripts/pe-import-audit.sh` |

### Pre-deploy import audit (uwp-crossbuild)

```bash
# After build-project produces a layout:
~/Workspace/tooling/uwp-crossbuild/scripts/pe-import-audit.sh /tmp/xllama-layout/xllama.exe
# Fail closed on RegOpenKey* / SetThreadAffinityMask / desktop CRT DLLs.
# CI MSVC images pass; a PE missing UWP guards fails before you deploy.
```

## W2 console A/B

```bash
source ~/.config/xllama/xbox-env
# Model must exist under LocalState\models\<id>\
# Package must be a CI MSVC build that includes W2 (see Launchable package).
./scripts/bench-spec-w2-console.sh qwen25-coder-3b --runs 4 --n-predict 128
# or lower tier if 3b is not provisioned:
./scripts/bench-spec-w2-console.sh qwen25-coder-1.5b --runs 3 --n-predict 96
```

Gate (declared): ≥1.4× decode on **code** regime for `qwen25-coder-3b`;
chat ≥0.98×; peak &lt; 3.5 GB.

**Measured 2026-08-07 (M3):** code **1.04× FAIL**, chat 0.99× PASS, peak ~2.0 GB
(`bench/results/phase15-spec-w2-console.csv`). Product default stays off.
