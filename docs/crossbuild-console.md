# Cross-building and deploying xllama from Linux

End-to-end path using **uwp-crossbuild** + **openappx** (no Windows VM).
Companion SSOTs: [uwp-constraints.md](uwp-constraints.md), campaign notes in
[phase15-re-opt.md](phase15-re-opt.md).

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
# From the xllama repo root
../uwp-crossbuild/scripts/build-project.sh \
  --project uwp/xllama.vcxproj \
  --out /tmp/xllama-layout \
  --property XllamaBackend=unified
```

Default is **static `/MT`** inside the AppContainer. It **links** filesystem-heavy
code cleanly, and `hello-uwp` launches with that CRT. **xllama’s larger static
pull** (registry / KERNEL32 / affinity imports from the static STL) still dies at
activation on Xbox (`0x8027025b`). Treat crossbuild packages as compile smoke
until that CRT surface is fixed upstream.

### Store `/MD` (opt-in experiment)

```bash
export UWP_STORE_CRT=1   # requires fetch-vclibs.sh output
# Manifest must declare Microsoft.VCLibs.140.00 (already in AppxManifest).
../uwp-crossbuild/scripts/build-project.sh ... --property XllamaBackend=unified
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

## Compile fixes specific to this tree

| Issue | Fix |
| --- | --- |
| `CACHE_LINE_SIZE` / clang-cl | `llama.cpp/ggml/src/ggml-cpu/ops.h` skips the feature-test under clang-cl |
| VCLibs missing | `PackageDependency` in `uwp/AppxManifest.xml` |

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
