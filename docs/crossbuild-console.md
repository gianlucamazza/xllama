# Cross-building and deploying xllama from Linux

End-to-end path using **uwp-crossbuild** + **openappx** (no Windows VM).
Companion SSOTs: [uwp-constraints.md](uwp-constraints.md), campaign notes in
[phase15-re-opt.md](phase15-re-opt.md).

**Currency:** 2026-08-08.

### One-page launch decision

| Goal | Path | Status on Series S |
| --- | --- | --- |
| Ship / measure product tok/s | CI MSVC `build-uwp` → `xllama-appx` → openappx pack/sign/deploy | **Launches** (proven through `1.5.2.864`+ post-gpubw) |
| Compile + package from Linux | uwp-crossbuild ≥ **0.5.0** + store `/MD` + dual-CRT stage | **Links; audit PASS**; dual-CRT stage required; **store PE still fails activation** (`0x80040904`) — layer 2 |
| hello-uwp / non-filesystem samples | uwp-crossbuild `/MT` or store `/MD` | Launches |

## Prerequisites

- `uwp-crossbuild` installed or checkout with `scripts/` on `PATH`
- `openappx` ≥ 0.6.3 (`pipx install 'openappx>=0.6.3'`)
- xwin splat + SDK: see uwp-crossbuild README (`fetch-sdk.sh`)
- Device: `~/.config/xllama/xbox-env` and/or `~/.config/uwp-crossbuild/device-env`
- Signing: PFX whose subject equals `Identity/@Publisher` in
  `uwp/AppxManifest.xml` (`CN=xllama-dev` for the dev identity)

## Launchable package today (shipping path)

**CI MSVC (`build-uwp` → artifact `xllama-appx`) is the proven product path on
Series S.** Measured 2026-08-07: CI `1.5.2.910` activates and loads GGUF.

## Architecture: two integrated runtimes (SSOT)

Shipping CI and Linux crossbuild are **not** “same MSIX, different compiler”.
They are two assembly lines that must reproduce the **same dual-CRT product
shape** documented in `scripts/build-uwp.ps1` and `uwp/xllama.vcxproj`.

```
┌─────────────────────────────────────────────────────────────────┐
│  Package (AppContainer)                                         │
│                                                                 │
│  xllama.exe  ──/MD APP CRT──►  MSVCP140_APP / VCRUNTIME140_APP  │
│       │                        (Microsoft.VCLibs framework dep) │
│       │ hard-import                                             │
│       ▼                                                         │
│  onnxruntime.dll + onnxruntime-genai.dll                        │
│       │  built desktop /MD                                      │
│       └──import MSVCP140.dll / VCRUNTIME140.dll                 │
│              ▲                                                  │
│              └── must be APP-LOCAL in package root              │
│                  (System32 not searchable in AppContainer)      │
│                  build-uwp.ps1 copies from VS CRT redist;       │
│                  vcxproj ships them via DeploymentContent       │
└─────────────────────────────────────────────────────────────────┘
```

| Layer | CI MSVC (`build-uwp.ps1`) | Linux crossbuild today |
| --- | --- | --- |
| EXE RuntimeLibrary | `/MD` → **APP CRT** (UWP toolset) | `/MD` + `UWP_STORE_CRT=1` → **APP CRT** imports |
| EXE family | `WINAPI_FAMILY=APP` (VS) | `--uwp` → `WINAPI_FAMILY=APP` |
| Static ggml | same solution, APP family | `--static-lib --uwp` when `AppContainerApplication` |
| Forbidden Win32 | not in PE | `pe-import-audit` + partition guards |
| ORT/GenAI DLLs | NuGet + optional **vendor patched** pins | NuGet paths from `packages/` (may differ binary/size) |
| **Desktop CRT for ORT** | **Copied into package** (MSVCP140*.dll, VCRUNTIME140*.dll) | **Missing** unless files already exist under `uwp/` (`Exists()` gate) |
| CppWinRT | NuGet **2.0.240405.15** | xwin projection **2.0.250303.1** (version skew) |
| Measured launch | **OK** | install OK; activate **fail** |

### Hybrid experiments (evidence)

| PE | Package | Result | Interpretation |
| --- | --- | --- | --- |
| CI | CI (incl. desktop CRT + ORT) | **Launch** | Full product shape |
| CI | store (no desktop CRT; different ORT/genai sizes) | `0x8027025b` | Loader cannot satisfy ORT’s desktop CRT / binary set |
| store | CI (desktop CRT present) | `0x80040904` | PE/activation path still wrong after ORT can load |
| store | store | `0x8027025b` | Layer-1 + layer-2 combined |

So the gap is **not** a single banlist symbol. It is **integration**:

1. **Package dual-CRT (layer 1)** — architectural requirement already in the
   vcxproj/build-uwp.ps1; crossbuild layout does not assemble it.
2. **Toolchain PE/projection (layer 2)** — store PE fails even beside a CI
   package (`0x80040904`): CppWinRT skew, entry/CRT startup, or factory/winmd
   identity. Needs dumps or a same-version projection, not import shims.

### What is architecture vs debt

| Practice | Class |
| --- | --- |
| `WINAPI_FAMILY=APP`, static-lib family, partition guards, pe-import-audit | Architecture |
| App-local desktop CRT next to desktop-built ORT | Architecture (CI SSOT) |
| Store `/MD` + `*_app` import libs for the **EXE** | Architecture |
| Extracting **desktop** `msvcprt` MD objs for `__std_fs_*` | Incomplete: MD yes, UWP filesystem surface no |
| CreateFileW→CreateFile2 import shims | **Debt — do not ship** |
| Dual `XLLAMA_UWP \|\| PARTITION` guards | **Debt — removed** from patch |
| Packaging Wine desktop CRT as product | **Forbidden** |

### Crossbuild compile contract (still required)

1. **Compile family = APP** (`--uwp`); static libs with
   `AppContainerApplication=true` get `--static-lib --uwp`.
2. **Guards = partition only** (`patches/0001-…`).
3. **Gate = pe-import-audit** (registry/affinity/desktop CRT **on the EXE**).
4. **Subsystem 6.02**; MTA held through `Application::Start` (gotcha 17).
5. **Layout must still match CI dual-CRT + vendor DLL pins** before claiming
   launch parity.

```bash
# Prefer: push a PR / main and download the workflow artifact, or:
gh run download <run-id> -n xllama-appx -D /tmp/xllama-ci-art
# then openappx sign + deploy (or scripts/deploy.sh / install-latest-build.sh)
```


## Linux packaging: dual-CRT stage (required for ORT)

uwp-crossbuild **0.5.0** audits the **EXE** only. After `build-project`, always:

```bash
export UWP_STORE_CRT=1
uwp-build-project --project uwp/xllama.vcxproj --out /tmp/xllama-layout \
  --property XllamaBackend=unified
# or: ./scripts/crossbuild-uwp.sh --out /tmp/xllama-layout

./scripts/stage-ort-desktop-crt.sh --layout /tmp/xllama-layout
# Source: XLLAMA_DESKTOP_CRT_DIR, uwp/, or ~/.cache/xllama/desktop-crt/
# (VC143 redist or DLLs from a CI package — never Wine stubs)
```

Fail closed if `onnxruntime.dll` is in the layout but any of
`MSVCP140.dll` / `MSVCP140_1.dll` / `VCRUNTIME140.dll` / `VCRUNTIME140_1.dll`
is missing.

**Layer 2 (open, 2026-08-08):** with dual-CRT + ORT pins + C++/WinRT pin
aligned to NuGet **2.0.240405.15** (`ensure-cppwinrt-pin.sh` +
`UWP_CPPWINRT_INCLUDE` / `UWP_CPPWINRT_EXE`), a **crossbuild** PE still fails
as **`0x80040904`**. CI PE in the same package **launches**. Remaining gap is
compiler/link (clang-cl + lld vs MSVC), not package dual-CRT or projection
version string.

```bash
# C++/WinRT pin (optional but recommended for CI parity of generated code)
source ./scripts/ensure-cppwinrt-pin.sh   # sets UWP_CPPWINRT_*
./scripts/crossbuild-uwp.sh --out /tmp/xllama-layout
```

## Build from Linux (uwp-crossbuild)

```bash
# 1) AppContainer guards (WINAPI_FAMILY partition) on the llama.cpp submodule
./scripts/apply-uwp-patches.sh

# 2) From the xllama repo root, with uwp-crossbuild on PATH (or full path to
#    its checkout). Prefer a checkout that sets WINAPI_FAMILY=APP for --uwp.
build-project \
  --project uwp/xllama.vcxproj \
  --out /tmp/xllama-layout \
  --property XllamaBackend=unified

# 3) Fail closed on AppContainer-forbidden imports before pack/deploy
~/Workspace/tooling/uwp-crossbuild/scripts/pe-import-audit.sh \
  /tmp/xllama-layout/xllama.exe
```

(`build-project` is provided by the **uwp-crossbuild** tooling tree, not by a
script under this repository.)

For **product-shaped** packages (ORT linked), use store `/MD` for the EXE
(`UWP_STORE_CRT=1` + `fetch-vclibs` + `gen-msvcprt-app-static`) **and** still
ship app-local **desktop** CRT next to ORT (same files CI copies — not Wine
substitutes; not a replacement for VCLibs). Default `/MT` remains fine for
hello-uwp-class apps without desktop-`/MD` native deps.

### Store `/MD` for the EXE (not a substitute for ORT dual-CRT)

```bash
export UWP_STORE_CRT=1   # requires fetch-vclibs.sh + msvcprt_app_static
# Manifest PackageDependency Microsoft.VCLibs.140.00 (AppxManifest).
# Layout must still include MSVCP140*.dll / VCRUNTIME140*.dll for ORT
# (vcxproj DeploymentContent; on Linux provide those files like build-uwp.ps1).
```

Never mix MT `libcpmt` into `/MD` (gotcha 21). Never treat Wine desktop CRT as
a product path.

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
| Desktop family compile (old toolchain) | PE imports registry/affinity → `0x8027025b` | uwp-crossbuild sets `WINAPI_FAMILY=APP`; apply AppContainer patch |
| Forbidden imports (registry / affinity) | `0x8027025b` after install | `./scripts/apply-uwp-patches.sh` + `pe-import-audit.sh` |
| Product launch (proven) | — | **CI MSVC**; re-gate crossbuild after APP-family rebuild + audit + device start |

### Pre-deploy import audit (uwp-crossbuild)

```bash
# After build-project produces a layout:
~/Workspace/tooling/uwp-crossbuild/scripts/pe-import-audit.sh /tmp/xllama-layout/xllama.exe
# Fail closed on RegOpenKey* / SetThreadAffinityMask / desktop CRT DLLs.
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
