# Cross-building and deploying xllama from Linux

End-to-end path using **uwp-crossbuild** + **openappx** (no Windows VM).
Companion SSOTs: [uwp-constraints.md](uwp-constraints.md), campaign notes in
[phase15-re-opt.md](phase15-re-opt.md).

**Currency:** 2026-08-08.

### One-page launch decision

| Goal | Path | Status on Series S |
| --- | --- | --- |
| Ship / measure product tok/s | CI MSVC `build-uwp` → `xllama-appx` → openappx pack/sign/deploy | **Launches** (proven `1.5.2.910`+; W2 A/B on `1.5.2.920`) |
| Compile + audit from Linux | uwp-crossbuild `build-project` (APP family + static-lib family) → `pe-import-audit` | **Links; audit PASS**; install OK; activation still `0x8027025b` (`/MT` vs CI APP CRT — see below) |
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

**Linux crossbuild contract (architecture, 2026-08-08):**

1. **Compile family = APP.** uwp-crossbuild `--uwp` sets
   `WINAPI_FAMILY=WINAPI_FAMILY_APP` (VS parity). Static libs with
   `AppContainerApplication=true` (e.g. `ggml-uwp`) get `--static-lib --uwp`
   so they compile under the same family (link flags stay image-only).
2. **Source guards = partition only.**
   `patches/0001-uwp-appcontainer-guards.patch` uses
   `WINAPI_FAMILY_PARTITION(DESKTOP)` alone — no dual `XLLAMA_UWP` workaround.
3. **Gate.** `pe-import-audit` fails closed on registry/affinity/desktop CRT.
   Measured: APP-family crossbuild PE **passes** audit (no banlist symbols).
4. **Store `/MD` links (APP CRT).** `UWP_STORE_CRT=1` + `fetch-vclibs` +
   `gen-msvcprt-app-static.sh` (MD `.obj` members from `msvcprt.lib` for
   `__std_fs_*` / find / `_Facet_Register`) produces a PE that imports
   `msvcp140_app` / `vcruntime140_app` / `api-ms-win-crt-*` like CI, and
   **passes `pe-import-audit`**.
5. **Entry / PE knobs.** `/subsystem:windows,6.02` (VS UWP); `wWinMain` keeps
   the MTA (`RoInitialize` + `init_apartment`, no `uninit` before
   `Application::Start` — gotcha 17).
6. **Remaining launch gap.** Audited store-MD packages still activate as
   **`0x8027025b`** on Series S while a CI MSVC PE of the same app launches
   (control reconfirmed). Hybrid PE swaps show the failure tracks the
   **crossbuild executable**, not only package DLLs. Product launch path
   remains **CI MSVC** until that residual PE gap is closed.

```bash
# Prefer: push a PR / main and download the workflow artifact, or:
gh run download <run-id> -n xllama-appx -D /tmp/xllama-ci-art
# then openappx sign + deploy (or scripts/deploy.sh / install-latest-build.sh)
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

Default is **static `/MT`** inside the AppContainer (hello-uwp launches with
that CRT). Store `/MD` is opt-in (`UWP_STORE_CRT=1` + `fetch-vclibs`) only when
the app is fully covered by the `*_app` import libs — xllama’s
`std::filesystem` helpers live in `libcpmt` (MT) and must not be mixed into MD
(uwp-crossbuild gotcha 21).

### Store `/MD` (opt-in experiment)

```bash
export UWP_STORE_CRT=1   # requires fetch-vclibs.sh output
# Manifest must declare Microsoft.VCLibs.140.00 (already in AppxManifest).
# Same uwp-crossbuild build-project entry as above, with store CRT env set.
```

xllama needs `__std_fs_*`, `__std_find_*`, `_Thrd_sleep_for`, etc. that live in
`libcpmt` (MT) and are **not** exports of `msvcp140_app.dll`. Blind
`libcpmt` last → LLD `FAILIFMISMATCH`. Do **not** sanitize MT archives into an
MD image — that is technical debt, not a shipping path. Default remains `/MT`
in the container until a real APP-CRT solution covers filesystem helpers.

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
