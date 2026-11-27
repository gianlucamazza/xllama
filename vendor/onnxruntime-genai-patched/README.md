# Patched onnxruntime-genai.dll (#2280)

Patched **`onnxruntime-genai.dll`** — DML Agility `CreateDevice` fallback for
XAML hosts (`887A0036`). Upstream
[microsoft/onnxruntime-genai#2280](https://github.com/microsoft/onnxruntime-genai/pull/2280)
**merged on `main` 2026-07-13**; **not** in NuGet GenAI **0.14.1**.

Patched by xllama; not an official ONNX Runtime build. MIT License, Copyright (c) Microsoft Corporation. — see `LICENSES/onnxruntime-genai-LICENSE`.

The DLL binary lives at `win-x64/onnxruntime-genai.dll` and is **gitignored** —
only this README + `SHA256SUMS` are tracked. Shipping CI downloads the pinned
DLL from the [`vendor-dlls-v1`](https://github.com/gianlucamazza/xllama/releases/tag/vendor-dlls-v1)
release (hash must match `SHA256SUMS`).

## Producing / refreshing the DLL

- **CI rebuild:** `gh workflow run build-uwp-patched.yml` uploads
  `onnxruntime-genai-patched-dll`. After re-validation, update the
  `vendor-dlls-v1` release asset and `SHA256SUMS`.
- **Local:** `./scripts/vendor-genai-dml-patch.ps1 -Build` (VS2022 + NuGet
  restore + Python; GenAI branch `rel-0.14.1` pin `a30f479` +
  `patches/onnxruntime-genai-2280-dml-fallback.patch`). `-SourceUrl` (or the
  `source_url` input of `build-uwp-patched.yml`) points the fetch at a mirror.

### Source pin and verification

The pin `a30f479af016cb098688726831a9acbb8d19f0b2` comes from the
[official 0.14.1 NuGet manifest](https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntimegenai.directml/0.14.1/microsoft.ml.onnxruntimegenai.directml.nuspec)
(checked 2026-11-27). That nuspec has no `<repository commit=...>` element; its
`<releaseNotes>` record `Branch: refs/heads/rel-0.14.1` and
`Commit: a30f479af016cb098688726831a9acbb8d19f0b2`. The pin lives once, next to
the version, in `scripts/prepare_genai_source.py`;
`tests/test_prepare_genai_source.py` checks it against `uwp/packages.config`
(and against nuget.org with `XLLAMA_ONLINE_NUSPEC=1`).

Before building, `prepare_genai_source.py` (logic in `scripts/vendor_source.py`)
requires HEAD to be the pin and the working tree to equal the pinned tree, or
the pinned tree with the reviewed patch applied (computed in a temporary
index). It refuses untracked files, ignored files (a previous `build/`; the
script removes its own first), `assume-unchanged` or `skip-worktree` entries
and staged changes, then applies the patch and prints the base commit and the
patched tree ID. Preserve both with rebuild logs. There is no text-substitution
fallback: a patch that does not apply to the pin fails the build.

**Upstream source unavailable (2026-11-27).**
`github.com/microsoft/onnxruntime-genai` returns 404 (since about 2026-11-17,
with no notice; see xllama #646), so the default fetch fails closed and no
rebuild can run from upstream. The shipped DLL, `SHA256SUMS` and the NuGet
package are unaffected. A mirror or fork is acceptable only when the rebuild
from it passes `prepare_genai_source.py` unchanged; that check, not the URL,
is what binds the content to the pin. Keep the existing DLL and digest until a
candidate also passes Windows packaging and the XAML + DirectML console smoke.

## Install over NuGet (local or CI)

```powershell
# Cached vendor path (after download or -Build):
./scripts/vendor-genai-dml-patch.ps1

# Or via packaging:
./scripts/build-uwp.ps1 -Backend unified -PatchedGenAI -PatchedOrt
```

Pin: ORT GenAI **0.14.1** + #2280. **Remove this vendor step** when Microsoft
ships #2280 in an official NuGet package (track ROADMAP “drop PatchedGenAI”).
