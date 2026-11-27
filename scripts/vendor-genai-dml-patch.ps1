#Requires -Version 5.1
<#
.SYNOPSIS
    Install a patched onnxruntime-genai.dll (microsoft/onnxruntime-genai#2280) over the NuGet copy.

.DESCRIPTION
    The vanilla NuGet DLL throws 887A0036 when ORT GenAI DML initializes inside a
    XAML host. PR #2280 falls back to the system D3D12 runtime on Agility
    CreateDevice failure — validated on Xbox Series S.

    Resolution order:
      1. vendor/onnxruntime-genai-patched/win-x64/onnxruntime-genai.dll (pre-built)
      2. -Build: fetch the pinned rel-0.14.1 commit from -SourceUrl, verify the
         checkout with scripts/prepare_genai_source.py (pinned commit and tree,
         reviewed patch applied), cmake --build against the restored NuGet ORT
         (ORT_HOME) (requires VS2022 + CMake + Python + prior 'nuget restore';
         full GenAI build — slow)

.PARAMETER Build
    Build the DLL from source. Ignores (and overwrites) a cached vendor DLL —
    the cache is only trusted on the install-only path.

.PARAMETER GenAiVersion
    onnxruntime-genai version to match uwp/packages.config (default 0.14.1).
    NB: upstream ships 0.14.1 as branch rel-0.14.1, not a tag.

.PARAMETER SourceUrl
    Git URL to fetch the pinned GenAI commit from (default: upstream
    microsoft/onnxruntime-genai, which returns 404 since about 2026-11-17).
    A mirror is acceptable: the build runs only if prepare_genai_source.py
    verifies the pinned commit and the reviewed tree.

.EXAMPLE
    ./scripts/vendor-genai-dml-patch.ps1
    ./scripts/vendor-genai-dml-patch.ps1 -Build
    ./scripts/vendor-genai-dml-patch.ps1 -Build -SourceUrl https://example.org/onnxruntime-genai.git
#>
param(
    [switch]$Build,
    [string]$GenAiVersion = "0.14.1",
    [string]$SourceUrl = "https://github.com/microsoft/onnxruntime-genai.git"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path $PSScriptRoot -Parent
$PatchFile = Join-Path $RepoRoot "patches/onnxruntime-genai-2280-dml-fallback.patch"
$VendorDll = Join-Path $RepoRoot "vendor/onnxruntime-genai-patched/win-x64/onnxruntime-genai.dll"
$NuGetDll = Join-Path $RepoRoot "uwp/packages/Microsoft.ML.OnnxRuntimeGenAI.DirectML.$GenAiVersion/runtimes/win-x64/native/onnxruntime-genai.dll"

if (-not (Test-Path $PatchFile)) {
    Write-Error "Patch not found: $PatchFile"
    exit 1
}

function Install-Dll([string]$Source) {
    if (-not (Test-Path $NuGetDll)) {
        Write-Error "NuGet DLL not found — run 'nuget restore uwp/xllama.sln' first: $NuGetDll"
        exit 1
    }
    $destDir = Split-Path $NuGetDll -Parent
    if (-not (Test-Path $destDir)) {
        New-Item -ItemType Directory -Path $destDir -Force | Out-Null
    }
    Copy-Item $Source $NuGetDll -Force
    Write-Host "Installed patched onnxruntime-genai.dll -> $NuGetDll"
}

if ((Test-Path $VendorDll) -and (-not $Build)) {
    Install-Dll $VendorDll
    exit 0
}

if (-not $Build) {
    # Fail closed (mirrors vendor-ort-extdata-patch.ps1): shipping CI always
    # downloads the pin before calling this script; silent exit would leave the
    # vanilla NuGet DLL and pass -PatchedGenAI packaging checks incorrectly.
    Write-Error @"
No patched DLL at:
  $VendorDll

Place a console-validated onnxruntime-genai.dll there (vendor-dlls-v1 pin or
built from microsoft/onnxruntime-genai branch rel-$GenAiVersion +
patches/onnxruntime-genai-2280-dml-fallback.patch), or re-run with -Build.

Upstream: https://github.com/microsoft/onnxruntime-genai/pull/2280
"@
    exit 1
}

# --- Build from source -------------------------------------------------------
$WorkDir = Join-Path $RepoRoot "build/vendor-onnxruntime-genai"
$CloneDir = Join-Path $WorkDir "onnxruntime-genai"
$Prepare = Join-Path $PSScriptRoot "prepare_genai_source.py"
# 0.14.1 exists upstream only as branch rel-0.14.1 (no v0.14.1 tag). The pin is
# the commit in the NuGet nuspec, kept once in prepare_genai_source.py.
$Branch = "rel-$GenAiVersion"
$PinnedCommit = (python $Prepare --pin).Trim()
if ($LASTEXITCODE -ne 0 -or $PinnedCommit -notmatch '^[0-9a-f]{40}$') {
    Write-Error "Could not read the GenAI source pin from $Prepare"
    exit 1
}

# Fetch the pinned commit from $SourceUrl (upstream or a mirror). The source is
# trusted only after prepare_genai_source.py verifies the commit and tree.
if (-not (Test-Path $CloneDir)) {
    New-Item -ItemType Directory -Path $CloneDir -Force | Out-Null
    git -C $CloneDir init --quiet
    if ($LASTEXITCODE -ne 0) { Write-Error "git init failed in $CloneDir"; exit 1 }
}
$Head = git -C $CloneDir rev-parse --verify --quiet HEAD
if ($Head -ne $PinnedCommit) {
    git -C $CloneDir fetch --depth 1 $SourceUrl $PinnedCommit
    if ($LASTEXITCODE -ne 0) {
        # Some mirrors refuse fetch-by-SHA; take the release branch and find the pin in it.
        git -C $CloneDir fetch $SourceUrl "refs/heads/${Branch}"
    }
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Could not fetch $PinnedCommit or branch $Branch from $SourceUrl"
        exit 1
    }
    git -C $CloneDir checkout --quiet --detach $PinnedCommit
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Could not check out pinned commit $PinnedCommit from $SourceUrl"
        exit 1
    }
}

# Submodule gitlinks are part of the verified tree; their checkouts must match them.
git -C $CloneDir submodule update --init --recursive
if ($LASTEXITCODE -ne 0) { Write-Error "GenAI submodule checkout failed"; exit 1 }
$Submodules = @(git -C $CloneDir submodule status --recursive)
if ($LASTEXITCODE -ne 0 -or ($Submodules -match '^[+U-]')) {
    Write-Error "GenAI submodules must be initialized at their recorded commits."
    exit 1
}
# A reused checkout keeps the previous build/ (ignored by Git), which the
# verification refuses: start every build from source only.
$OldBuild = Join-Path $CloneDir "build"
if (Test-Path $OldBuild) { Remove-Item -Recurse -Force $OldBuild }
# Refuses untracked, ignored or hidden edits and any tree other than the pin or
# the pin + the reviewed patch; applies the patch when the tree is the pin.
python $Prepare $CloneDir $GenAiVersion
if ($LASTEXITCODE -ne 0) {
    Write-Error "GenAI source verification failed; see output above."
    exit 1
}

# ORT_HOME staged from the restored NuGet ORT (uwp/packages.config), so the DLL
# links the exact onnxruntime the MSIX ships. Without ORT_HOME the GenAI build
# downloads a nightly ORT (1.25-dev) — ABI mismatch with 1.24.4 at runtime.
$OrtVersion = ([xml](Get-Content (Join-Path $RepoRoot "uwp/packages.config"))).packages.package |
    Where-Object { $_.id -eq "Microsoft.ML.OnnxRuntime.DirectML" } | Select-Object -ExpandProperty version
$OrtPkg = Join-Path $RepoRoot "uwp/packages/Microsoft.ML.OnnxRuntime.DirectML.$OrtVersion"
if (-not (Test-Path $OrtPkg)) {
    Write-Error "ORT NuGet package not restored: $OrtPkg — run 'nuget restore' in uwp/ first."
    exit 1
}
$OrtHome = Join-Path $WorkDir "ort-home"
New-Item -ItemType Directory -Path (Join-Path $OrtHome "include") -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $OrtHome "lib") -Force | Out-Null
Copy-Item (Join-Path $OrtPkg "build/native/include/*") (Join-Path $OrtHome "include") -Recurse -Force
Copy-Item (Join-Path $OrtPkg "runtimes/win-x64/native/*") (Join-Path $OrtHome "lib") -Force

Write-Host "Building onnxruntime-genai (DirectML, Release x64, ORT $OrtVersion) — this may take several minutes ..."
Push-Location $CloneDir
cmake -B build -G "Visual Studio 17 2022" -A x64 `
    -DUSE_DML=ON -DUSE_CUDA=OFF `
    -DENABLE_PYTHON=OFF -DENABLE_TESTS=OFF -DENABLE_MODEL_BENCHMARK=OFF `
    -DORT_HOME="$OrtHome"
if ($LASTEXITCODE -ne 0) {
    Pop-Location
    Write-Error "cmake configure failed"
    exit 1
}
cmake --build build --config Release --target onnxruntime-genai -j
if ($LASTEXITCODE -ne 0) {
    Pop-Location
    Write-Error "cmake build failed"
    exit 1
}
Pop-Location

$Built = Get-ChildItem -Path (Join-Path $CloneDir "build") -Filter "onnxruntime-genai.dll" -Recurse |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $Built) {
    Write-Error "Build finished but onnxruntime-genai.dll not found under $CloneDir/build"
    exit 1
}

$vendorDir = Split-Path $VendorDll -Parent
New-Item -ItemType Directory -Path $vendorDir -Force | Out-Null
Copy-Item $Built.FullName $VendorDll -Force
Write-Host "Cached patched DLL at $VendorDll"
Install-Dll $VendorDll