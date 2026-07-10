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
      2. -Build: clone onnxruntime-genai @ rel-0.14.1 (pinned commit), apply
         patch, cmake --build against the restored NuGet ORT (ORT_HOME)
         (requires VS2022 + CMake + prior 'nuget restore'; full GenAI build — slow)

.PARAMETER Build
    Build the DLL from source. Ignores (and overwrites) a cached vendor DLL —
    the cache is only trusted on the install-only path.

.PARAMETER GenAiVersion
    onnxruntime-genai version to match uwp/packages.config (default 0.14.1).
    NB: upstream ships 0.14.1 as branch rel-0.14.1, not a tag.

.EXAMPLE
    ./scripts/vendor-genai-dml-patch.ps1
    ./scripts/vendor-genai-dml-patch.ps1 -Build
#>
param(
    [switch]$Build,
    [string]$GenAiVersion = "0.14.1"
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
    Write-Host @"
No patched DLL at:
  $VendorDll

Place a console-validated onnxruntime-genai.dll there (built from
microsoft/onnxruntime-genai branch rel-$GenAiVersion + patches/onnxruntime-genai-2280-dml-fallback.patch),
or re-run with -Build to compile from source.

Upstream: https://github.com/microsoft/onnxruntime-genai/pull/2280
"@
    exit 0
}

# --- Build from source -------------------------------------------------------
$WorkDir = Join-Path $RepoRoot "build/vendor-onnxruntime-genai"
$CloneDir = Join-Path $WorkDir "onnxruntime-genai"
# 0.14.1 exists upstream only as branch rel-0.14.1 (no v0.14.1 tag); pin the
# commit so a moving branch cannot change what we build.
$Branch = "rel-$GenAiVersion"
$PinnedCommit = "a30f479af016cb098688726831a9acbb8d19f0b2"

if (-not (Test-Path $CloneDir)) {
    New-Item -ItemType Directory -Path $WorkDir -Force | Out-Null
    git clone --depth 1 --branch $Branch https://github.com/microsoft/onnxruntime-genai.git $CloneDir
    if ($LASTEXITCODE -ne 0) {
        Write-Error "git clone of $Branch failed"
        exit 1
    }
}

Push-Location $CloneDir
$Head = git rev-parse HEAD
if ($Head -ne $PinnedCommit) {
    git fetch --depth 1 origin $PinnedCommit
    git checkout $PinnedCommit
    if ($LASTEXITCODE -ne 0) {
        Pop-Location
        Write-Error "Could not check out pinned commit $PinnedCommit (branch $Branch drifted?)"
        exit 1
    }
}

# Apply the patch, or accept a tree where it is already applied (reverse-check).
git apply --check $PatchFile 2>$null
if ($LASTEXITCODE -eq 0) {
    git apply $PatchFile
    if ($LASTEXITCODE -ne 0) {
        Pop-Location
        Write-Error "git apply failed for $PatchFile"
        exit 1
    }
} else {
    git apply --reverse --check $PatchFile 2>$null
    if ($LASTEXITCODE -ne 0) {
        Pop-Location
        Write-Error "Patch $PatchFile neither applies nor is already applied — refusing to build an unpatched DLL."
        exit 1
    }
    Write-Host "Patch already applied; continuing."
}
Pop-Location

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