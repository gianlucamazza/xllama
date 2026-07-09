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
      2. -Build: clone onnxruntime-genai @ v0.14.1, apply patch, cmake --build
         (requires VS2022 + CMake; full GenAI build — slow)

.PARAMETER Build
    Build the DLL from source when no pre-built copy exists.

.PARAMETER GenAiVersion
    onnxruntime-genai tag to match uwp/packages.config (default 0.14.1).

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

if (Test-Path $VendorDll) {
    Install-Dll $VendorDll
    exit 0
}

if (-not $Build) {
    Write-Host @"
No patched DLL at:
  $VendorDll

Place a console-validated onnxruntime-genai.dll there (built from
microsoft/onnxruntime-genai tag v$GenAiVersion + patches/onnxruntime-genai-2280-dml-fallback.patch),
or re-run with -Build to compile from source.

Upstream: https://github.com/microsoft/onnxruntime-genai/pull/2280
"@
    exit 0
}

# --- Build from source -------------------------------------------------------
$WorkDir = Join-Path $RepoRoot "build/vendor-onnxruntime-genai"
$CloneDir = Join-Path $WorkDir "onnxruntime-genai"
$Tag = "v$GenAiVersion"

if (-not (Test-Path $CloneDir)) {
    New-Item -ItemType Directory -Path $WorkDir -Force | Out-Null
    git clone --depth 1 --branch $Tag https://github.com/microsoft/onnxruntime-genai.git $CloneDir
} else {
    Push-Location $CloneDir
    git fetch --depth 1 origin tag $Tag 2>$null
    git checkout $Tag
    Pop-Location
}

Push-Location $CloneDir
git apply --check $PatchFile 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Warning "Patch may already be applied; continuing."
} else {
    git apply $PatchFile
}
Pop-Location

Write-Host "Building onnxruntime-genai (DirectML, Release x64) — this may take several minutes ..."
Push-Location $CloneDir
cmake --preset dml_vs2022 2>$null
if ($LASTEXITCODE -ne 0) {
    cmake -B build -G "Visual Studio 17 2022" -A x64 -DUSE_DML=ON -DCMAKE_BUILD_TYPE=Release
}
cmake --build build --config Release --target onnxruntime-genai -j
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