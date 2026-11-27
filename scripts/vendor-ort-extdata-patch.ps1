#Requires -Version 5.1
<#
.SYNOPSIS
    Install a patched onnxruntime.dll (DirectML) that survives the AppContainer
    weakly_canonical walk in ValidateExternalDataPath, unblocking fp16 models with
    external .onnx.data >2 GB on the Xbox. See docs/uwp-constraints.md §8 and
    patches/onnxruntime-extdata-appcontainer.patch.

.DESCRIPTION
    Resolution order (mirrors scripts/vendor-genai-dml-patch.ps1):
      1. vendor/onnxruntime-patched/win-x64/onnxruntime.dll (pre-built) — installed
         over the NuGet copy.
      2. -Build: clone microsoft/onnxruntime @ v<OrtVersion>, apply the
         reviewed AppContainer patch to its verified commit, then build onnxruntime.dll (DirectML,
         Release x64) from source. This is a FULL ORT build — hours, needs the ORT
         build toolchain (VS2022, Python, CMake, protoc). Much heavier than the
         GenAI DLL build; run it in CI or a dedicated Windows box.

.PARAMETER Build
    Build from source instead of installing a cached vendor DLL.

.PARAMETER OrtVersion
    onnxruntime version to match uwp/packages.config (default read from the file).

.EXAMPLE
    ./scripts/vendor-ort-extdata-patch.ps1            # install cached vendor DLL
    ./scripts/vendor-ort-extdata-patch.ps1 -Build     # build from source
#>
param(
    [switch]$Build,
    [string]$OrtVersion = ""
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path $PSScriptRoot -Parent
$VendorDll = Join-Path $RepoRoot "vendor/onnxruntime-patched/win-x64/onnxruntime.dll"

if (-not $OrtVersion) {
    $OrtVersion = ([xml](Get-Content (Join-Path $RepoRoot "uwp/packages.config"))).packages.package |
        Where-Object { $_.id -eq "Microsoft.ML.OnnxRuntime.DirectML" } |
        Select-Object -ExpandProperty version
}
$NuGetDll = Join-Path $RepoRoot "uwp/packages/Microsoft.ML.OnnxRuntime.DirectML.$OrtVersion/runtimes/win-x64/native/onnxruntime.dll"

function Install-Dll([string]$Source) {
    if (-not (Test-Path $NuGetDll)) {
        Write-Error "NuGet DLL not found — run 'nuget restore uwp/xllama.sln' first: $NuGetDll"
        exit 1
    }
    Copy-Item $Source $NuGetDll -Force
    Write-Host "Installed patched onnxruntime.dll -> $NuGetDll"
}

if ((Test-Path $VendorDll) -and (-not $Build)) {
    Install-Dll $VendorDll
    exit 0
}

if (-not $Build) {
    Write-Error @"
No patched DLL at:
  $VendorDll

Place a console-validated onnxruntime.dll there (release vendor-dlls-v1 or
build-uwp-ort-patched artifact), or re-run with -Build to compile from source
(FULL ORT DirectML build — hours). Shipping CI downloads the pinned DLL before
calling this script.
"@
    exit 1
}

# --- Build from source -------------------------------------------------------
$WorkDir = Join-Path $RepoRoot "build/vendor-onnxruntime"
$CloneDir = Join-Path $WorkDir "onnxruntime"
$Tag = "v$OrtVersion"

if (-not (Test-Path $CloneDir)) {
    New-Item -ItemType Directory -Path $WorkDir -Force | Out-Null
    git clone --depth 1 --branch $Tag --recursive https://github.com/microsoft/onnxruntime.git $CloneDir
    if ($LASTEXITCODE -ne 0) { Write-Error "git clone of $Tag failed"; exit 1 }
}

$Submodules = @(git -C $CloneDir submodule status --recursive)
if ($LASTEXITCODE -ne 0 -or ($Submodules -match '^[+U-]')) {
    throw "ORT submodules must be initialized at their recorded commits."
}
# A reused checkout keeps the previous build/ (ignored by Git), which the
# verification refuses: start every build from source only.
$OldBuild = Join-Path $CloneDir "build"
if (Test-Path $OldBuild) { Remove-Item -Recurse -Force $OldBuild }
python (Join-Path $PSScriptRoot "prepare_ort_source.py") $CloneDir $OrtVersion
if ($LASTEXITCODE -ne 0) { throw "ORT source verification failed; see output above." }

Write-Host "Building onnxruntime.dll (DirectML, Release x64) from source — this is a FULL ORT build and may take hours ..."
Push-Location $CloneDir
python tools/ci_build/build.py `
    --config Release --parallel `
    --use_dml --build_shared_lib --skip_tests `
    --build_dir build
$rc = $LASTEXITCODE
Pop-Location
if ($rc -ne 0) { Write-Error "ORT build failed"; exit 1 }

$Built = Get-ChildItem -Path (Join-Path $CloneDir "build") -Filter "onnxruntime.dll" -Recurse |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $Built) { Write-Error "Build finished but onnxruntime.dll not found under $CloneDir/build"; exit 1 }

$vendorDir = Split-Path $VendorDll -Parent
New-Item -ItemType Directory -Path $vendorDir -Force | Out-Null
Copy-Item $Built.FullName $VendorDll -Force
Write-Host "Cached patched DLL at $VendorDll"
Install-Dll $VendorDll
