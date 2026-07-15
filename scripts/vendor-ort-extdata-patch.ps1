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
         AppContainer guard (git apply, else a context-tolerant in-place transform),
         verify the change is present, then build onnxruntime.dll (DirectML,
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
$PatchFile = Join-Path $RepoRoot "patches/onnxruntime-extdata-appcontainer.patch"
$VendorDll = Join-Path $RepoRoot "vendor/onnxruntime-patched/win-x64/onnxruntime.dll"

if (-not $OrtVersion) {
    $OrtVersion = ([xml](Get-Content (Join-Path $RepoRoot "uwp/packages.config"))).packages.package |
        Where-Object { $_.id -eq "Microsoft.ML.OnnxRuntime.DirectML" } |
        Select-Object -ExpandProperty version
}
$NuGetDll = Join-Path $RepoRoot "uwp/packages/Microsoft.ML.OnnxRuntime.DirectML.$OrtVersion/runtimes/win-x64/native/onnxruntime.dll"
$TargetFile = "onnxruntime/core/framework/tensorprotoutils.cc"
$TargetFile2 = "onnxruntime/core/platform/windows/env.cc"

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
    Write-Host @"
No patched DLL at:
  $VendorDll

Place a console-validated onnxruntime.dll there (built from microsoft/onnxruntime
tag v$OrtVersion + patches/onnxruntime-extdata-appcontainer.patch), or re-run with
-Build to compile from source (FULL ORT DirectML build — slow).
"@
    exit 0
}

# --- Apply the two AppContainer external-data fixes --------------------------
# (1) tensorprotoutils.cc: weakly_canonical guard (helper injected before
#     ValidateExternalDataPath + rewrite of the one-arg calls).
# (2) env.cc: shrink the ReadFileIntoBuffer chunk 1 GB -> 16 MB (errcode 1450).
# Context-tolerant: prefers `git apply` (both files), else per-file in-place edits.
function Apply-Guard([string]$CloneDir) {
    $abs = Join-Path $CloneDir $TargetFile
    $abs2 = Join-Path $CloneDir $TargetFile2
    if (-not (Test-Path $abs)) { Write-Error "Not found: $abs"; exit 1 }
    if (-not (Test-Path $abs2)) { Write-Error "Not found: $abs2"; exit 1 }
    $content = Get-Content $abs -Raw
    $content2 = Get-Content $abs2 -Raw

    if (($content -match "XllamaSafeCanonical") -and ($content2 -match "1 << 24")) {
        Write-Host "Both fixes already present."
        return
    }

    # Prefer the reference git patch (fast path when context matches; both files).
    Push-Location $CloneDir
    git apply --check $PatchFile 2>$null
    $gitOk = ($LASTEXITCODE -eq 0)
    if ($gitOk) { git apply $PatchFile }
    Pop-Location
    if ($gitOk) { Write-Host "Applied via git apply."; return }

    Write-Host "git apply rejected the reference diff (context drift) — using in-place transforms."

    # --- Fix 2: env.cc ReadFileIntoBuffer chunk size (idempotent) ---
    if ($content2 -match "1 << 24") {
        Write-Host "env.cc chunk-size fix already present."
    } elseif ($content2 -match "k_max_bytes_to_read = 1 << 30") {
        $content2 = $content2 -replace "k_max_bytes_to_read = 1 << 30", "k_max_bytes_to_read = 1 << 24"
        Set-Content -Path $abs2 -Value $content2 -NoNewline
        Write-Host "env.cc: k_max_bytes_to_read 1 GB -> 16 MB."
    } else {
        Write-Error "env.cc: k_max_bytes_to_read = 1 << 30 not found — regenerate the patch by hand."
        exit 1
    }

    # --- Fix 1: tensorprotoutils.cc weakly_canonical guard ---
    if ($content -match "XllamaSafeCanonical") { Write-Host "weakly_canonical guard already present."; return }

    # 1. Rewrite the one-arg weakly_canonical() calls FIRST (before injecting the
    #    helper, so the helper's own two-arg call is not rewritten).
    $content = $content -replace 'std::filesystem::weakly_canonical\(', 'XllamaSafeCanonical('

    # 2. Inject the helper immediately before the function definition.
    $helper = @'
// xllama (AppContainer): std::filesystem::weakly_canonical() walks path segments
// from the drive root and throws ACCESS_DENIED on the inaccessible
// Q:\Users\UserMgr0 segment (docs/uwp-constraints.md §8), crashing model load.
// Use the std::error_code overload and fall back to a purely lexical normalization
// when the walk is denied — the external-data file is a trusted local file.
static std::filesystem::path XllamaSafeCanonical(const std::filesystem::path& p) {
  std::error_code ec;
  std::filesystem::path r = std::filesystem::weakly_canonical(p, ec);
  if (ec || r.empty()) {
    return p.lexically_normal();
  }
  return r;
}

Status ValidateExternalDataPath(
'@
    if ($content -notmatch [regex]::Escape("Status ValidateExternalDataPath(")) {
        Write-Error "Could not locate ValidateExternalDataPath in $TargetFile — regenerate the patch by hand."
        exit 1
    }
    $content = $content -replace [regex]::Escape("Status ValidateExternalDataPath("), $helper
    Set-Content -Path $abs -Value $content -NoNewline
    Write-Host "Applied via in-place transform."
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

Apply-Guard $CloneDir

# Verify BOTH fixes are really in place before the (expensive) build.
$verify = Get-Content (Join-Path $CloneDir $TargetFile) -Raw
if ($verify -notmatch "XllamaSafeCanonical") {
    Write-Error "weakly_canonical guard not present after apply — refusing to build an unpatched DLL."
    exit 1
}
$verify2 = Get-Content (Join-Path $CloneDir $TargetFile2) -Raw
if ($verify2 -notmatch "k_max_bytes_to_read = 1 << 24") {
    Write-Error "env.cc chunk-size fix not present after apply — refusing to build an unpatched DLL."
    exit 1
}

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
