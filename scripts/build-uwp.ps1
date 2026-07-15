#Requires -Version 5.1
<#
.SYNOPSIS
    Build the xllama UWP package for Xbox Series S|X.

.DESCRIPTION
    Invokes MSBuild on uwp/xllama.sln to produce a signed .msix package.
    Generates a self-signed test certificate if none is provided.
    Requires Visual Studio 2022 with "Universal Windows Platform development" workload.

.PARAMETER Configuration
    Build configuration. Default: Release

.PARAMETER Platform
    Target platform. Default: x64

.EXAMPLE
    ./scripts/build-uwp.ps1
    ./scripts/build-uwp.ps1 -Configuration Debug -Platform x64
#>

param(
    [string]$Configuration  = "Release",
    [string]$Platform       = "x64",
    [switch]$ForceNewCert   = $false,
    # 'ort' (default): ORT GenAI only. 'llamacpp': ggml/llama CPU backend only
    # (XLLAMA_USE_ORT absent). 'unified': BOTH backends compiled, chosen at
    # runtime per model (Qwen3.5/LFM2 GGUF via llama.cpp, rest via ORT).
    # 'llamacpp'/'unified' require the submodule + scripts/apply-uwp-patches.sh.
    [ValidateSet("ort", "llamacpp", "unified")]
    [string]$Backend        = "ort",
    # Replace onnxruntime-genai.dll with the #2280 DML fallback build when available
    # (vendor/onnxruntime-genai-patched/ or -Build via vendor-genai-dml-patch.ps1).
    [switch]$PatchedGenAI   = $false,
    # Replace onnxruntime.dll with the AppContainer external-data build when
    # available (vendor/onnxruntime-patched/ or vendor-ort-extdata-patch.ps1 -Build).
    # Shipping CI downloads the pinned DLL from the vendor-dlls-v1 release.
    [switch]$PatchedOrt     = $false,
    # MSIX version Revision (the 4th component). Each CI build passes a unique,
    # monotonic value (the workflow run number) so every package gets a distinct,
    # increasing version — in-place console updates never hit the same-identity
    # block. 0 (the local default) leaves the committed X.Y.Z.0 unchanged, so a
    # local build never pollutes the working tree.
    [int]$BuildRevision     = $(if ($env:GITHUB_RUN_NUMBER) { [int]$env:GITHUB_RUN_NUMBER } else { 0 })
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path $PSScriptRoot -Parent
$SlnPath  = Join-Path $RepoRoot "uwp\xllama.sln"
$PfxPath  = Join-Path $RepoRoot "uwp\xllama-test.pfx"
$CerPath  = Join-Path $RepoRoot "uwp\xllama-test.cer"
$CertPwd  = "xllama-test"

if (-not $IsWindows) {
    Write-Error "UWP packaging requires Windows with Visual Studio 2022, the UWP workload, and Windows SDK tools."
    exit 1
}

if (-not (Test-Path $SlnPath)) {
    Write-Error "Solution file not found: $SlnPath"
    exit 1
}

# ---------------------------------------------------------------------------
# Certificate handling: reuse existing cert unless -ForceNewCert is passed.
# This avoids reinstalling the trust cert on the console every build.
# ---------------------------------------------------------------------------
$cert = $null
if ((-not $ForceNewCert) -and (Test-Path $PfxPath) -and (Test-Path $CerPath)) {
    Write-Host "Reusing existing test certificate ..."
    $cert = Get-PfxCertificate -FilePath $PfxPath -Password (ConvertTo-SecureString -String $CertPwd -Force -AsPlainText) -ErrorAction SilentlyContinue
    if (-not $cert) {
        Write-Warning "Failed to load existing PFX; generating a new certificate."
    }
}

if (-not $cert) {
    Write-Host "Generating self-signed test certificate ..."
    $cert = New-SelfSignedCertificate `
        -Type Custom `
        -Subject "CN=xllama-dev" `
        -KeyUsage DigitalSignature `
        -FriendlyName "xllama test cert" `
        -CertStoreLocation "Cert:\CurrentUser\My" `
        -TextExtension @("2.5.29.37={text}1.3.6.1.5.5.7.3.3", "2.5.29.19={text}")

    $pwd = ConvertTo-SecureString -String $CertPwd -Force -AsPlainText
    Export-PfxCertificate -Cert "Cert:\CurrentUser\My\$($cert.Thumbprint)" `
        -FilePath $PfxPath -Password $pwd | Out-Null
    Export-Certificate -Cert "Cert:\CurrentUser\My\$($cert.Thumbprint)" `
        -FilePath $CerPath | Out-Null
}

Write-Host "Certificate thumbprint: $($cert.Thumbprint)"
Write-Host "PFX: $PfxPath"
Write-Host "CER: $CerPath"

# ---------------------------------------------------------------------------
# Locate MSBuild
# ---------------------------------------------------------------------------
$VsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $VsWhere)) {
    Write-Error "vswhere.exe not found. Install Visual Studio 2022 with UWP workload."
    exit 1
}

$MsBuild = & $VsWhere -latest -requires Microsoft.Component.MSBuild `
    -find MSBuild\**\Bin\MSBuild.exe | Select-Object -First 1

if (-not $MsBuild) {
    Write-Error "MSBuild not found via vswhere."
    exit 1
}

Write-Host "Using MSBuild: $MsBuild"

# ---------------------------------------------------------------------------
# Copy VC++ CRT DLLs for onnxruntime AppContainer compatibility.
# onnxruntime.dll and onnxruntime-genai.dll are built with the desktop /MD
# runtime and import MSVCP140.dll (not MSVCP140_APP.dll). On Xbox AppContainer,
# System32 is not in the loader search path for these DLLs, so they must be
# bundled app-locally.
# ---------------------------------------------------------------------------
$UwpDir = Join-Path $RepoRoot "uwp"
$CrtDlls = @("MSVCP140.dll", "MSVCP140_1.dll", "VCRUNTIME140.dll", "VCRUNTIME140_1.dll")
$CrtRedistDir = Get-ChildItem `
    -Path "${env:ProgramFiles}\Microsoft Visual Studio\2022\*\VC\Redist\MSVC\*\x64\Microsoft.VC143.CRT" `
    -ErrorAction SilentlyContinue |
    Sort-Object FullName -Descending | Select-Object -First 1

if ($CrtRedistDir) {
    Write-Host "Found CRT redist: $($CrtRedistDir.FullName)"
    foreach ($dll in $CrtDlls) {
        $src = Join-Path $CrtRedistDir.FullName $dll
        if (Test-Path $src) {
            Copy-Item $src $UwpDir -Force
            Write-Host "  Copied $dll"
        }
    }
} else {
    Write-Warning "VC++ CRT redist directory not found. onnxruntime may fail in AppContainer on Xbox."
}

# ---------------------------------------------------------------------------
# NuGet restore (packages.config in uwp\ → restores to uwp\packages\)
# ---------------------------------------------------------------------------
Write-Host "Restoring NuGet packages ..."
nuget restore $SlnPath

if ($PatchedGenAI -and $Backend -ne "llamacpp") {
    $VendorScript = Join-Path $RepoRoot "scripts/vendor-genai-dml-patch.ps1"
    Write-Host "Installing patched onnxruntime-genai.dll (#2280) ..."
    & $VendorScript
    if ($LASTEXITCODE -ne 0) {
        Write-Error "vendor-genai-dml-patch.ps1 failed ($LASTEXITCODE) — refusing to package a mislabeled patched build."
        exit $LASTEXITCODE
    }
}

if ($PatchedOrt -and $Backend -ne "llamacpp") {
    $VendorOrtScript = Join-Path $RepoRoot "scripts/vendor-ort-extdata-patch.ps1"
    Write-Host "Installing patched onnxruntime.dll (AppContainer extdata) ..."
    & $VendorOrtScript
    if ($LASTEXITCODE -ne 0) {
        Write-Error "vendor-ort-extdata-patch.ps1 failed ($LASTEXITCODE) — refusing to package a mislabeled patched build."
        exit $LASTEXITCODE
    }
    # Fail closed: -PatchedOrt requires a real install, not the "no DLL" advisory exit 0.
    $OrtVer = ([xml](Get-Content (Join-Path $RepoRoot "uwp/packages.config"))).packages.package |
        Where-Object { $_.id -eq "Microsoft.ML.OnnxRuntime.DirectML" } |
        Select-Object -ExpandProperty version
    $VendorOrt = Join-Path $RepoRoot "vendor/onnxruntime-patched/win-x64/onnxruntime.dll"
    $NuGetOrt = Join-Path $RepoRoot "uwp/packages/Microsoft.ML.OnnxRuntime.DirectML.$OrtVer/runtimes/win-x64/native/onnxruntime.dll"
    if (-not (Test-Path $VendorOrt)) {
        Write-Error "PatchedOrt requested but vendor DLL missing: $VendorOrt (download vendor-dlls-v1 or run -Build)."
        exit 1
    }
    if ((Get-FileHash $VendorOrt).Hash -ne (Get-FileHash $NuGetOrt).Hash) {
        Write-Error "PatchedOrt: NuGet onnxruntime.dll does not match vendor cache after install."
        exit 1
    }
}

# ---------------------------------------------------------------------------
# Version stamping. Major.Minor.Build (the semantic version) stays the source of
# truth in the committed AppxManifest.xml; the Revision (4th component) is set to
# $BuildRevision so every CI build is uniquely and monotonically versioned. This
# is what lets the console take an in-place update without a manual version bump.
# The negative-lookbehind avoids the TargetDeviceFamily Min/MaxVersion attributes;
# only the first (Identity) Version is rewritten.
# ---------------------------------------------------------------------------
if ($BuildRevision -gt 0) {
    $ManifestPath = Join-Path $UwpDir "AppxManifest.xml"
    $manifestText = Get-Content -Raw $ManifestPath
    $rx = [regex]'(?<!\w)Version="(\d+)\.(\d+)\.(\d+)\.\d+"'
    $newText = $rx.Replace($manifestText, {
        param($m)
        'Version="{0}.{1}.{2}.{3}"' -f $m.Groups[1].Value, $m.Groups[2].Value, $m.Groups[3].Value, $BuildRevision
    }, 1)
    Set-Content -Path $ManifestPath -Value $newText -NoNewline
    $stamped = ([regex]::Match($newText, '(?<!\w)Version="(\d+\.\d+\.\d+\.\d+)"')).Groups[1].Value
    Write-Host "Version stamped: $stamped (revision = build $BuildRevision)"
}

Write-Host "Building $Configuration|$Platform ..."

# ---------------------------------------------------------------------------
# Build + sign
# ---------------------------------------------------------------------------
$MsBuildArgs = @(
    $SlnPath,
    "/p:Configuration=$Configuration",
    "/p:Platform=$Platform",
    "/p:AppxPackageSigningEnabled=true",
    "/p:PackageCertificateKeyFile=$PfxPath",
    "/p:PackageCertificatePassword=$CertPwd",
    "/p:PackageCertificateThumbprint=$($cert.Thumbprint)",
    "/m",
    "/nologo"
)
if ($Backend -ne "ort") {
    Write-Host "Backend: $Backend (links the static ggml/llama lib)"
    $MsBuildArgs += "/p:XllamaBackend=$Backend"
    # Pre-build the static ggml/llama lib explicitly: the solution maps it
    # ActiveCfg-only (no Build.0 — the ORT variants must not compile it, the
    # submodule may be absent there), so the solution build resolves the
    # ProjectReference path but does not build the lib itself (LNK1181).
    $GgmlProj = Join-Path $RepoRoot "uwp/ggml-uwp.vcxproj"
    Write-Host "Pre-building ggml-uwp.vcxproj ($Configuration|$Platform) ..."
    & $MsBuild $GgmlProj "/p:Configuration=$Configuration" "/p:Platform=$Platform" "/m" "/nologo"
    if ($LASTEXITCODE -ne 0) {
        Write-Error "ggml-uwp build failed ($LASTEXITCODE)"
        exit $LASTEXITCODE
    }
}

$buildExitCode = 0
try {
    & $MsBuild @MsBuildArgs
    $buildExitCode = $LASTEXITCODE
} finally {
}

if ($buildExitCode -ne 0) {
    Write-Error "MSBuild failed with exit code $buildExitCode"
    exit $buildExitCode
}

Write-Host "Build succeeded."
$Msix = Get-ChildItem -Path (Join-Path $RepoRoot "uwp\AppPackages") -Filter "*.msix" -Recurse |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1

if (-not $Msix) {
    Write-Host "Package location: uwp\AppPackages\"
    exit 0
}

Write-Host "Package: $($Msix.FullName)"
