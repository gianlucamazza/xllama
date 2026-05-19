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
    [string]$Configuration = "Release",
    [string]$Platform      = "x64"
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path $PSScriptRoot -Parent
$SlnPath  = Join-Path $RepoRoot "uwp\xllama.sln"
$PfxPath  = Join-Path $RepoRoot "uwp\xllama-test.pfx"
$CerPath  = Join-Path $RepoRoot "uwp\xllama-test.cer"
$CertPwd  = "xllama-test"

if (-not (Test-Path $SlnPath)) {
    Write-Error "Solution file not found: $SlnPath"
    exit 1
}

# ---------------------------------------------------------------------------
# Generate a self-signed test certificate (once per build; not committed).
# Xbox Dev Mode requires the package to be signed; the .cer is installed on
# the console via Device Portal before deploying the .msix.
# ---------------------------------------------------------------------------
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
# NuGet restore (packages.config in uwp\ → restores to uwp\packages\)
# ---------------------------------------------------------------------------
Write-Host "Restoring NuGet packages ..."
nuget restore $SlnPath

Write-Host "Building $Configuration|$Platform ..."

# ---------------------------------------------------------------------------
# Build + sign
# ---------------------------------------------------------------------------
$buildExitCode = 0
try {
    & $MsBuild $SlnPath `
        /p:Configuration=$Configuration `
        /p:Platform=$Platform `
        /p:AppxPackageSigningEnabled=true `
        /p:PackageCertificateKeyFile="$PfxPath" `
        /p:PackageCertificatePassword="$CertPwd" `
        /p:PackageCertificateThumbprint="$($cert.Thumbprint)" `
        /m `
        /nologo
    $buildExitCode = $LASTEXITCODE
} finally {
    # $(IntDir) = xllama\$(Platform)\$(Configuration)\ relative to uwp\
    $IntDir = "xllama\$Platform\$Configuration"

    Write-Host "`n=== DIAGNOSTIC: NuGet packages restored ==="
    $pkgDir = Join-Path $RepoRoot "uwp\packages"
    if (Test-Path $pkgDir) {
        Get-ChildItem $pkgDir -Directory | Select-Object -ExpandProperty Name
    } else {
        Write-Host "(uwp\packages\ not found)"
    }

    Write-Host "`n=== DIAGNOSTIC: cppwinrt reference RSP ==="
    $rsp = Join-Path $RepoRoot "uwp\$IntDir\xllama.vcxproj.cppwinrt_ref.rsp"
    if (Test-Path $rsp) { Get-Content $rsp } else { Write-Host "(rsp not found at $rsp)" }

    Write-Host "`n=== DIAGNOSTIC: Generated Files\ root (component stubs) ==="
    $genDir = Join-Path $RepoRoot "uwp\Generated Files"
    if (Test-Path $genDir) {
        Get-ChildItem $genDir -File | Select-Object -ExpandProperty Name
    } else {
        Write-Host "(uwp\Generated Files\ not found)"
    }

    Write-Host "`n=== DIAGNOSTIC: cppwinrt output locations ==="
    @(
        "uwp\$IntDir\winrt",
        "uwp\$IntDir\Generated Files\winrt",
        "uwp\Generated Files\winrt"
    ) | ForEach-Object {
        $d = Join-Path $RepoRoot $_
        if (Test-Path $d) {
            $n = (Get-ChildItem $d -Filter "*.h" -ErrorAction SilentlyContinue | Measure-Object).Count
            Write-Host "[$d] — $n .h files"
        } else {
            Write-Host "[$d] — not found"
        }
    }

    Write-Host "`n=== DIAGNOSTIC: search App.g.cpp under uwp\ ==="
    $found = Get-ChildItem -Path (Join-Path $RepoRoot "uwp") -Filter "App.g.cpp" -Recurse -ErrorAction SilentlyContinue
    if ($found) {
        $found | Select-Object -ExpandProperty FullName | ForEach-Object { Write-Host $_ }
    } else {
        Write-Host "(App.g.cpp not found under uwp\)"
    }
}

if ($buildExitCode -ne 0) {
    Write-Error "MSBuild failed with exit code $buildExitCode"
    exit $buildExitCode
}

Write-Host "Build succeeded."
$Msix = Get-ChildItem -Path (Join-Path $RepoRoot "uwp\AppPackages") -Filter "*.msix" -Recurse |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1

if ($Msix) {
    Write-Host "Package: $($Msix.FullName)"
} else {
    Write-Host "Package location: uwp\AppPackages\"
}
