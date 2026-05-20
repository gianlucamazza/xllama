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
    [string]$Platform      = "x64",
    [switch]$ForceNewCert  = $false
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
# NuGet restore (packages.config in uwp\ → restores to uwp\packages\)
# ---------------------------------------------------------------------------
Write-Host "Restoring NuGet packages ..."
nuget restore $SlnPath

Write-Host "Building $Configuration|$Platform ..."

# ---------------------------------------------------------------------------
# Create empty XBF stubs before MSBuild.
# MarkupCompilePass2 is disabled (WMC9999 crash), so App.xbf / MainPage.xbf
# are never generated. AppXPackage.Targets:1638 expects them in the output dir.
# The XAML runtime never reads these stubs: InitializeComponent calls
# LoadComponent with the .xaml URI, not the .xbf path.
# ---------------------------------------------------------------------------
$XbfDir = Join-Path $RepoRoot "uwp\$Platform\$Configuration\xllama"
New-Item -ItemType Directory -Force -Path $XbfDir | Out-Null
[System.IO.File]::WriteAllBytes("$XbfDir\App.xbf",      [byte[]]@())
[System.IO.File]::WriteAllBytes("$XbfDir\MainPage.xbf", [byte[]]@())
Write-Host "Created empty XBF stubs in $XbfDir"

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

# ---------------------------------------------------------------------------
# Inject XAML text files into the MSIX.
#
# MSBuild's AppX packaging pipeline handles XAML items via the XBF/PRI path
# (_GenerateProjectPriConfigurationFiles → resources.pri). The text XAML files
# are NOT added as loose package files by the recipe generator, even when staged
# in $(OutDir). We post-process the MSIX to inject them so that
# LoadComponent(ms-appx:///MainPage.xaml) resolves at runtime on Xbox.
# ---------------------------------------------------------------------------
Write-Host "Injecting XAML text files into package ..."

$WinSdkBin = "${env:ProgramFiles(x86)}\Windows Kits\10\bin"
$MakeAppxExe = Get-ChildItem "$WinSdkBin" -Filter "MakeAppx.exe" -Recurse |
    Where-Object { $_.DirectoryName -like "*\x64" } |
    Sort-Object FullName -Descending | Select-Object -First 1 -ExpandProperty FullName
$SignToolExe = Get-ChildItem "$WinSdkBin" -Filter "signtool.exe" -Recurse |
    Where-Object { $_.DirectoryName -like "*\x64" } |
    Sort-Object FullName -Descending | Select-Object -First 1 -ExpandProperty FullName

if (-not $MakeAppxExe) { Write-Error "MakeAppx.exe not found in Windows SDK"; exit 1 }
if (-not $SignToolExe)  { Write-Error "signtool.exe not found in Windows SDK"; exit 1 }

$UnpackDir = Join-Path ([System.IO.Path]::GetTempPath()) "xllama_unpack_$(Get-Random)"
$TmpMsix   = ($Msix.FullName -replace '\.msix$', '_tmp.msix')

try {
    # Unpack (disable signature validation so we can unpack a signed package)
    & $MakeAppxExe unpack /p $Msix.FullName /d $UnpackDir /nv /o
    if ($LASTEXITCODE -ne 0) { throw "MakeAppx unpack failed (exit $LASTEXITCODE)" }

    # Inject source XAML so LoadComponent(ms-appx:///MainPage.xaml) resolves.
    # App.xaml: strip x:Class before injection — the XAML runtime calls
    # IXamlMetadataProvider::GetXamlType("xllama.App") when x:Class is present;
    # our no-op stub returns null, causing E_XAMLPARSEFAILED (0x802B000A) on Xbox.
    # x:Class is kept in the source file for MarkupCompilePass1 (generates App.g.h).
    $AppXamlSrc = Get-Content (Join-Path $RepoRoot "uwp\App.xaml") -Raw
    $AppXamlRuntime = $AppXamlSrc -replace '\s*x:Class="[^"]*"', ''
    Set-Content -Path "$UnpackDir\App.xaml" -Value $AppXamlRuntime -Encoding UTF8 -NoNewline
    # MainPage.xaml: strip x:Class for the same reason as App.xaml.
    # LoadComponent(*this, ms-appx:///MainPage.xaml) passes the existing instance so
    # the XAML parser does not need to call GetXamlType to create a new object.
    # Without stripping, the XAML runtime calls GetXamlType("xllama.MainPage") during
    # app startup (before OnLaunched) and gets nullptr from our stub -> E_XAMLPARSEFAILED.
    $MainPageXamlSrc = Get-Content (Join-Path $RepoRoot "uwp\MainPage.xaml") -Raw
    $MainPageXamlRuntime = $MainPageXamlSrc -replace '\s*x:Class="[^"]*"', ''
    Set-Content -Path "$UnpackDir\MainPage.xaml" -Value $MainPageXamlRuntime -Encoding UTF8 -NoNewline
    Write-Host "  App.xaml + MainPage.xaml (both x:Class stripped) added to layout"

    # Repack (unsigned)
    & $MakeAppxExe pack /d $UnpackDir /p $TmpMsix /h sha256 /o
    if ($LASTEXITCODE -ne 0) { throw "MakeAppx pack failed (exit $LASTEXITCODE)" }

    # Re-sign with the same test certificate used by MSBuild
    & $SignToolExe sign /fd SHA256 /f $PfxPath /p $CertPwd $TmpMsix
    if ($LASTEXITCODE -ne 0) { throw "SignTool sign failed (exit $LASTEXITCODE)" }

    # Atomic replace
    Move-Item $TmpMsix $Msix.FullName -Force
    Write-Host "XAML injection complete. Final package: $($Msix.FullName)"
} finally {
    if (Test-Path $UnpackDir) { Remove-Item $UnpackDir -Recurse -Force -ErrorAction SilentlyContinue }
    if (Test-Path $TmpMsix)   { Remove-Item $TmpMsix   -Force         -ErrorAction SilentlyContinue }
}
