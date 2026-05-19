#Requires -Version 5.1
<#
.SYNOPSIS
    Build the xllama UWP package for Xbox Series S|X.

.DESCRIPTION
    Invokes MSBuild on uwp/xllama.sln to produce an .appx package.
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

if (-not (Test-Path $SlnPath)) {
    Write-Error "Solution file not found: $SlnPath"
    exit 1
}

Write-Host "Building $Configuration|$Platform ..."

# Locate MSBuild via vswhere (ships with Visual Studio 2017+)
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

& $MsBuild $SlnPath `
    /p:Configuration=$Configuration `
    /p:Platform=$Platform `
    /p:AppxPackageSigningEnabled=false `
    /m `
    /nologo

if ($LASTEXITCODE -ne 0) {
    Write-Error "MSBuild failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}

Write-Host "Build succeeded."
$Appx = Get-ChildItem -Path (Join-Path $RepoRoot "uwp\AppPackages") -Filter "*.appx" -Recurse |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1

if ($Appx) {
    Write-Host "Package: $($Appx.FullName)"
} else {
    Write-Host "Package location: uwp\AppPackages\"
}
