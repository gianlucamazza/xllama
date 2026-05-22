#Requires -Version 5.1
<#
.SYNOPSIS
    Prepare or verify a Windows VM for xllama UWP/MSIX builds.

.DESCRIPTION
    Installs Visual Studio 2022 Build Tools with the UWP workload through winget
    when requested, then verifies the tools required by scripts/build-uwp.ps1.
    Run from an elevated PowerShell session inside the Windows VM.

.PARAMETER Install
    Install missing Visual Studio Build Tools components with winget.

.PARAMETER VsEdition
    Visual Studio edition to verify. Defaults to BuildTools.

.EXAMPLE
    ./scripts/setup-windows-uwp-dev.ps1
    ./scripts/setup-windows-uwp-dev.ps1 -Install
#>

param(
    [switch]$Install = $false,
    [ValidateSet("BuildTools", "Community", "Professional", "Enterprise")]
    [string]$VsEdition = "BuildTools"
)

$ErrorActionPreference = "Stop"

if (-not $IsWindows) {
    Write-Error "This script must run inside the Windows UWP build VM."
    exit 1
}

$RequiredComponents = @(
    "Microsoft.VisualStudio.Workload.UniversalBuildTools",
    "Microsoft.VisualStudio.Workload.VCTools",
    "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
    "Microsoft.VisualStudio.Component.Windows10SDK.22621"
)

function Require-Command {
    param([string]$Name)
    $cmd = Get-Command $Name -ErrorAction SilentlyContinue
    if (-not $cmd) {
        throw "Required command not found: $Name"
    }
    return $cmd.Source
}

function Get-VsWhere {
    $path = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $path)) {
        return $null
    }
    return $path
}

function Install-BuildTools {
    Require-Command winget | Out-Null
    $packageId = if ($VsEdition -eq "BuildTools") {
        "Microsoft.VisualStudio.2022.BuildTools"
    } else {
        "Microsoft.VisualStudio.2022.$VsEdition"
    }

    $overrideArgs = @(
        "--quiet",
        "--wait",
        "--norestart",
        "--nocache"
    )
    foreach ($component in $RequiredComponents) {
        $overrideArgs += @("--add", $component)
    }

    Write-Host "Installing ${packageId} with UWP/MSVC components ..."
    winget install --id $packageId --exact --source winget --accept-package-agreements --accept-source-agreements --override ($overrideArgs -join " ")
}

if ($Install) {
    Install-BuildTools
}

$VsWhere = Get-VsWhere
if (-not $VsWhere) {
    Write-Error "vswhere.exe not found. Run with -Install or install Visual Studio 2022 with UWP tools manually."
    exit 1
}

$VsInstall = & $VsWhere -latest -products * -requires Microsoft.Component.MSBuild -property installationPath
if (-not $VsInstall) {
    Write-Error "No Visual Studio instance with MSBuild found."
    exit 1
}

$MsBuild = & $VsWhere -latest -products * -requires Microsoft.Component.MSBuild -find "MSBuild\**\Bin\MSBuild.exe" | Select-Object -First 1
if (-not $MsBuild) {
    Write-Error "MSBuild.exe not found via vswhere."
    exit 1
}

$WinSdkBin = Join-Path ${env:ProgramFiles(x86)} "Windows Kits\10\bin"
$MakeAppxExe = Get-ChildItem $WinSdkBin -Filter "MakeAppx.exe" -Recurse -ErrorAction SilentlyContinue |
    Where-Object { $_.DirectoryName -like "*\x64" } |
    Sort-Object FullName -Descending |
    Select-Object -First 1 -ExpandProperty FullName
$SignToolExe = Get-ChildItem $WinSdkBin -Filter "signtool.exe" -Recurse -ErrorAction SilentlyContinue |
    Where-Object { $_.DirectoryName -like "*\x64" } |
    Sort-Object FullName -Descending |
    Select-Object -First 1 -ExpandProperty FullName

if (-not $MakeAppxExe) {
    Write-Error "MakeAppx.exe not found. Install Windows SDK 10.0.22621 or the UWP workload."
    exit 1
}
if (-not $SignToolExe) {
    Write-Error "signtool.exe not found. Install Windows SDK 10.0.22621 or the UWP workload."
    exit 1
}

$NuGet = Get-Command nuget -ErrorAction SilentlyContinue
if (-not $NuGet) {
    Write-Warning "nuget.exe not found in PATH. Visual Studio may still restore packages, but scripts/build-uwp.ps1 calls nuget directly."
}

Write-Host "xllama Windows UWP environment:"
Write-Host "  Visual Studio: $VsInstall"
Write-Host "  MSBuild:       $MsBuild"
Write-Host "  MakeAppx:      $MakeAppxExe"
Write-Host "  SignTool:      $SignToolExe"
if ($NuGet) {
    Write-Host "  NuGet:         $($NuGet.Source)"
}
Write-Host "Environment check passed."
