# Windows UWP Development VM

The Xbox/UWP package cannot be built in a Linux Docker container. The local
Docker daemon is a Linux container host, while Windows containers require a
Windows host. Build and package xllama in a Windows VM or use the existing
`build-uwp` GitHub Actions workflow pinned to `windows-2022`.

## Host preflight

On the Arch host:

```bash
./scripts/check-uwp-host.sh
```

Expected host tools:

- Docker can stay Linux-only; it is useful for other work, not UWP packaging.
- QEMU/libvirt provide the Windows VM runtime.
- The user should be in `kvm` and `libvirt`.
- Start libvirt before creating or running the VM:

```bash
sudo systemctl enable --now libvirtd
```

Use at least 8 GB RAM and 70 GB disk for the Windows VM. More disk is better
because Visual Studio, Windows SDKs, and package artifacts are large.

## VM setup

Use either a Microsoft Windows developer VM, if downloads are available, or a
regular Windows 11 x64 ISO/evaluation install. The Microsoft developer VM is
preferred when available because it already includes Visual Studio 2022 with
UWP workloads, but Microsoft may temporarily disable those downloads.

For a manual Windows 11 VM:

1. Create a VM with `virt-manager` or `virt-install`.
2. Enable Developer Mode in Windows settings.
3. Install Git for Windows and PowerShell.
4. Open an elevated PowerShell prompt.
5. Clone this repository with submodules.
6. Run the Windows UWP setup check:

```powershell
.\scripts\setup-windows-uwp-dev.ps1 -Install
```

If Visual Studio is already installed, omit `-Install` to verify only:

```powershell
.\scripts\setup-windows-uwp-dev.ps1
```

The check must find:

- Visual Studio 2022 Build Tools or Community
- MSBuild
- Windows SDK tools: `MakeAppx.exe` and `signtool.exe`
- `nuget.exe`

## Build

Inside the Windows VM:

```powershell
git submodule update --init --recursive
.\scripts\build-uwp.ps1 -Configuration Release -Platform x64
```

The build script:

1. Restores NuGet packages (`nuget restore`).
2. Builds with MSBuild (`-Backend unified` matches the shipping artifact;
   `llamacpp` is the bench-only lane and the script default is an ORT-only local build).
3. Signs the package with the test certificate.

No model is packaged: distribution artifacts are prepared separately and must be self-contained (`scripts/merge_onnx_external_data.py`, `docs/uwp-constraints.md §8`).

The package output is under:

```text
uwp\AppPackages\
```

The final artifact set must include:

- the main `.msix`
- dependency `.appx` files from `Dependencies\x64`
- `uwp\xllama-test.cer`

## Deploy and diagnose

From the Linux host, after copying the artifacts back:

```bash
source ~/.config/xllama/xbox-env
./scripts/deploy.sh path/to/xllama_*.msix
./scripts/deploy.sh diagnose-startup
```

The startup log should show:

```text
[xllama] App::App()
[xllama] App::App() complete
[xllama] App::OnLaunched
[xllama] building MainPageController
[xllama] MainPageController built
[xllama] MainPageController init done
[xllama] Window.Content set
[xllama] Window activated
```

If the app crashes before printing any log entry, collect a minidump via WDP:

```bash
curl --basic -u "${XBOX_USER}:${XBOX_PASS}" -k -sS \
     "https://${XBOX_IP}:11443/api/debug/dump/usermode/dumps"
```

Use `./scripts/deploy.sh diagnose-startup` for process state, logs and minidumps;
Device Portal details are in `docs/device-portal.md`.
