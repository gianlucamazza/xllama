# Xbox Device Portal

## Enabling Dev Mode

1. On the console: **Settings → System → Developer Settings → Activate Developer Mode**.
2. If you already have an active Xbox dev subscription, activation is free and instant.
3. The console reboots into Dev Home (green header). Switch back to Retail Mode from the same menu.

## Enabling Device Portal

In Dev Mode, open **Dev Home** on the console and enable **Device Portal**.

The portal is then accessible at:
```
https://<console-ip>:11443
```
Accept the self-signed TLS certificate on first visit (use `-k` with curl).

### Finding the console IP

**Settings → Network → Advanced settings → IP address**

Or from Dev Home: the IP is displayed on the main screen.

### Setting credentials

From Dev Home: **Settings → Device Portal credentials**. Set username and password. These are required for all REST API calls.

## REST API: Deploying a package

```bash
curl -sS \
    --basic \
    -u "$XBOX_USER:$XBOX_PASS" \
    -k \
    -X POST \
    -F "file=@xllama_0.1.0.0_x64.msix;type=application/octet-stream" \
    "https://$XBOX_IP:11443/api/app/packagemanager/package?package=xllama_0.1.0.0_x64.msix"
```

Use `./scripts/deploy.sh` which wraps this call and polls installation status.

## REST API: Listing installed packages

```bash
curl -sS --basic -u "$XBOX_USER:$XBOX_PASS" -k \
    "https://$XBOX_IP:11443/api/app/packagemanager/packages" | jq .
```

## REST API: Transferring model files

Model files must be placed in the app's `LocalFolder`. Via Device Portal:

```bash
# Upload a GGUF model to the app's LocalFolder
curl -sS --basic -u "$XBOX_USER:$XBOX_PASS" -k \
    -X POST \
    -F "file=@qwen3-1.7b-Q4_K_M.gguf;type=application/octet-stream" \
    "https://$XBOX_IP:11443/api/filesystem/apps/file?knownfolderid=LocalAppData&packagefullname=VenereLabs.xllama_0.1.0.0_x64__<token>&path=\\LocalState\\models"
```

Replace `<token>` with the package publisher token (visible in the installed packages list).

## REST API: Installing the test certificate

Packages must be signed, and the signing certificate must be trusted on the console before deployment. Install it once per new build:

```bash
curl -sS \
    --basic \
    -u "$XBOX_USER:$XBOX_PASS" \
    -k \
    -X POST \
    -F "file=@xllama-test.cer;type=application/octet-stream" \
    "https://$XBOX_IP:11443/api/app/packagemanager/certificate?package=xllama-test.cer"
```

`scripts/deploy.sh` performs this automatically when the `.cer` is found alongside the `.msix`.

## REST API: Reading files from LocalState

`ApplicationData.LocalFolder` in UWP maps to the `LocalState/` subdirectory of the app's data folder. Use the `filename` parameter (lowercase) to read individual files:

```bash
PFN="VenereLabs.xllama_0.1.0.0_x64__<token>"

# Read xllama.log
curl -sS --basic -u "$XBOX_USER:$XBOX_PASS" -k \
    "https://$XBOX_IP:11443/api/filesystem/apps/file?knownfolderid=LocalAppData&packagefullname=${PFN}&path=\\LocalState&filename=xllama.log"

# Read bench-result.csv
curl -sS --basic -u "$XBOX_USER:$XBOX_PASS" -k \
    "https://$XBOX_IP:11443/api/filesystem/apps/file?knownfolderid=LocalAppData&packagefullname=${PFN}&path=\\LocalState&filename=bench-result.csv"
```

Note: the `path` parameter specifies the **directory** (`\\LocalState` or `\\LocalState\\models`), while `filename` specifies the file within it. Combining them into a single `path` parameter returns a 400 error.

## File Explorer (GUI)

Navigate to `https://<ip>:11443/#fileExplorer` for a browser-based file manager — easier for one-off file transfers.

## References

- [Device Portal for Xbox One](https://learn.microsoft.com/en-us/windows/uwp/xbox-apps/device-portal-xbox)
- [Windows Device Portal overview](https://learn.microsoft.com/en-us/windows/uwp/debug-test-perf/device-portal)
- [Package Manager REST API](https://learn.microsoft.com/en-us/windows/uwp/xbox-apps/reference/packagemanager-api)
