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

The bundled SmolLM2-360M model is included in the MSIX — no upload is required for the standard flow. Use this only for swapping models in development.

Models are ONNX GenAI **directories** (not single files). Upload each file in the directory:

```bash
source ~/.config/xllama/xbox-env
PFN=$(./scripts/deploy.sh pfn)

# Upload all files in a model directory
./scripts/deploy.sh upload-dir ./my-model-dir/ "$PFN" "models\\my-model-name"
```

Or upload individual files:

```bash
curl -sS --basic -u "$XBOX_USER:$XBOX_PASS" -k \
    -X POST \
    -F "file=@genai_config.json;type=application/octet-stream" \
    "https://$XBOX_IP:11443/api/filesystem/apps/file?knownfolderid=LocalAppData&packagefullname=VenereLabs.xllama_0.1.0.0_x64__<token>&path=\\LocalState\\models\\my-model-name"
```

Replace `<token>` with the package publisher token (visible in the installed packages list).

**Important**: `OgaCreateModel` receives the **directory path** (containing `genai_config.json`), not a path to `model.onnx`. The directory name is what `model.txt` or the hardcoded default refers to.

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

`ApplicationData.LocalFolder` maps to `LocalState/` in the app's data folder. Use the `filename` parameter (lowercase) to read individual files:

```bash
PFN="VenereLabs.xllama_0.1.0.0_x64__<token>"

# Read xllama.log
curl -sS --basic -u "$XBOX_USER:$XBOX_PASS" -k \
    "https://$XBOX_IP:11443/api/filesystem/apps/file?knownfolderid=LocalAppData&packagefullname=${PFN}&path=\\LocalState&filename=xllama.log"

# Read bench-result.csv
curl -sS --basic -u "$XBOX_USER:$XBOX_PASS" -k \
    "https://$XBOX_IP:11443/api/filesystem/apps/file?knownfolderid=LocalAppData&packagefullname=${PFN}&path=\\LocalState&filename=bench-result.csv"
```

Note: `path` specifies the **directory** (`\\LocalState` or `\\LocalState\\models\\<name>`), while `filename` specifies the file within it. Combining them into a single `path` parameter returns a 400 error.

## File Explorer (GUI)

Navigate to `https://<ip>:11443/#fileExplorer` for a browser-based file manager.

## Lifecycle gotchas (verified on console, 2026-07-07)

Five non-obvious behaviours of Device Portal + UWP package lifecycle that have
each invalidated at least one bench run or upload in this project:

1. **Only a forward upgrade (higher version) preserves `LocalState`.**
   Uninstall + install — including any downgrade — purges all data for the
   package family: uploaded models and logs are gone. Plan version numbers
   accordingly before deploying a test build.
2. **`LocalState` does not exist until the app has launched once** after a
   clean install. Run `deploy.sh start-app` (and wait a few seconds) before
   any upload to a fresh package, otherwise mkdir/upload fail.
3. **The WDP file APIs return HTTP 200 with `"Success": false`** in the body
   (`"File move failed"`, path not found). Always check the response body
   _and_ verify with a post-upload listing. `deploy.sh upload-dir` discards
   responses (`/dev/null`): it can report "Uploaded N file(s)" with zero files
   actually arrived. Multi-level mkdir fails silently when the parent is
   missing — create directories one level at a time.
4. **The PFN read right after "Installation succeeded" can be stale**: for a
   few seconds the listing may still show the previous version, or both
   versions at once (staging window). Re-read the PFN before using it in
   file/taskmanager endpoints.
5. **A console running a game is unreachable via Device Portal** (total
   timeouts). Not a fault — the portal only responds from Dev Home.

## References

- [Device Portal for Xbox One](https://learn.microsoft.com/en-us/windows/uwp/xbox-apps/device-portal-xbox)
- [Windows Device Portal overview](https://learn.microsoft.com/en-us/windows/uwp/debug-test-perf/device-portal)
- [Package Manager REST API](https://learn.microsoft.com/en-us/windows/uwp/xbox-apps/reference/packagemanager-api)
