# Xbox Device Portal

## Enabling Dev Mode

1. Create a Microsoft Partner Center account at [partner.microsoft.com](https://partner.microsoft.com).
2. On the console: **Settings → System → Developer Settings → Developer Mode**.
3. Activate Developer Mode — a one-time fee of ~$19 USD is charged to your Microsoft account.
4. The console reboots into Dev Mode. Switch back to Retail Mode from the same menu.

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
    --digest \
    -u "$XBOX_USER:$XBOX_PASS" \
    -k \
    -X POST \
    -F "file=@xllama_0.1.0_x64.appx;type=application/octet-stream" \
    "https://$XBOX_IP:11443/api/app/packagemanager/package"
```

Use `./scripts/deploy.sh` which wraps this call and polls installation status.

## REST API: Listing installed packages

```bash
curl -sS --digest -u "$XBOX_USER:$XBOX_PASS" -k \
    "https://$XBOX_IP:11443/api/app/packagemanager/packages" | jq .
```

## REST API: Transferring model files

Model files must be placed in the app's `LocalFolder`. Via Device Portal:

```bash
# Upload a GGUF model to the app's LocalFolder
curl -sS --digest -u "$XBOX_USER:$XBOX_PASS" -k \
    -X POST \
    -F "file=@qwen3-1.7b-Q4_K_M.gguf;type=application/octet-stream" \
    "https://$XBOX_IP:11443/api/filesystem/apps/file?knownfolderid=LocalAppData&packagefullname=VenereLabs.xllama_0.1.0.0_x64__<token>&path=\\models"
```

Replace `<token>` with the package publisher token (visible in the installed packages list).

## File Explorer (GUI)

Navigate to `https://<ip>:11443/#fileExplorer` for a browser-based file manager — easier for one-off file transfers.

## References

- [Device Portal for Xbox One](https://learn.microsoft.com/en-us/windows/uwp/xbox-apps/device-portal-xbox)
- [Windows Device Portal overview](https://learn.microsoft.com/en-us/windows/uwp/debug-test-perf/device-portal)
- [Package Manager REST API](https://learn.microsoft.com/en-us/windows/uwp/xbox-apps/reference/packagemanager-api)
