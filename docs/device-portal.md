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
    -F "file=@xllama_X.Y.Z.R_x64.msix;type=application/octet-stream" \
    "https://$XBOX_IP:11443/api/app/packagemanager/package?package=xllama_X.Y.Z.R_x64.msix"
```

Use `./scripts/deploy.sh` which wraps this call and polls installation status.
Other wrapped operations: `upload-file` / `upload-dir` / `mkdir-localstate`
(LocalState writes), `fetch-file` / `get-log` / `list-localstate` (reads),
`start-app` / `stop-app` / `diagnose-startup`, `install-cert`, `pfn`.

## REST API: Listing installed packages

```bash
curl -sS --basic -u "$XBOX_USER:$XBOX_PASS" -k \
    "https://$XBOX_IP:11443/api/app/packagemanager/packages" | jq .
```

## REST API: Transferring model files

The standard flow needs no upload: the app downloads catalogue models on first use (`uwp/models/manifest.json` → `models-v1` GitHub Release). Use this only for provisioning models that have no download URL, or for development swaps.

Catalogue models use a directory under `LocalState\models\<name>`. ORT GenAI
entries contain `genai_config.json`, ONNX weights and tokenizer files; GGUF
entries contain the selected `.gguf` file. Upload the complete directory:

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
    "https://$XBOX_IP:11443/api/filesystem/apps/file?knownfolderid=LocalAppData&packagefullname=$(./scripts/deploy.sh pfn)&path=\\LocalState\\models\\my-model-name"
```

Replace `<token>` with the package publisher token (visible in the installed packages list).

**Important**: ORT receives the directory containing `genai_config.json`; the
llama.cpp backend resolves the GGUF inside its catalogue directory. Select
models by catalogue id through the UI/settings rather than relying on a
hardcoded filename.

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
PFN=$(./scripts/deploy.sh pfn)

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

## The `/ext/` endpoints: seeing and signing in

Two endpoints outside `/api/` that this project depends on. Neither is in the
Microsoft reference pages linked below, and both were found the hard way.

### `GET /ext/screenshot` — a PNG of what is on the TV

```bash
curl -sS --basic -u "$XBOX_USER:$XBOX_PASS" -k \
    -o frame.png "https://$XBOX_IP:11443/ext/screenshot"
```

1920×1080 PNG of the current console output. A plain GET, so no CSRF token.

**Measured, 2026-07-30** (`scripts/bench-screenshot-rate.sh`, 30 back-to-back
requests after one warm-up):

| condition       | latency min / p50 / p90 | sustained   |
| --------------- | ----------------------- | ----------- |
| console idle    | 0.029 / 0.037 / 0.226 s | **11.5 Hz** |
| during a decode | 0.027 / 0.032 / 0.160 s | **13.7 Hz** |

The endpoint is an order of magnitude faster than the 1 Hz the demo pipeline
assumed — that 1 Hz was a `sleep 1` in `capture-demo-video.sh`, never a measured
ceiling. It is also not slowed by inference: the frame under load is _smaller_
(53 KB vs 79 KB) because a chat screen compresses better than Dev Home, and the
capture is served by the system rather than by the app's threads.

**What capture costs the app**, which is the direction that matters for a demo
that displays tok/s. LAN API, `lfm25-350m`, 250 tokens, temperature 0, three
runs each:

|                     | decode                      | median |
| ------------------- | --------------------------- | ------ |
| no capture          | 93.70 / 93.66 / 93.75 tok/s | 93.70  |
| capture at ~13.5 Hz | 91.28 / 91.98 / 92.53 tok/s | 91.98  |

**−1.8%**, with non-overlapping ranges, so it is a real effect and not noise. A
capture at ~10 Hz is therefore affordable: the numbers a demo shows are within
2% of the numbers without a camera on them.

This is the only way to see the app's UI from the host, and it is the answer
whenever a failure leaves no textual trace. During the DirectML metacommands
work the app died at launch with no log, no crash dump and no WER report; a
screenshot showed a "Sign in to start this app (0x8004090a)" dialog waiting for
a button press (`dml-metacommands-runbook.md`). Nothing else on this list would
have found that.

Consumers in the repo:

- `scripts/capture-demo-video.sh` polls it to reconstruct a demo video. Note
  what that implies — there is **no video capture endpoint**; the "video" is a
  sequence of stills;
- `scripts/capture-store-screenshots.sh` takes one frame per named UI state for
  the Store listing, synchronised with the app through the `mark` autopilot op
  rather than by sleeping;
- `scripts/validate-console.sh` keeps the last two frames of every gate run and
  writes them out only when the gate fails (`XLLAMA_GATE_SHOTS=0` disables it,
  `XLLAMA_GATE_SHOTS_DIR` moves the output);
- `scripts/bench-screenshot-rate.sh` measures what the endpoint actually
  sustains. The demo pipeline's "1 Hz" is a `sleep 1`, not a measured ceiling,
  and a video rebuilt from stills is only as smooth as this number.

### `PUT /ext/user` — sign a user in

```bash
curl -sS --basic -u "$XBOX_USER:$XBOX_PASS" -k -X PUT \
    -H "X-CSRF-Token: $CSRF_TOKEN" -H 'Content-Type: application/json' \
    -d '{"Users":[{"UserId":"<id>","SignedIn":true}]}' \
    "https://$XBOX_IP:11443/ext/user"
```

The console can lose its signed-in user across a reboot, and an app that
requires one then fails to launch in the silent way described above. Send the
CSRF token as with any other state-changing call — the requirement is verified
for POST/DELETE and assumed here rather than measured. Signing in also moves the
package's `LocalState` (`dml-metacommands-runbook.md` has the consequences).

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
   _and_ verify with a post-upload listing. `deploy.sh upload-dir` now parses
   every response body, prints `ERROR: upload of <file> failed` per file, appends
   `(N FAILED)` to its summary and exits non-zero; it also verifies the remote
   directory exists before uploading. Raw `curl` calls still need the body check
   done by hand. Multi-level mkdir fails silently when the parent is missing —
   create directories one level at a time.
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
