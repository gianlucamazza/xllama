# Demo video runbook — Phase 6 close-out

**Goal:** 60–90 s clip of xllama running **on Xbox Series S** (local chat +
image gen). Closes the last Phase 6 product bullet in `ROADMAP.md`.

**Shipping target:** [v1.2.0.0](https://github.com/gianlucamazza/xllama/releases/tag/v1.2.0.0)
(`xllama_1.2.0.536_x64.msix`). Do **not** demo GPU text routing (#91 gate).

---

## Preflight status (host-verified)

| Check | Status |
| --- | --- |
| Console WDP | `192.168.1.44` Series S reachable |
| Package | **`1.2.0.536`** installed + started |
| Default model settings | **`lfm25-350m`** (routing CPU-only `0`) |
| `lfm25-350m` on disk | Yes (~219 MB GGUF + `.complete`) |
| `sd-turbo-fp16` on disk | Yes (encoder / unet / vae / tokenizer) |
| Headless `bench.flag` / `autopilot.flag` | Absent (UI path) |
| `api.flag` | Present (LAN API OK; harmless for demo) |

Re-run deploy if the console was wiped:

```bash
source ~/.config/xllama/xbox-env
gh release download v1.2.0.0 -D /tmp/xllama-120 --clobber
./scripts/deploy.sh install-cert /tmp/xllama-120/xllama-test.cer
./scripts/deploy.sh /tmp/xllama-120/xllama_1.2.0.536_x64.msix
./scripts/deploy.sh start-app
```

**Dev Home (human, 30 s):** tile → View details → **App type = Game**.

---

## Storyboard (60–90 s)

| t | Scene | Action | Prompt / note |
| --- | --- | --- | --- |
| 0–5 s | Cold open | HDMI on dashboard → open xllama | Optional title overlay in post |
| 5–15 s | Launch | Chat UI visible | Model already warm (no download) |
| 15–40 s | Chat #1 | Short user message + stream | *Explain Xbox Series S in two sentences.* |
| 40–55 s | Chat #2 | Follow-up | *Now say that in one line.* (shows multi-turn) |
| 55–80 s | Image | `[*] Image` → Generate | *pixel art robot, simple colors* — Steps 1 |
| 80–90 s | Close | Hold on image or return to chat | Caption: *v1.2.0 · local · no cloud* |

**Audio:** mute + captions (recommended).

**Avoid on camera:** Settings routing GPU/Auto as a feature; long downloads;
`bench.flag` / headless; error dialogs.

---

## Capture options

1. HDMI capture card → host OBS/ffmpeg (best)
2. Phone/camera on TV (stable tripod; research-grade OK)
3. Console native capture if available in Dev Mode (do not rely on it)

Export: 1080p or 720p, 30 fps, H.264, ~60–90 s final.

**Preferred publish path (avoid large git blobs):**

```bash
# after trim
gh release upload v1.2.0.0 ./xllama-demo-v1.2.0.mp4 --clobber
```

Then link from README / Discussion #76.

---

## After the clip exists

1. Upload asset (release or `docs/screenshots/` if small enough).
2. Docs close-out:
   - `ROADMAP.md` — demo `[x]`; Phase 6 header DONE
   - `CHANGELOG.md` — note under Docs/Measured
   - `docs/vendor-lifecycle-plan.md` — R1 done
   - `docs/project-analysis-2026-07.md` — demo residual closed
   - `README.md` — Demo link near Status
3. Optional: comment on Discussion #76 with the video URL.

---

## Prompts (copy-paste)

Chat 1:

```text
Explain Xbox Series S in two sentences.
```

Chat 2:

```text
Now say that in one line.
```

Image:

```text
pixel art robot, simple colors
```

---

## Done criteria

- [ ] Clip shows live tok/s chat + SD image on Xbox UI
- [ ] Public URL works (release asset preferred)
- [ ] ROADMAP Phase 6 demo checked
- [ ] No GPU-routing claim in captions
