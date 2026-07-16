# Demo video runbook — Phase 6 close-out

**Status:** **DONE** 2026-07-17.

**Published clip:** [xllama-demo-v1.2.0.mp4](https://github.com/gianlucamazza/xllama/releases/download/v1.2.0.0/xllama-demo-v1.2.0.mp4)
(~74 s) on [v1.2.0.0](https://github.com/gianlucamazza/xllama/releases/tag/v1.2.0.0).
Local: `docs/screenshots/xllama-demo-v1.2.0.mp4`. Still:
`docs/screenshots/xllama-demo-diffuse-robot.png`.

---

## What was captured

| Segment | Content |
| --- | --- |
| Chat #1 | LFM2.5-350M — “Explain Xbox Series S in two sentences.” (~59 tok/s UI) |
| Chat #2 | Follow-up — “Now say that in one line.” (KV-reuse) |
| Image | SD-Turbo 1-step — “pixel art robot, simple colors” (~6.2 s) |

Package: **1.2.0.536**. Path: CPU/GGUF chat (#91) + GPU diffusion. No GPU text routing.

---

## Reproduce (host + console)

```bash
source ~/.config/xllama/xbox-env
# package 1.2.0.x installed; lfm25-350m + sd-turbo-fp16 in LocalState
./scripts/capture-demo-video.sh   # → docs/screenshots/xllama-demo-v1.2.0.mp4
```

Mechanism: autopilot drives UI; Device Portal `GET /ext/screenshot` grabs
1920×1080 PNG at ~1 Hz; ffmpeg builds H.264 720p with caption overlay.

Optional re-upload:

```bash
gh release upload v1.2.0.0 docs/screenshots/xllama-demo-v1.2.0.mp4 --clobber
```

---

## Done criteria (all met)

- [x] Clip shows live chat + SD image on Xbox UI
- [x] Public URL on GitHub Release v1.2.0.0
- [x] ROADMAP Phase 6 demo checked / Phase 6 DONE
- [x] No GPU-routing claim in captions
