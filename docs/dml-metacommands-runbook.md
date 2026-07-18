# DML metacommands opt-out — #91 experiment runbook

**Hypothesis.** The #91 broken text logits (contrib `GroupQueryAttention` /
`MultiHeadAttention` wrong on the Series S GPU, NMSE ~1, while decomposed
attention — SD-Turbo — is correct on the same device, and the same weights are
correct on CPU/desktop) match the profile of a broken **driver metacommand**:
DirectML delegates fused attention to vendor metacommands when available, and
falls back to its own compiled shaders when they are disabled.

**Why a vendored patch.** The `OrtDmlApi` DML/DML1 entry points hardcode
`disable_metacommands=false`, and ORT GenAI must go through DML1 to reuse its
own `IDMLDevice`/command queue — so the knob is unreachable from
`genai_config.json` provider options (GenAI's DML branch only parses
`luid`/`device_index`). `patches/onnxruntime-dml-metacommands-optout.patch`
adds the session config key `ep.dml.disable_metacommands` in
`DMLProviderFactory` (same `ep.dml.*` pattern as graph capture), which GenAI
already forwards: unknown `session_options` keys become `AddConfigEntry`.
Covers the plain-ORT diffusion sessions too.

## Procedure

1. **Build the patched DLL** (CI, 1–3 h):
   `gh workflow run build-uwp-ort-patched.yml` — the lane now applies both the
   extdata patch and the metacommands opt-out via
   `scripts/vendor-ort-extdata-patch.ps1 -Build`.
2. **Deploy** the resulting build to the console (normal deploy flow; the
   patched `onnxruntime.dll` replaces the NuGet copy at packaging).
3. **Switch the DML text model to the opt-out config**:
   ```bash
   source ~/.config/xllama/xbox-env
   ./scripts/test-dml-config.sh --model smollm2-360m-dml-fp16 \
       --config bench/configs/genai_config-dml-metacmd-off.json
   ```
4. **Verdict** — logit parity on device:
   ```bash
   MODEL=smollm2-360m-dml-fp16 ./scripts/validate-logit-parity.sh
   ```
   Read the `compare-logits.py` verdict (NMSE + top-token agreement), not just
   the exit code.
5. **Restore** the original config: `./scripts/test-dml-config.sh --model smollm2-360m-dml-fp16 --restore`.

## Outcomes

- **PASS** → metacommand confirmed as the #91 culprit. Next: bench prefill-1k /
  decode with metacommands off (`bench-xbox-ort.sh`) to price the fallback
  shaders, then decide whether to lift `kDmlTextLogitsBroken`
  (`include/xllama/routing_policy.h`) behind the existing `token_threshold`
  routing — and report the finding on upstream #91/onnxruntime#29739.
- **FAIL (same wrong logits)** → the bug is in DML's own shader path, not the
  driver metacommand; drop this patch (single hunk, no behavior change unless
  the config key is set) and move to plan B: decomposed-attention text model
  (graph surgery), whose correctness is already proven by SD-Turbo.

Status: **run on console 2026-07-19 — FAIL (same wrong logits).**

## Result (2026-07-19, XDKS.1 sandbox, OS 10.0.26100.8569)

Build: `build-uwp-ort-patched` run 29622472630 (msix 1.3.0.4; packaged
`onnxruntime.dll` sha256 `640875b2…` verified == the lane's freshly patched
artifact, not the pinned `vendor-dlls-v1` DLL). On-device
`genai_config.json` verified byte-identical to
`bench/configs/genai_config-dml-metacmd-off.json` before the run.

| config                          | max_abs_diff | NMSE      | top1                     | top10_overlap |
| ------------------------------- | ------------ | --------- | ------------------------ | ------------- |
| stock DML (baseline)            | 17.438866    | 9.826e-01 | ` the` (golden ` Paris`) | 0.30          |
| `ep.dml.disable_metacommands=1` | 17.438866    | 9.826e-01 | ` the`                   | 0.30          |

Knob on vs off is **bit-identical** — disabling metacommands changes nothing
on this driver. Either the Series S driver never uses metacommands for these
attention ops (knob is a no-op → the wrongness lives in DML's compiled shader
path), or the fused-attention path is wrong in the same way on both routes.
Per the FAIL branch above: drop the patch, move to plan B
(decomposed-attention graph surgery, correctness already proven by SD-Turbo).
`kDmlTextLogitsBroken` stays.

Gotchas hit during the run (fixed / documented):

- `test-dml-config.sh` POSTs were missing the `X-CSRF-Token` header the Xbox
  WDP requires, so config uploads **failed silently** and the first "knob" run
  actually measured stock config. The script now sends the token and verifies
  the uploaded file round-trip (`cmp`) before declaring success.
- The console had lost its signed-in user: the app died at launch with no log,
  no crash dump, no WER report — only a WDP screenshot revealed the
  "Sign in to start this app (0x8004090a)" dialog. Fixed via
  `PUT /ext/user {"Users":[{"UserId":…,"SignedIn":true}]}`.
- After sign-in the package LocalState moved from `Q:\Users\DefaultAccount\…`
  to `Q:\Users\UserMgr0\…`, and WDP uploads fail until the app has run once
  and created its LocalState. Uninstalling a package (needed for version
  downgrades: WDP refuses to install a lower version) wipes LocalState,
  including provisioned models.
