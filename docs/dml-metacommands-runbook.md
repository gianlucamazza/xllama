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

Status: **not yet run on console** (patch + config landed, DLL rebuild pending).
