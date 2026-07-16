# Using xllama on the Xbox

App guide for the gamepad UI. For installation see
[install-release.md](./install-release.md); for recommended models and settings see
[recommended-config.md](./recommended-config.md); for the engineering background see
[technical-report.md](./technical-report.md).

## First launch — model download

The MSIX ships no model (~19 MB). On first launch the app downloads the default
chat model (**LFM2.5-350M** Q4_K_M, ~219 MB, on unified shipping builds; ORT-only
builds still use SmolLM2-360M-Instruct INT4, ~417 MB) from the
[`models-v1` GitHub Release](https://github.com/gianlucamazza/xllama/releases/tag/models-v1)
with a progress bar, writes it to `LocalState\models\`, and opens the chat.
The console needs internet access for this step; afterwards everything runs
offline. If the download fails, provisioning via Device Portal or USB
(`E:\xllama\models\<name>`) is described in
[model-selection.md](./model-selection.md).

## Chat

Type with the on-screen keyboard (or a USB keyboard) and send. The toolbar:

- **`[=] History`** — saved conversations: reopen, delete, or start a
  **New conversation**. History persists in `LocalState`.
- **`[S] Settings`** — see below.
- **`[*] Image`** — image generation, see below.

Generation shows live tok/s; **■ Cancel** stops a running reply.

## Settings (`[S]`)

- **Model** — ComboBox populated from the model catalogue
  (`uwp/models/manifest.json`; overridable, see
  [model-selection.md](./model-selection.md)). Selecting an entry with a
  download URL fetches it on the spot; entries without one expect USB or
  Device Portal provisioning.
- **Sampling** — Temperature, Top-p, Top-k, Repetition penalty, Max new tokens.
- **KV-cache reuse** (default on) — reuses the conversation's KV cache across
  turns; measured **4.87×** faster turn-2 prefill on console. Leave it on
  unless debugging.
- **EP routing (per conversation)** — where inference runs.
  **Superseded by #91**: while `kDmlTextLogitsBroken` holds (the DML EP
  computes wrong text logits on the Series S driver), every mode resolves to
  the CPU model, the `gpu_model` is not auto-downloaded (#95), and a missing
  `gpu_model` never blocks a turn (#100). Diffusion (plain ORT) stays on GPU.
  The pre-#91 semantics, which return when the gate lifts (still subject to
  `gpu_available` = provisioned `gpu_model` — see below):
  - **CPU only (default)** — best decode throughput at 360M scale (~66 tok/s).
  - **GPU only (DML)** — prefers DirectML when `gpu_model` is provisioned (e.g.
    `smollm2-360m-dml-fp16` in LocalState); if `gpu_available` is false, falls
    back to the CPU model.
  - **Auto (long prompts → GPU)** — long first prompts route to GPU fp16 only
    when `gpu_available` is true and the prompt exceeds the token threshold
    (prefill is 1.8× faster at ~1k tokens); short chats, or any turn with a
    missing GPU model, stay on CPU. The choice is sticky per conversation.

**Note for GGUF models** (`kind: "gguf"` in the catalogue): **KV-cache reuse
works** (the llama.cpp path keeps a persistent context and appends only the new
turn's delta — turn-2 prefill ~4× faster, see [benchmarks.md](benchmarks.md)).
Only **EP routing** is disabled (greyed out): the llama.cpp UWP build is
CPU-only, so there is no GPU model to route to.

Settings persist to `LocalState\settings.json` and take effect on the next
inference call.

## Image generation (`[*]`)

The dialog shows the last generated image (if any), a prompt box, a **Steps**
slider (1–4; SD-Turbo needs 1), and an optional **TAESD fast VAE** toggle
(~5 MB download, targets ~4.5 s total vs ~5–7 s with the full VAE). Pressing
**Generate**:

1. writes the prompt/steps/seed to `LocalState`,
2. runs SD-Turbo fp16 on the GPU **in-process** (~5–7 s for 512×512; progress
   appears in the status bar),
3. on completion, reopen `[*] Image` to view `diffuse-out.png`.

Plain ORT DirectML (diffusion) coexists with the XAML compositor. The D3D12
device conflict (`887A0036`, `uwp-constraints.md §7`) applies only to ORT
**GenAI** chat DML, not to this pipeline. Press **Cancel** during generation to
abort between UNet steps.

The `sd-turbo-fp16` model (2.4 GB: text encoder, UNet, VAE decoder + CLIP
tokenizer) is **downloaded automatically on the first Generate** from the
model catalogue (the Dev Mode disk budget is 90 GB after raising it via Dev
Home — see [uwp-constraints.md §9](uwp-constraints.md); free space if you have
not raised it); Device Portal provisioning remains available as an
alternative ([../diffusion/README.md](../diffusion/README.md)). Each
generation writes
`diffuse-out.png`, `diffuse-result.csv` (per-stage timings) and a live
`diffuse-progress.txt`; no cleanup is needed between runs.
