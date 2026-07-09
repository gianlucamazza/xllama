# Using xllama on the Xbox

App guide for the gamepad UI. For installation see
[install-release.md](./install-release.md); for the engineering background see
[technical-report.md](./technical-report.md).

## First launch — model download

The MSIX ships no model (~19 MB). On first launch the app downloads the default
chat model (SmolLM2-360M-Instruct INT4, ~417 MB) from the
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
- **EP routing (per conversation)** — where inference runs:
  - **CPU only (default)** — best decode throughput at 360M scale (~66 tok/s).
  - **GPU only (DML)** — forces DirectML (needs the `gpu_model`, e.g.
    `smollm2-360m-dml-fp16`, in LocalState).
  - **Auto (long prompts → GPU)** — long first prompts route to GPU fp16
    (prefill is 1.8× faster at ~1k tokens), short chats stay on CPU; the
    choice is sticky per conversation.

**Note for GGUF models** (`kind: "gguf"` in the catalogue): the **KV-cache reuse**
and **EP routing** controls are disabled (greyed out). The llama.cpp backend is
stateless and runs CPU-only on Xbox; delta-only reuse would produce incorrect
output and GPU routing does not apply.

Settings persist to `LocalState\settings.json` and take effect on the next
inference call.

## Image generation (`[*]`)

The dialog shows the last generated image (if any), a prompt box, and a
**Steps** slider (1–4; SD-Turbo needs 1). Pressing **Generate (restarts app)**:

1. writes the prompt/steps/seed and the `diffuse.flag`,
2. **restarts the app** into a headless mode that runs SD-Turbo fp16 on the
   GPU (~7 s for 512×512, plus ~7 s of model load) and exits,
3. at the next launch, open `[*] Image` again to see the result.

The restart is a workaround for a D3D12 device conflict between DirectML and
the XAML compositor (`uwp-constraints.md §7`); removing it is in progress
(in-process experiment + upstream fix
[onnxruntime-genai#2280](https://github.com/microsoft/onnxruntime-genai/pull/2280)).

The `sd-turbo-fp16` model (2.4 GB: text encoder, UNet, VAE decoder + CLIP
tokenizer) is **downloaded automatically on the first Generate** from the
model catalogue, disk permitting (the Dev Mode budget is tight — free space
first if needed); Device Portal provisioning remains available as an
alternative ([../diffusion/README.md](../diffusion/README.md)). Each
generation writes
`diffuse-out.png`, `diffuse-result.csv` (per-stage timings) and a live
`diffuse-progress.txt`; no cleanup is needed between runs.
