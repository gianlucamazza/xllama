# Using xllama on the Xbox

App guide for the gamepad UI. For installation see
[install-release.md](./install-release.md); for recommended models and settings see
[recommended-config.md](./recommended-config.md); for the engineering background see
[technical-report.md](./technical-report.md).

## First launch — model download

The MSIX ships no model (~19 MB). On first launch the app downloads the default
chat model (**LFM2.5-350M** Q4_K_M, ~229 MB download, on unified shipping builds;
ORT-only builds still use SmolLM2-360M-Instruct INT4, ~421 MB) from the
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

Every completed assistant response has **Like**, **Dislike**, and **Correct**
actions. A correction requires the preferred answer. Each response can be rated
once; the choice is stored with the conversation and appended to
`LocalState\training\samples.jsonl`. Cancelled/partial responses cannot be rated.

## Model tiers: chat, coding, thinking

The catalogue is not one model in several sizes. Three kinds behave differently
in ways you will notice, and the picker does not explain them:

- **Chat** (the default, LFM2.5-350M, and the larger LFM2.5-1.2B / LFM2-2.6B) —
  a 2048-token context. This is what a first launch gives you.
- **Coding** (`qwen25-coder-0.5b` / `-1.5b` / `-3b`) — the same UI, but the
  session opens a **4096-token** context, because the point of the tier is
  pasting real source. That is the only difference you set: there is no separate
  mode to switch on, the model's catalogue entry carries it.
- **Thinking** (`lfm25-1.2b-thinking`) — reasons before answering. You **see the
  reasoning stream on screen while it generates**, and the message that is kept
  when it finishes holds only the final answer. The chain of thought is not
  saved to the conversation.

Two consequences of the thinking tier worth knowing before you blame the app:

- If the model spends its whole token budget reasoning and never reaches an
  answer, the turn is kept and says so:
  `(reasoning only — the answer did not fit; raise "Max new tokens" in Settings)`.
  Selecting this model sets **Max new tokens to 1024** (catalogue `n_predict`).
  Short prompts usually finish under that budget; **hard multi-step questions
  may still never close the reasoning block** even higher (measured: the
  “train leaves at 14:05…” prompt spent 768 tokens still inside `<think>`,
  looping the same arithmetic). Prefer chat or coding tiers unless you want to
  watch a model reason. Console gate `thinkdone` pins a short happy path.
- **KV-cache reuse is skipped for thinking models.** The saved conversation holds
  the stripped answer while the cache holds the full reasoning, so the two can
  never match on the way back; reusing it would be wrong rather than fast.
  Returning to a thinking conversation therefore re-reads it, and the first turn
  after a switch is slower than it would be on a chat model.

## When the context fills up

Every model has a fixed context, and a long conversation eventually exceeds it.
What happens then is deliberate, and only one half of it is visible:

- **Old turns stop being sent.** The oldest turns are dropped from what the model
  sees so the newest ones and your reply still fit. **They stay on your screen** —
  the conversation is not edited — so the model can appear to forget something you
  can still read. Starting a new conversation is the way to clear it.
- **The reply keeps its room.** The prompt is budgeted so that the answer you
  asked for still fits alongside it, rather than being cut short in silence once
  the context is nearly full.
- **A single message longer than the whole context is refused**, not truncated:
  `prompt too long: N tokens exceed n_ctx=M — shorten the message or start a new
chat`. On a coding session that ceiling is 4096 tokens, roughly 13 KB of dense
  source.

## Settings (`[S]`)

- **Model** — ComboBox populated from the model catalogue
  (`uwp/models/manifest.json`; overridable, see
  [model-selection.md](./model-selection.md)). Selecting an entry with a
  download URL fetches it on the spot; entries without one expect USB or
  Device Portal provisioning. After a successful on-device personalize train
  (below), a **Personalized (from your feedback)** entry (`personalized`)
  appears — a GGUF merged from your preference samples.
  The read-only **Runtime LoRA** line shows the selected entry's adapter and
  scale, or `none`; adapter configuration remains part of the catalogue manifest.
- **Personalize** — status line shows how many **usable** preference samples
  are in `LocalState\training\samples.jsonl` (Like/Correct count; Dislike is
  stored but not used for training) and which base GGUF will be used. The
  secondary dialog button **Train on my feedback** runs an in-process last-block
  partial fine-tune (Lane B), shows epoch/loss on the status bar, then switches
  the model picker to `personalized` when done. Requires a base GGUF:
  `LocalState\training\base-f16.gguf` (operator upload, same as the device-train
  harness) or a provisioned SmolLM2 GGUF in the catalogue. Cancel requests a
  cooperative abort between epochs. Details:
  [training-architecture.md §11](./training-architecture.md).
- **Sampling** — Temperature, Top-p, Top-k, Repetition penalty, Max new tokens.
- **KV-cache reuse** (default on) — reuses the conversation's KV cache across
  turns (large turn-2 prefill win on CPU paths; measured figures in
  [benchmarks.md](./benchmarks.md)). Leave it on unless debugging.
- **EP routing (per conversation)** — where inference runs. Routing requires a
  parity-validated DML text asset as `gpu_model` (`dml_text_model_ok`,
  #91 postmortem: the broken DML RMSNorm kernel is worked around by the
  `smollm2-360m-dml-fp16-v2` decomposed graph). With any other `gpu_model`
  every mode resolves to the CPU model, it is not auto-downloaded (#95), and a
  missing `gpu_model` never blocks a turn (#100). Diffusion (plain ORT) stays
  on GPU. Semantics (subject to `gpu_available` = provisioned `gpu_model`):
  - **CPU only (default)** — best decode throughput at 360M scale (ORT int4;
    see [benchmarks.md](./benchmarks.md)).
  - **GPU only (DML)** — prefers DirectML when `gpu_model` is provisioned (e.g.
    `smollm2-360m-dml-fp16-v2` in LocalState); if `gpu_available` is false, falls
    back to the CPU model.
  - **Auto (long first prompt → GPU)** — a long first prompt routes to GPU fp16
    when `gpu_available` is true and it exceeds the token threshold. What the GPU
    buys there is a faster **first token** on long prompts (GPU sessions warm up
    at load, so the turn runs at the warm prefill regime — §5e); it
    does **not** make the conversation faster overall — the GPU cannot reuse a KV
    cache, so from the second turn the CPU wins, and the routing choice is sticky
    per conversation (decided once, on turn 1). Short chats, or any turn with a
    missing GPU model, stay on CPU. The threshold (1550) is a product call, not
    a physical constant: the re-derivation concluded no single prompt-length
    crossover exists — see [docs/uwp-constraints.md §5d](./uwp-constraints.md).
- **LAN API** — enables or stops the OpenAI-compatible endpoint immediately,
  without restarting the app. The port must be 1025–49151 except 11443. The
  status line reports the active listener or bind error. Besides chat, the
  endpoint can record preferences, report training status, and generate images
  (same guardrails as the pad UI). The endpoint **shares the one loaded model
  with the chat UI**: a LAN request while you are generating on the pad gets
  a "busy" reply, and a LAN request naming a different model swaps the loaded
  model under your conversation (the next pad turn transparently re-reads the
  full context). It is unauthenticated: enable it only on a
  trusted LAN; see [api-endpoint.md](api-endpoint.md).

**Note for GGUF models** (`kind: "gguf"` in the catalogue): **KV-cache reuse
works** (persistent `llama_context`; measured ratios in
[benchmarks.md](benchmarks.md)). Only **EP routing** is greyed out — the
llama.cpp UWP build is CPU-only.

Settings persist to `LocalState\settings.json` and take effect on the next
inference call. Exception: changing the **model** loads it eagerly as soon as
it is Ready (the session pre-loads in the background), so the first send after
a model switch pays only prompt reading + generation.

## Image generation (`[*]`)

The dialog shows the last generated image and seed (if any), a prompt box, a
**Steps** slider (1–4; SD-Turbo needs 1), a **Seed** field (`0` = random), and an
optional **TAESD fast VAE** toggle
(~5 MB download, targets ~4.5 s total vs ~5–7 s with the full VAE). Pressing
**Generate**:

1. validates and persists the seed preference, then writes the concrete
   prompt/steps/seed to `LocalState`,
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
