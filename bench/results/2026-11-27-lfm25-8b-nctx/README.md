# lfm25-8b-a1b at n_ctx 2048, 4096 and 8192 on Series S (#649)

Series S (`xbox-series-s`, operator-declared), Developer Mode, CI package **1.10.0.2117**, an in-place
install. Driver: `scripts/measure-nctx-sweep.sh`, same method as
[`2026-11-25-nctx-train-bound`](../2026-11-25-nctx-train-bound/):

- A temporary LocalState `manifest.json` override changes only the entry's `n_ctx`.
- One short turn loads the model, then one streamed request of about the target prompt length.
- At the end the override was removed (`manifest.json` absent, as before), `model.txt` and
  `api.flag` were restored, and the app was relaunched.

Model: the catalogue's `LFM2.5-8B-A1B-UD-IQ3_S.gguf`. It is an LFM2-MoE hybrid: 24 layers, 6 of
them attention, `n_ctx_train` 128000.

## Results

| n_ctx | Admission estimate (host)             | Prompt processed | Max PrivateWorkingSet | `done_ms` (client) |
| ----- | ------------------------------------- | ---------------- | --------------------- | ------------------ |
| 2048  | 3693 MB (w 3397, ctx 24, compute 272) | — (see below)    | 3508 MB               | —                  |
| 4096  | 3719 MB (w 3397, ctx 48, compute 274) | 3840 tokens      | 3593 MB               | 61 314             |
| 8192  | 3771 MB (w 3397, ctx 96, compute 278) | 7809 tokens      | 3647 MB               | 131 189            |

- **Memory.** The 8192 peak is 3647 MB against the measured 4864 MB heap ceiling
  ([`phase15-ramceil`](../../../docs/model-matrix.md)). The context and compute figures equal the
  host prediction on #649 to the MiB: KV 12 KiB per token on the six attention layers, recurrent
  state 0.28 MiB.
- **The 2048 cell sent no prompt.** The 1800-token target prompt plus the reply reserve did not fit
  the 2048 budget, and `fit_prompt` refused it (`prompt too long … n_ctx is 2048`). The load peak is
  still recorded.
- **Time.** `done_ms` is the client's time for the whole request: about 63 tokens/s of prefill at
  3840 tokens and 60 tokens/s at 7809, network and HTTP included.
- **Not a decode measurement.** The sweep sends `max_tokens` 8, which this thinking model spends on
  reasoning, so no content chunk arrived ("no content delta before [DONE]") and no decode figure
  came out. Decode speed past 2048 tokens and answer quality are not measured here.

**Decision:** the catalogue entry opens at `n_ctx` 8192. The memory fits with about 1.2 GB to
spare, the admission estimate matches the device, and agent prompts of 1700–1900 tokens no longer
leave a 2048 window without room for the reply. A long prompt still costs its prefill: about 2
minutes at 7800 tokens.

## Files

- `cells.csv`: one row per cell.
- `<cell>.log`: the cell's `xllama.log` lines.
- `<cell>.ttfb.json`: client timing; the console address reads `console`.
