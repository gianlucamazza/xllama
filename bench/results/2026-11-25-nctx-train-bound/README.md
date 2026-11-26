# Long GGUF contexts under the memory admission (#570)

Series S (`xbox-series-s-declared`, operator-declared), Game designation, CI
package **1.8.1.1885** from `feat/nctx-train-bound` (#579, stacked on #571),
installed in place over 1.8.1.1819. Driver: `scripts/measure-nctx-sweep.sh`.

Each cell sets the catalogue entry's `n_ctx` through a temporary LocalState
`manifest.json` override (a copy of the bundled entry with only `n_ctx`
changed), restarts the app with the LAN API, and loads the model with one short
turn. If a prompt length is given, it then sends one streamed request of about
that many tokens. `first_content_ms` and `done_ms` are client-side
(`measure-api-ttfb.py`). The same prompt sent non-streamed gives
`usage.prompt_tokens`. The override was removed after the run.

- `cells.csv`: one row per cell.
- `<cell>.log`: the `xllama.log` lines of the cell.
- `<cell>.ttfb.json`: the client timing, with the console address redacted.

Each log starts with one extra load: the app's own startup model (n_ctx 4096,
`n_ctx_train 32768`) before the API turn switches to the cell's model.

## Admission and the trained-context cap

| Model        | n_ctx asked | Opened at  | Estimate (host)                               | WS after context | Outcome                           |
| ------------ | ----------- | ---------- | --------------------------------------------- | ---------------- | --------------------------------- |
| `lfm25-350m` | 2048        | 2048       | 370 MB (w 206, ctx 24, compute 140)           | 346 MB           | admitted                          |
| `lfm25-350m` | 8192        | 8192       | 448 MB (w 206, ctx 96, compute 146)           | 416 MB           | admitted                          |
| `lfm25-350m` | 16384       | 16384      | 552 MB (w 206, ctx 192, compute 154)          | 511 MB           | admitted                          |
| `lfm25-350m` | 32768       | 32768      | 760 MB (w 206, ctx 384, compute 170)          | 704–706 MB       | admitted                          |
| `lfm25-350m` | 131072      | **128000** | 1868 MB (w 206, ctx 1500, compute 162)        | 1824 MB          | admitted, capped at `n_ctx_train` |
| `llama32-3b` | 8192        | 8192       | 2653 MB (w 1463, ctx 896, compute 294)        | 2510 MB          | admitted                          |
| `llama32-3b` | 16384       | 16384      | 3557 MB (w 1463, ctx 1792, compute 302)       | 3401 MB          | admitted                          |
| `llama32-3b` | 32768       | —          | 5359 MB (w 1463, ctx 3584, compute 312) + 135 | —                | **refused** (400, over 5120 MB)   |

For every admitted load, the estimate is above the working set after the context.
The working set includes the app's base, about 100–135 MB.
`n_ctx 131072 capped at the model's trained context 128000` is logged before
the dry run. `llama32-3b` at 32768 answers
`insufficient memory: this model at n_ctx 32768 needs 5359 MB (weights 1463, context 3584, compute 312) with 135 MB already in use, over the 5120 MB app limit`
before any weight is read.

## Long prompts

| Model        | n_ctx | Prompt tokens | Prefill (client `done_ms`) | First content | Max private WS |
| ------------ | ----- | ------------- | -------------------------- | ------------- | -------------- |
| `lfm25-350m` | 2048  | 1700          | 4.8 s                      | 4.8 s         | 308 MB         |
| `lfm25-350m` | 8192  | 7685          | 28.2 s                     | — (EOG first) | 387 MB         |
| `lfm25-350m` | 16384 | 15836         | 75.9 s                     | 75.7 s        | 493 MB         |
| `lfm25-350m` | 32768 | 30401         | 207.3 s                    | — (EOG first) | 700 MB         |
| `llama32-3b` | 8192  | 7562          | 262.8 s                    | 262.4 s       | 2519 MB        |
| `llama32-3b` | 16384 | 15583         | 712.9 s                    | 711.6 s       | 3425 MB        |

- **Prefill throughput falls as the prompt grows.** `lfm25-350m` runs at 354, 272, 209 and 147 tok/s. `llama32-3b` runs at 29 and 22 tok/s. A 16k prompt on a 3B dense model therefore takes about 12 minutes before the first token. That is why catalogue `n_ctx` values stay unchanged in #579: a larger context admits a long prompt that used to be trimmed.
- **"EOG first":** on two filler prompts `lfm25-350m` ended with end-of-generation as its first token (`EOG after 0 tokens`). The stream therefore had no content delta, and the client could not record a first-content time. `done_ms` still measures the prefill. The log confirms `prefill done: 30401 tokens in 207141 ms`. This is the model's answer to the filler text, not a session failure. Long-context answer quality is not measured here.
- **Hybrid prefix checkpoint at long context:** the non-streamed repeat of the 30401-token prompt restored the checkpoint at 30393 of 30401 tokens and prefilled 8 tokens in 1.4 s.
- **Budget refusal:** the first 32768 cell's prompt came out at 33209 tokens. The API refused it with `400 prompt too long: … needs 33209 tokens plus 8 for the reply; but n_ctx is 32768` (`lfm25-350m-32768-30000.*`). The cell was rerun with a smaller prompt (`lfm25-350m-32768-27500.*`).

## Gates on the same package (bundled catalogue, no override)

`longchat` (#169) PASS and `coderpaste` (#193) PASS:
`console-longchat-1885.log`, `console-coderpaste-1885.log`.
