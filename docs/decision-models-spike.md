# Decision-model spike (H2O-Lightning-4B)

**Status: NO-GO on pin `d812350`.** Cells stay `pending`, not zero. No host
or console JevBench numbers were measured. Close this spike on the current
pin.

Limited, owner-approved validation of System One decision models in xllama.
One-way scoring only: a single prefill, then softmax over the candidate label
tokens. No generation on the decider path. No llama.cpp bump.

**204/231** on public JevBench is **vendor-claimed** (H2O-Lightning-4B Q4_K_M
card). It is not an xllama measurement. Unmeasured Series S cells stay
`pending` in [bench/results/decision-models/template.json](../bench/results/decision-models/template.json).

## Load gate (pin `d812350`) — **FAILED**

Checked first, as required. **The stock GGUF does not load on this pin.**
No llama.cpp backport and no #30361 cherry-pick were made.

Stock file `h2o-lightning-4b-stock-Q4_K_M.gguf` from
[h2oai/h2o-lightning-4b-gguf](https://huggingface.co/h2oai/h2o-lightning-4b-gguf)
(SHA256 `6356a49ed0812da07d772df752ed9f57bc3da6d0a10c7df19f5c7726af6943d9`,
2 708 804 768 bytes, runtime "standard llama.cpp + shim").

GGUF header (`gguf_init_from_file`, no_alloc):

| Key                           | Value                                                                                     |
| ----------------------------- | ----------------------------------------------------------------------------------------- |
| `general.architecture`        | `qwen35`                                                                                  |
| `general.name`                | H2O Lightning 4B                                                                          |
| `qwen35.block_count`          | 33                                                                                        |
| `qwen35.embedding_length`     | 2560                                                                                      |
| `qwen35.nextn_predict_layers` | 1                                                                                         |
| `qwen35.decision.type`        | **absent**                                                                                |
| `qwen35.decision.labels`      | **absent**                                                                                |
| tensors                       | 426; trunk `blk.0`–`blk.31` only; **no `blk.32.*`**; no `output.weight` (tied embeddings) |

Host load with this pin (`xllama-cli --decide`, `llama_model_default_params`,
`load_mtp=false`):

```text
llama_model_load: error loading model: check_tensor_dims: tensor 'blk.32.attn_norm.weight' not found
decide FAIL: failed to load model: …/h2o-lightning-4b-stock-Q4_K_M.gguf
```

Why: `n_layer() = block_count - nextn = 32`, so the pin treats `blk.32` as the
MTP block. `qwen35.cpp` sets `mtp_flags = TENSOR_SKIP` when `load_mtp` is
false, but `TENSOR_SKIP` alone still goes through
`check_tensor_dims(..., required=true)`. The stock file has no MTP weights.
Other arches on this pin pair `TENSOR_SKIP` with `TENSOR_NOT_REQUIRED`; qwen35
does not. That is a pin/loader mismatch, not a missing `qwen35` arch.

`COMMON_DECISION_TYPE_H2O_LIGHTNING`, `LLM_KV_DECISION_LABELS`, and
`llama_model_decision_label` are **not** on this pin. The non-stock
`h2o-lightning-4b-Q4_K_M.gguf` (#30361, decision type `h2o-lightning`) was not
loaded.

Scoring helpers and the JevBench harness still land so a later pin (or a GGUF
that includes `blk.32`) can run them. They are not a measured Series S result.

### No-bump `kv_overrides` workaround — **FAILED**

Tried next, on host, against this pin's libllama (`d812350`).
`llama_model_params.kv_overrides` is applied before hparams are read
(`llama-model.cpp` `get_key(LLM_KV_BLOCK_COUNT)` / `LLM_KV_NEXTN_PREDICT_LAYERS`).
Exact key names from `llama.cpp/src/llama-arch.cpp`:

| Enum                          | Format                    | Override used                       |
| ----------------------------- | ------------------------- | ----------------------------------- |
| `LLM_KV_BLOCK_COUNT`          | `%s.block_count`          | `qwen35.block_count=int:32`         |
| `LLM_KV_NEXTN_PREDICT_LAYERS` | `%s.nextn_predict_layers` | `qwen35.nextn_predict_layers=int:0` |

The loader accepted both overrides, then failed on the next hparam that is
sized to the original `block_count`:

```text
validate_override: Using metadata override (  int) 'qwen35.block_count' = 32
validate_override: Using metadata override (  int) 'qwen35.nextn_predict_layers' = 0
llama_model_load: error loading model: error loading model hyperparameters: key qwen35.attention.recurrent_layers has wrong array length; expected 32, got 33
llama_model_load_from_file_impl: failed to load model
```

`qwen35.attention.recurrent_layers` is `arr[bool,33]` in the stock file
(`LLM_KV_ATTENTION_RECURRENT_LAYERS` = `%s.attention.recurrent_layers`).
`qwen35.cpp` `get_key_or_arr(..., hparams.n_layer_all)` requires that array
length to equal the (now overridden) block count. `kv_overrides` can replace a
scalar; it cannot truncate that array. No further keys were tried. No
llama.cpp edit. No pin bump. No host JevBench sanity (the model still does
not load).

## GGUF KV labels

xllama reads metadata with the gguf API in `read_gguf_decision_meta`
(`gguf_init_from_file`, `no_alloc`):

1. `general.architecture`
2. `{arch}.decision.type` if present
3. `{arch}.decision.labels` (string array) if present

It does **not** call `llama_model_decision_label`. On the stock file those
keys are absent, so labels fall back to `serve_config.json` (255 entries) with
a leading space applied, matching the #30361 converter:

```text
# the answer follows "Answer:", so the token of a label has a leading space
self.gguf_writer.add_decision_labels([" " + label for label in cfg["labels"]…])
```

## Prompt contract (#30361, mirrored in xllama)

Cited from ggml-org/llama.cpp#30361 / commit `7e3eeee04`
`tools/server/server-decision.cpp` and the H2O converter in that PR:

| Contract         | Upstream                                                                       | xllama                                        |
| ---------------- | ------------------------------------------------------------------------------ | --------------------------------------------- |
| Max options      | `n_options_max = labels.size()` after init (~202); converter `max_options` 255 | `kH2oMaxOptions = 255`                        |
| State cap        | `DECISION_H2O_MAX_STATE_TOKENS = 32000` (line 24); `h2o_cut_state` ~725–745    | `kH2oMaxStateTokens`; `cut_h2o_state`         |
| Options JSON     | `render_h2o` ~782–824: `{label, key, description}`                             | `render_h2o_options_json`                     |
| User turn        | converter `_systemone_template`: `{{ o.label \| trim }}) {{ o.key }}: …`       | `render_decision_user` trims the label        |
| Prefill          | `prompt["prefill"]` = `Answer:`                                                | `kDefaultDecisionPrefill`                     |
| Score tokens     | GGUF label texts (leading space)                                               | `h2o_score_labels_for`                        |
| Assistant suffix | `_systemone_template`: `" {{ '\\n\\n' }} {{ '\\n\\n' }}" + prefill`            | `<\|im_start\|>assistant\n \n\n  \n\nAnswer:` |

`render_h2o` is the JSON object handed to the `systemone` Jinja template. The
stock GGUF has no `systemone` template, so xllama applies that layout in C++
instead of `llama_model_chat_template(model, "systemone")`.

## Multi-token labels

#30361 init (~171–178) **rejects** a label that does not tokenize to one
distinct token. This pin cannot do that check against
`llama_model_decision_label`.

xllama scores the **first token** of each label at the last prompt position
(`llama_get_logits_ith(-1)`), softmax over the candidate set only.
`LabelScore.token_count` records the full length. Two labels that share a
first token are a collision (`first_token_collision`). That is a **known
limitation** versus #30361 and versus the upstream shim, which only ever
read single-token labels.

## Known mismatches with the stock shim

The stock GGUF is served by `package/h2o_lightning_shim.py` (plain ChatML,
thinking off):

|                  | #30361 / this spike                           | Stock shim                                                       |
| ---------------- | --------------------------------------------- | ---------------------------------------------------------------- |
| Assistant prefix | systemone blank lines (see above)             | `<think>\n\n</think>\n\n`                                        |
| Labels in GGUF   | `{arch}.decision.labels` with a leading space | absent; shim uses `serve_config.json` letters                    |
| User layout      | systemone render of `render_h2o` JSON         | `record:/question:/A) name: desc` (same visible text after trim) |
| Decision type    | `h2o-lightning` on the non-stock file         | none (ordinary chat)                                             |

Accuracy on stock-Q4_K_M with the systemone suffix may differ from the shim's
vendor-claimed 204/231. Do not treat a host or console number as that card
figure.

## Harness

Public JevBench 231 = 48 easy + 72 original + 111 hard from
[fstandhartinger/jevbench](https://github.com/fstandhartinger/jevbench) at
`609a664e81f46e03731600bd7bd9f1b7a61257c5`. Fetch:

```bash
./scripts/fetch-jevbench.sh
```

Host CPU (label every result `host-cpu` / not Xbox):

```bash
./scripts/run-jevbench.sh \
  --job bench/results/decision-models/job-score.example.json \
  --model /path/to/h2o-lightning-4b-stock-Q4_K_M.gguf
```

Phi-4-mini JSON-grammar baseline: `job-phi4-json.example.json` (`mode` =
`generate-json`).

One-shot CLI:

```bash
./build/linux-test/bin/xllama-cli -m /path/to/model.gguf \
  --decide --label A --label B -p '<systemone prompt ending in Answer:>'
```

`--label A` is scored as `" A"`.

## Series S procedure

Live Xbox cannot run from the Linux CI VM. After a deployed unified or
llamacpp package:

1. Source `~/.config/xllama/xbox-env` (stays outside the repo).
2. Put `h2o-lightning-4b-stock-Q4_K_M.gguf` in LocalState (operator path;
   no hardcoded host or engine index).
3. `./scripts/fetch-jevbench.sh`
4. Edit a copy of `bench/results/decision-models/job-score-console.example.json`
   so `model_path`, `backend` (`cpu` or `d3d12`), and `n_gpu_layers` match the
   console layout.
5. `./scripts/run-jevbench.sh --console --job that.json`
6. The script uploads the JSONL, `decide_job.json`, and `decide.flag`, waits
   for `decide.done`, and fetches the report.
7. Copy measured fields into a new file under
   `bench/results/decision-models/`. Leave `template.json` as the pending
   skeleton. Keep `vendor_claimed` labelled vendor-claimed.

`decide.flag` is store-SKU gated with the other headless flags.

## Defaults (filenames only)

| Constant                    | Filename                             |
| --------------------------- | ------------------------------------ |
| `kDefaultLightningGgufName` | `h2o-lightning-4b-stock-Q4_K_M.gguf` |
| `kDefaultPhi4MiniName`      | `phi4-mini`                          |
| `kDefaultJevbenchDirName`   | `jevbench`                           |
| `kDefaultDecideJobName`     | `decide_job.json`                    |

Resolved by the existing path helpers / the job JSON. Never a host, console
path, or D3D12 engine index.
