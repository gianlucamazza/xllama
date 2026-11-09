# Licence text sources

Exact upstream copies for the converted builds re-hosted on the `models-v1`
release. Do not paraphrase these files. `SOURCES.md` itself is not a licence.

The GitHub lane uploads the files listed here to `models-v1` before this
change merges: the catalogue pins them, and a missing asset fails the model
download. This PR does not publish release assets.

## SD-Turbo — Stability AI Community License

- **Local file:** `LICENSES/stabilityai-sd-turbo-LICENSE`
- **Upstream file:** https://huggingface.co/stabilityai/sd-turbo/resolve/main/LICENSE.md
- **Repo revision:** `b261bac6fd2cf515557d5d0707481eafa0485ec2` (Hugging Face
  `stabilityai/sd-turbo`, last updated 2024-07-10)
- **Blob oid:** `1d9ce2ee1067327543544de197291726e4fc57a4` (11,852 bytes)
- **Notice:** `LICENSES/stabilityai-sd-turbo-NOTICE` keeps the §IV.a attribution
  line verbatim, then states that the `models-v1` `sd-turbo-fp16` build is an
  ONNX fp16 conversion of `stabilityai/sd-turbo` (no fine-tuning). The upstream
  repo does not ship a NOTICE file.

## SmolLM2 — Apache-2.0

Hugging Face model cards `HuggingFaceTB/SmolLM2-360M-Instruct` and
`HuggingFaceTB/SmolLM2-1.7B-Instruct` declare `license: apache-2.0` and ship no
`LICENSE` or `NOTICE` file.

- **Local file:** `LICENSES/HuggingFaceTB-SmolLM2-LICENSE`
- **Licence text source:** https://raw.githubusercontent.com/huggingface/smollm/main/LICENSE
  (the SmolLM2 project tree that those model cards point at)
- **Licence-file commit:** `7d94cb44aa1f13ad608b520219b9e1013435f76c` (2024-11-04)
- **Model-card revisions (no licence file):**
  - `HuggingFaceTB/SmolLM2-360M-Instruct` `a10cc1512eabd3dde888204e902eca88bddb4951`
    (2025-09-22)
  - `HuggingFaceTB/SmolLM2-1.7B-Instruct` `31b70e2e869a7173562077fd711b654946d38674`
    (2025-04-21). This SHA is the **licence check** used to confirm the
    card ships no LICENSE file. It is **not** the conversion input for the
    ONNX int4 build (see the 1.7B provenance notes below).
- **Notices (xllama-authored; no upstream NOTICE file):** the conversion
  revision used for each ONNX build is not recorded in the repo, so the
  sidecars say "upstream revision not recorded".
  - `LICENSES/HuggingFaceTB-SmolLM2-360M-cpu-int4-NOTICE` — ONNX int4 from
    `HuggingFaceTB/SmolLM2-360M-Instruct` (`smollm2-360m-cpu-int4_NOTICE`)
  - `LICENSES/HuggingFaceTB-SmolLM2-360M-dml-fp16-v2-NOTICE` — ONNX fp16 from
    HuggingFaceTB/SmolLM2-360M-Instruct, with RMSNorm nodes decomposed into
    primitive ops for DirectML (`smollm2-360m-dml-fp16-v2_NOTICE`,
    sha256 `bfb504953441e7a7dc7e5e1b44ea0a192bb4fb12c296f9ce402b6d28ef7645e2`)
  - `LICENSES/HuggingFaceTB-SmolLM2-1.7B-cpu-int4-NOTICE` — ONNX int4 from
    `HuggingFaceTB/SmolLM2-1.7B-Instruct` (`smollm2-1.7b-cpu-int4_NOTICE`,
    sha256 `ce00add4201f10fb0db32193fc183fb5082ff96e2800319d51fbd6ba18388691`).
    Conversion input revision remains unrecorded. The sidecar is left
    byte-identical so it stays pinned to the `models-v1` asset and
    `uwp/models/manifest.json`.

### SmolLM2-1.7B CPU int4 — conversion provenance

What can be cited for `smollm2-1.7b-cpu-int4`. No conversion SHA is
invented.

**Conversion input revision: not recorded.** Looked for and not found:

- No `build-*-1.7b*` script. The in-tree ORT GenAI builder is
  `scripts/build-fp16-dml-model.sh` (fp16 DirectML, not this asset).
- No GitHub Actions workflow converts models (`build-linux.yml`,
  `build-uwp.yml`, `build-uwp-ort-patched.yml`, `build-uwp-patched.yml`,
  `codeql.yml`, `release-research.yml`).
- The `models-v1` release body describes the 360M merged int4 CPU build,
  not a 1.7B Hugging Face revision.
- Shipped `smollm2-1.7b-cpu-int4_genai_config.json` has no Hugging Face
  revision field (`provider_options` is empty, the CPU default).
- The ONNX protobuf does not contain `31b70e2e869a7173562077fd711b654946d38674`
  or `HuggingFaceTB`.
- Commit `f051587dcad77f499fd7529b9c0bb73191a6fef3` (2026-07-08) records
  "cpu-int4 OK (1.4GB merged)" without an upstream SHA. The four
  weight/tokenizer assets were uploaded to `models-v1` on 2026-07-15
  (`d3c237214dd84efad88f358ee795476569920e9e`).

**Model-card licence check (not the conversion input):**
`HuggingFaceTB/SmolLM2-1.7B-Instruct`
`31b70e2e869a7173562077fd711b654946d38674` (2025-04-21). Upstream
`tokenizer.json` at that revision is about 2.1 MB; models-v1
`smollm2-1.7b-cpu-int4_tokenizer.json` is 3,522,871 bytes (ORT
GenAI-emitted; sha256
`bf346d64f6f0fbcefb4c1b6928a98241467dff36c6fbae5fe1785c4ff90667f4`,
shared with the 360M assets). Architecture numbers at that card
(`hidden_size` 2048, 24 layers, vocab 49152) match the shipped
`genai_config.json`; that is expected for this model family and is not a
revision proof.

**Conversion recipe (as known from in-tree sources, not a builder log):**

1. ORT GenAI model builder:
   `python3 -m onnxruntime_genai.models.builder`
   `-m HuggingFaceTB/SmolLM2-1.7B-Instruct -o <OUT> -p int4 -e cpu`.
   Pattern from `scripts/build-fp16-dml-model.sh` (`-p fp16 -e dml`) and
   the documented 360M DML int4 recipe in `docs/model-selection.md`
   (`-p int4 -e dml`). The exact 1.7B CLI invocation is not in the repo.
   CPU is the shipped EP: empty `provider_options`, same shape as the
   original 360M CPU `genai_config.json` on `models-v1`.
2. Merge for AppContainer: `scripts/merge_onnx_external_data.py`.
   `f051587` says the 1.4 GB artifact was merged;
   `scripts/package-catalogue-ort-model.sh` requires a merged
   `model.onnx`; the shipped asset is a single 1,474,861,442-byte file.

**Tool versions:** shipping ORT GenAI pin is **0.14.1**
(`a30f479af016cb098688726831a9acbb8d19f0b2`, `uwp/packages.config`).
The builder CLI, torch, transformers, and onnx versions used for the
1.7B conversion run are not recorded.

**`models-v1` assets** (sha256 matches `uwp/models/manifest.json`;
verified by downloading the six release assets; no mismatch):

- `smollm2-1.7b-cpu-int4_LICENSE` (11357 bytes)
  sha256 `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4`
- `smollm2-1.7b-cpu-int4_NOTICE` (170 bytes)
  sha256 `ce00add4201f10fb0db32193fc183fb5082ff96e2800319d51fbd6ba18388691`
- `smollm2-1.7b-cpu-int4_genai_config.json` (1458 bytes)
  sha256 `c76e19ce57f3463f19588142c8266f805d18cb93a38288d8c5b9d35dae030753`
- `smollm2-1.7b-cpu-int4_tokenizer.json` (3522871 bytes)
  sha256 `bf346d64f6f0fbcefb4c1b6928a98241467dff36c6fbae5fe1785c4ff90667f4`
- `smollm2-1.7b-cpu-int4_tokenizer_config.json` (452 bytes)
  sha256 `47b4a92de79d133d06b4cec56e9b8226d2be0b62936209577181d5a76444a08a`
- `smollm2-1.7b-cpu-int4_model.onnx` (1474861442 bytes)
  sha256 `8b46fe437133f584ddab2443f92cf60e531d17318a81a3c99e5e0993f6e7020e`

## Qwen3.5-0.8B — Apache-2.0

- **Local file:** `LICENSES/Qwen-Qwen3.5-0.8B-LICENSE`
- **Upstream file:** https://huggingface.co/Qwen/Qwen3.5-0.8B/resolve/main/LICENSE
- **Repo revision:** `2fc06364715b967f1860aea9cf38778875588b17` (Hugging Face
  `Qwen/Qwen3.5-0.8B`, last updated 2026-03-02)
- **Blob oid:** `f938136e3adacfd92be087f6e113b5d6d97f678f` (11,544 bytes, CRLF)
- No upstream `NOTICE` file. The `models-v1` GGUF is the unsloth
  `Qwen3.5-0.8B-Q4_K_M.gguf` re-hosted unchanged
  (`LICENSES/Qwen-Qwen3.5-0.8B-NOTICE` → `qwen35-0.8b_NOTICE`).

## TAESD — MIT

The Hugging Face model card `madebyollin/taesd` declares `license: mit` and
ships no `LICENSE` file. The copyright line is taken from the project's GitHub
licence.

- **Local file:** `LICENSES/madebyollin-taesd-LICENSE`
- **Licence text source:** https://raw.githubusercontent.com/madebyollin/taesd/main/LICENSE
- **Licence-file commit:** `f38b7ed32dd0aa17cb8ead2f830ec5f9ecf3475b` (2023-04-16)
- **HF model-card revision:** `614f76814bbe30edbe2e627ace1c2234c81a2c0e` (2024-11-07)

## ONNX Runtime — MIT

Patched `onnxruntime.dll` pin (`vendor/onnxruntime-patched/`), not a
`models-v1` asset.

- **Local file:** `LICENSES/onnxruntime-LICENSE`
- **Upstream file:** https://github.com/microsoft/onnxruntime/blob/v1.24.4/LICENSE
- **Repo revision:** tag `v1.24.4` (`2d924974ef147392ced8409d36bd6d2e7fcc8a74`)
- **Blob oid:** `48bc6bb4996ac924359e8e28b9ae88970e5ed3fc` (1,073 bytes)
- **Third-party notices:** `LICENSES/onnxruntime-ThirdPartyNotices.txt` from the
  same tag (blob `fbd9f9a95f6013d8ecaef81e02b0033e5882a675`, 325,054 bytes)

## ONNX Runtime GenAI — MIT

Patched `onnxruntime-genai.dll` pin (`vendor/onnxruntime-genai-patched/`),
not a `models-v1` asset.

- **Local file:** `LICENSES/onnxruntime-genai-LICENSE`
- **Upstream file:** https://github.com/microsoft/onnxruntime-genai/blob/a30f479af016cb098688726831a9acbb8d19f0b2/LICENSE
- **Repo revision:** `a30f479af016cb098688726831a9acbb8d19f0b2` (`rel-0.14.1`)
- **Blob oid:** `9e841e7a26e4eb057b24511e7b92d42b257a80e5` (1,141 bytes)
- **Third-party notices:** `LICENSES/onnxruntime-genai-ThirdPartyNotices.txt`
  from the same commit (blob `e7f603e41f447ab01d1e2cbfb3de4fcf48e4e435`,
  339,676 bytes)
