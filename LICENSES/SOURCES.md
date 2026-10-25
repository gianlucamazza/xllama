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
    (2025-04-21)
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
    `HuggingFaceTB/SmolLM2-1.7B-Instruct` (`smollm2-1.7b-cpu-int4_NOTICE`)

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
