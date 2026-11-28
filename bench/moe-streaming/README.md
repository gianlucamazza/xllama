# MoE contribution review and parity preparation

This directory supports maintainer review of the existing contributor code. It does not enable MoE streaming in xLlama. The [readiness report](../../docs/research/moe-upstream-readiness.md) maps the source, completed checks and remaining gates; [#482](https://github.com/gianlucamazza/xllama/pull/482) contains the proposed integration sequence.

## Upstream portable candidate

Run the registered `test_moe_expert_cache.cpp` cases through the Linux test preset. These exercise the upstream implementation directly. They do not require fork source or a model.

## Historical independent cache check

The following records the November 16 procedure and its limitations. The November 24 program requires clean-room development. Use the registered upstream tests for the current candidate.

`cache-contract.cpp` exercises the contributor's real cache header with a deterministic in-memory Reader. It checks payloads and actual read offsets, including pinned leases, short/throwing/over-reported reads, LRU eviction, budgets and mixed quant layouts. It is a standalone review harness, not a shipping test or a model-math test.

Use a local checkout of `vetax-by/xllama` at `61bf63ad541510a03a36993e60ec204f98bc0b3e` and a local llama.cpp checkout at `d81235049384534c167caea52b85a694f6103d14`. No model is needed. The commands below create only an extracted review header in a temporary directory; do not apply these patches to the shipping submodule.

```bash
moe_repo="$PWD"
moe_fork=/absolute/path/to/contributor-checkout
moe_llama=/absolute/path/to/pinned-llama-checkout
moe_review="$(mktemp -d)"
python3 - "$moe_fork" "$moe_llama" "$moe_review" <<'PY'
import hashlib, json, subprocess, sys
from pathlib import Path
fork, llama, review = map(Path, sys.argv[1:])
def head(path):
    return subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
assert head(fork) == '61bf63ad541510a03a36993e60ec204f98bc0b3e'
assert head(llama) == 'd81235049384534c167caea52b85a694f6103d14'
lock_bytes = (fork/'ua50/PATCH-LOCK.json').read_bytes()
assert hashlib.sha256(lock_bytes).hexdigest() == '921fc2a5d9724fc28851146266fc67b7627e54a45c8a10775f30c52eb2a20ebe'
for item in json.loads(lock_bytes)['patches']:
    raw = (fork/'ua50'/item['file']).read_bytes()
    assert len(raw) == item['bytes'] and hashlib.sha256(raw).hexdigest() == item['sha256']
text = (fork/'ua50/patch/runtime-llama.patch').read_text()
section = text.split('+++ b/src/ua50/expert_cache.h\n', 1)[1].split('--- /dev/null', 1)[0]
target = review/'src/ua50/expert_cache.h'
target.parent.mkdir(parents=True)
target.write_text('\n'.join(line[1:] for line in section.splitlines() if line.startswith('+'))+'\n')
subprocess.run(['git', 'apply', '--include=src/ua50/expert_cache.h', str((fork/'ua50/patch/cache-ab-llama.patch').resolve())], cwd=review, check=True)
assert hashlib.sha256(target.read_bytes()).hexdigest() == 'a5187e1e42a2f135e67702f6f2c50472f1f327bb0fcde4d3cd04339c85236daf'
assert hashlib.sha256((llama/'ggml/src/ggml-common.h').read_bytes()).hexdigest() == '0061131b615c5721fc88a78feeb22c1f8c450f1c2646a317d80796a653bf595c'
PY
g++ -std=c++17 -Wall -Wextra -Werror \
  -fsanitize=address,undefined -fno-omit-frame-pointer -g \
  -I "$moe_review/src/ua50" -I "$moe_llama/ggml/src" \
  "$moe_repo/bench/moe-streaming/cache-contract.cpp" \
  -o "$moe_review/cache-contract"
ASAN_OPTIONS=detect_leaks=0 "$moe_review/cache-contract"
```

Expected: eight groups pass and exit status is zero. Leak checking was disabled in the recorded cloud run because LeakSanitizer refuses the traced environment. ASan and UBSan instrumentation remained enabled. This result does not prove absence of leaks. On a compatible host, also run with leak detection enabled.

The recorded sensitivity control removed only the extracted header's short-read refusal and rebuilt the harness. It exited 1 at `short_read: expected refusal`, proving that the short-read check rejects that deliberate defect. The contributor checkout was not modified.

## Replay the recorded fixed-slot cache

The two [24B console profiles](../../docs/research/evidence/2026-11-16-moe-24b-console/README.md) contain independently captured requests from the same workload. To replay them from an empty LRU cache on a host:

```bash
python3 scripts/replay-moe-cache.py docs/research/evidence/2026-11-16-moe-24b-console
python3 -m unittest tests/test_replay_moe_cache.py
```

The reader supports the contributor's `UA50TRC1` binary format at the project's [archived package source](https://github.com/gianlucamazza/xllama/blob/75fe271ecb695e05288292aa6f2ee056b5beae56/patches/0005-ua50-cache-ab-llama.patch). Use the version-controlled `result.json` as the trusted receipt. Exit 0 requires matching output and trace SHA-256 digests, checkpoint identity, complete frames and identical ordered slice requests and lengths across profiles. The script compares reconstructed requests, hits, evictions and logical Reader bytes with the binary footer and the output counters. Exit 2 rejects incomplete or inconsistent inputs, including the [partial 8B captures](../../docs/research/evidence/2026-11-16-moe-8b-console/README.md).

The replay counts demand requests for each recorded slot capacity. Add `--reader-timing` to attribute the original synchronous Reader measurements to the prompt and the remainder of the same generation. It checks the one-worker CPU mode, bound token IDs, non-overlapping native frames, cumulative Reader/acquire times, footer and common workload. The [archived timing breakdown](../../docs/research/evidence/2026-11-16-moe-24b-console/README.md#where-the-single-cpu-worker-spent-time) uses application clocks. Lease intervals, physical I/O, performance on held-out routes and timings of a new runtime require additional measurement.

## Prepared numerical campaign

[The 24B declaration](parity-preparation.json) binds four prompt files, the checkpoint identity and a strict same-source/same-kernel comparison diagnostic. It is explicitly **not executable yet**: rendered token IDs, resident-derived continuation IDs, a prepared build receipt and an upstream candidate identity are absent. Nulls are missing prerequisites, not wildcard bindings. There are no model results.

On a host with enough resident-model RAM:

1. Verify the checkpoint bytes and hash, template and prepared source receipt. Confirm identical CPU kernels, nonfused execution, no repacking and one worker for both arms. If that equivalence cannot be established, do not run this declaration as a parity gate.
2. Render each prompt using the actual pinned template and tokenizer. Save exact IDs with one tokenizer-owned BOS. Refuse a prompt above the declared token limit; never silently truncate it.
3. Use the resident reference to obtain a fixed continuation of at least nine tokens, retaining EOS as data if encountered. Freeze the exact token sequence and prefill/decode row positions before running either streamed cache configuration. Export the complete vocabulary row at the end of prefill and after feeding 1, 2, 3, 5 and 9 continuation tokens. The 24B declaration stores the decode positions as zero-based indices into the continuation, so index 0 means the row after feeding its first token. Do not compare independently sampled sequences.
4. Close the resident owner, then run both streamed arms on the same frozen sequence. Bind model/tokenizer/vocabulary hashes and exact token positions in the dump sidecars and campaign receipt; the current comparator does not enforce every one of those provenance fields automatically.
5. Run the existing comparator for every pair of full rows:

```bash
python3 scripts/compare-logits.py reference.bin streamed.bin \
  --max-abs-diff 0 --nmse 0 --top-k 10 --top-k-min 1
```

This strict diagnostic requires zero numeric difference and identical top-token/top-10 sets. A mismatch fails this diagnostic; it is not automatically a model-quality finding. A different floating-point comparison protocol needs a separately reviewed declaration before new candidate measurements. Do not relax the declaration to rewrite a failed result.

Existing comparator tests and synthetic equal/different-row controls were run during preparation. They validate the comparison tool, not MoE model parity. Runtime export/wiring and full-checkpoint parity remain PR 2 work.

## Reproducing the 8B resident reference

The [8B declaration](parity-preparation-8b.json) records a resident-only CPU export from the complete checkpoint. It binds the input and output hashes, frozen token IDs, model settings and build receipt. Candidate identity and both streamed exports are pending.

From the repository root, use the pinned llama.cpp submodule and a host with enough RAM for the full GGUF and inference workspace. Download the exact [checkpoint revision](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B-GGUF/tree/49c14831707011e64d70b2ebd8462ba08d608434), then verify the size and SHA-256 against the declaration before running.

```bash
cmake --preset linux-test
cmake --build build/linux-test --target llama-common -j4
moe_libs="$PWD/build/linux-test/bin"
moe_output="$PWD/build/moe-reference"
mkdir -p "$moe_output"
g++ -std=c++17 -g -O2 -Wall -Wextra -Werror -I include \
  -isystem llama.cpp/include -isystem llama.cpp/ggml/include \
  -isystem llama.cpp/common -isystem llama.cpp/vendor \
  bench/moe-streaming/resident-reference.cpp -L "$moe_libs" \
  -Wl,-rpath,"$moe_libs" -lllama-common -lllama -lggml \
  -o "$moe_output/resident-reference"
GGML_CPU_DISABLE_FUSION=1 "$moe_output/resident-reference" \
  /absolute/path/to/LFM2.5-8B-A1B-Q4_K_M.gguf "$moe_output/8b" \
  bench/moe-streaming/prompts/{greeting-ru,arithmetic-it,code-en,prose-it}.txt
```

Use a fresh output directory for each run and retain it only after a successful exit. Each case contains the rendered text, token IDs and one complete little-endian float32 vocabulary row per declared step on the x86-64 host. Step zero is the end of prefill. Later steps count continuation IDs already fed to the context. The raw files stay outside version control. For each case, compare the token IDs and SHA-256 hashes with the declaration before using a regenerated reference for candidate evaluation. Bind streamed rows to the tokenizer and vocabulary using the hash rules in the declaration.

## Verifying the 8B reference and streamed parity

Use Python with NumPy. Keep the row directories unchanged after capture.
The verifier reads the resident export directly.

```bash
python3 scripts/verify-moe-parity.py bench/moe-streaming/parity-preparation-8b.json \
  /path/to/resident --reference-only
```

The verifier checks `chat-template.jinja`, each case's `rendered.txt` and
`tokens.json`, and every `<case>/row-<step>.bin` against the declaration.
`REFERENCE_VERIFIED` certifies the resident inputs only.

Once the native CPU candidate and both cache captures are available:

```bash
python3 scripts/verify-moe-parity.py bench/moe-streaming/parity-preparation-8b.json \
  /path/to/resident --streamed-2 /path/to/cache-2 --streamed-512 /path/to/cache-512 \
  --receipt /path/to/receipt.json
```

Each streamed row requires a sibling `row-<step>.bin.json` sidecar with
`checkpoint_sha256`, `template_sha256`, `tokenizer_sha256`, `vocab_sha256`,
`vocab_size`, the rendered `prompt`, `prompt_token_ids`, and
`fed_continuation_token_ids` for the first _step_ continuation IDs (empty at
prefill). It also carries `candidate_commit`, integer `cache_slots`,
`build_sha256` and `row_sha256`. The receipt is a JSON object with:

- `schema` set to `xllama-moe-parity-receipt-v1`.
- `declaration_sha256` set to the digest of the declaration file as used for the run.
- `candidate_commit` set to the full native CPU candidate SHA.
- `cpu_kernel_review` set to the reference to the review of kernel equivalence.
- `cpu_settings` with the actual shared `platform`, `cpu`, `compiler`, `cmake`
  and `environment` from the resident build record, plus `threads`, `context`,
  `n_batch` and `n_ubatch` from execution and `repacking` and `flash_attention`
  both set to false. The verifier checks the recorded settings. The review
  must establish that the candidate dispatches the same kernels.
- `streamed_build_sha256` mapping the keys `"2"` and `"512"` to the SHA-256 of
  the actual streamed build artifacts or their build receipts. Use the same
  digest in the corresponding per-row sidecars. A shared binary can have the same digest in both profiles.

Treat the receipt as a maintainer attestation. Review the source and build artifacts before signing off on kernel equivalence. The verifier checks consistency between that receipt, the declaration and the row sidecars.

The verifier checks all inputs before comparing the 48 pairs with the declared
thresholds. For a valid CLI invocation, the JSON report on stdout lists outcomes
in cache, case and step order, along with the receipt and streamed build hashes.
It reports `max_abs_diff`, `nmse`, `top1_match` and `top_k_overlap` for each
pair. A zero reference has NMSE zero for an identical row. For any nonzero
candidate against a zero reference, the report uses JSON `null` for NMSE and
marks the pair `FAIL`. Exit 0 means the requested check passed, 1 means
numerical divergence, and 2 means invalid input or provenance. Keep the report
with the candidate receipt and raw captures. After shared metadata and receipt
validation, row preflight records every missing or invalid row or sidecar in
`row_errors` with its arm, case, step and cause. Any row error withholds all
numerical comparisons. Numerical divergence with valid inputs still reports
every pair, including every failed row.
