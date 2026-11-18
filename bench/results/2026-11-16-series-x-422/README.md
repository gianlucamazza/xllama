# Series X raw runs for issue 422, measured 2026-11-16

Operator: Joshua Henne, Eidon Labs Inc.
Hardware: Xbox Series X in Developer Mode, OS 10.0.26100.9662.
Package: xllama 1.7.0.1626, Game designation, app memory limit 5120 MB.
Package file: xllama_1.7.0.1626_x64.msix.
Package SHA-256: 1ea07c3407e8b9f376c9255a775ad3bf9b644c7bb5aac4f116fdc863ea0c3a50.
Installed package family name: GianlucaMazza.xllama_1.7.0.1626_x64__pj67f1fcj4n14.

What this folder is

These are the raw run files behind the Series X matrix and the first
divergence follow up posted on issue 422.
This is external, operator declared data.
It was measured on the operator's own console and has not been re-measured
by the maintainers.
The receipts and method notes sit next to the data, in the receipts folder
and in this file, so the numbers can be read against how they were taken.

Method, in short

Bench harness runs with fixed 512 token decode and ignore eog.
One dropped warmup plus three recorded runs per cell.
Medians are the reported figures; the CSVs hold every recorded run.
An idle gate was passed before every cell.
Prompts are the repo prompt files, used word for word, bound by hash in
receipts/bindings.json.
Acceptance basis is the issue 374 protocol; the method follows PR 459.

What is in this folder

35 CSV files at the top level.
33 are bench CSVs in schema v2, and all 33 pass the repo validator.
2 are KV CSVs from the repo KV instrument, named with the kv prefix.
The divergence folder holds 27 files: the token sequences, the generation
records, the probe logit dumps and sidecars, and its own RESULTS.md with
the full comparison.
The morning-gpu folder holds 4 files from the first GPU path runs of the
same day: the CPU control runs, the GPU runs, the 350M witness runs, and
the app log excerpt.
The receipts folder holds 11 files: the device identity receipt, the
package bindings, the prompt bindings, the resources excerpt, the Phi-4-mini
push receipt, and the validation suite logs with their exit codes.

Models, bound by the catalogue pins

lfm25-350m, file LFM2.5-350M-QAD-Q4_0.gguf, 219312832 bytes.
SHA-256 3d10b6ab8fc91a919534b9558e266255aca0bbc7f6d015963599aa9e74e05b1d.
lfm25-1.2b-instruct, file LFM2.5-1.2B-Instruct-QAD-Q4_0.gguf, 695755488 bytes.
SHA-256 bb741ebb106d543e9de114b843a3d3d73d51c74b5801e69da2abde821a0cb3e1.
phi4-mini, file Phi-4-mini-instruct-Q4_K_M.gguf, 2491874272 bytes.
SHA-256 88c00229914083cd112853aab84ed51b87bdf6b9ce42f532d8c85c7c63b1730a.
The Phi-4-mini console copy was downloaded back and verified byte for byte
against the pin on the same day; see receipts/PUSH_RECEIPT.md.

Findings and qualifications, stated plainly

Routing gate failure.
In validation on this console and package, validate-api passed 11 of 11.
validate-console passed 9 of 10.
The one failure is the routing gate: a turn of 1769 tokens, over the 1550
token threshold, routed auto to CPU where the gate expects GPU.
The full gate text is in receipts/console_suite.log.
No bench row depends on that routing; the grid drives the backends
directly. It is recorded here as a finding in its own right.

KV instrument qualifications.
The KV CSV rows are stamped xbox-series-s in the host column by default.
That stamp is the instrument's fixed label; the rows were produced on the
Series X named in the receipts, and the raw rows are unaltered.
The KV instrument also shows no GPU difference between layer settings 0
and 99: under 99 it produced CPU path numbers, because that override does
not reach the bench turns mode and the script exposes no backend knob.
The per backend multi turn evidence therefore rests on the KV instrument
on the CPU path plus the timed transcripts, not on a KV GPU arm.

Context 512 cells.
The two n_ctx 512 cells, one per backend on the 1.2B, generated 214 tokens,
not 512.
The 298 token standard prompt leaves room for only 214 new tokens inside a
512 token context, and the rows record exactly that.

Named gap on the package binding.
The staged record for package 1.7.0.1626 carries no CI run id and no
source SHA.
The package is bound by its MSIX SHA-256 and its installed package family
name only, as the receipts show.
No CI run id or source SHA is claimed for it anywhere in this folder.

Divergence, in short

The divergence rerun used the same console, package, model, prompt, chat
template, temperature 0 and 128 token cap as the morning runs, and
reproduced the split exactly: 121 tokens on CPU, 125 tokens on D3D12.
Repeat generations were byte identical on each backend, which confirms
the greedy path.
The first differing token is generated token 35.
The full record, including the named gap on decode time logit rows for
the templated context, is in divergence/RESULTS.md.

Held back

The per cell harness logs and idle sidecars from the matrix day, and the
large per position logit dumps from the divergence rerun, stay in the
operator's own pack and are not in this folder.
Their hashes are recorded in the operator's manifest, and any of them can
be supplied on request.
