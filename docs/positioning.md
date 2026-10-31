# Positioning

What xllama is for, and what it does not claim. Structure lives in
[architecture.md](architecture.md); numbers in [benchmarks.md](benchmarks.md);
Release status in [release-readiness.md](release-readiness.md); LAN protocol in
[api-endpoint.md](api-endpoint.md).

## Thesis

xllama is a **local inference stack** for Xbox Series S consoles.
Tested and measured on Series S. Not yet tested on Series X. The intended uses
are **games** (NPC dialogue, agents) and **on-console AI assistants**.

The gamepad chat UI is a **demo / showcase shell** so the stack can be
exercised without a game runtime. It is not the product thesis.

## What exists today

- Dual-backend inference: `*.gguf` through llama.cpp; everything else through
  ONNX Runtime GenAI + DirectML. One resident session
  (`SessionHub` — never 2× model in RAM).
- Opt-in OpenAI-compatible **LAN endpoint** — the current integration probe
  for a PC or tool on the same network. Default OFF. Dev Mode / LAN research
  only. Not a game SDK and not a public inbound service.
- Headless benches (tok/s, memory, disk, GPU probes).
- SD-Turbo diffusion and on-device / host training lanes.

## Honest limits

- **Developer Mode only.** Sideloaded onto a console in Developer Mode.
- **Not Microsoft-affiliated.** Independent research. Microsoft-published
  runtimes (UWP, DirectML, ORT) are used as any third party would.
- **Not an Xbox / GDK game SDK.** No AAA drop-in plugin, no ID@Xbox, no
  published engine integration. Do not claim otherwise.
- **LAN is a probe, not a product contract.** Unauthenticated,
  foreground-only, single-slot. See [api-endpoint.md](api-endpoint.md).
- **Numbers are measured.** Published tok/s come from `bench/results/` via
  [benchmarks.md](benchmarks.md). Host timings are not Series S results. Do
  not invent throughput.

## Direction

Prove local inference on the console toward game-runtime and assistant use.
The next integration surface is the LAN endpoint, not a richer chat app.
