# Positioning

What xllama is for, and what it does not claim. Structure lives in
[architecture.md](architecture.md); numbers in [benchmarks.md](benchmarks.md);
Release status in [release-readiness.md](release-readiness.md); LAN protocol in
[api-endpoint.md](api-endpoint.md). Reuse of the inference core in a game is
[reuse-in-a-game.md](reuse-in-a-game.md).

## Thesis

xllama is a **local inference stack** for Xbox Series S and Series X consoles
in Developer Mode. Supported on Series S and Series X. The project measures on
Series S; the Series X runs come from an independent tester.

Two roles:

1. **In-house inference on the console.** Chat, LAN API generation, and image
   generation run on the console. Inference runs on the console. Fine-tuning
   does not: host PEFT runs on a workstation, and in-app training is not
   walked end to end on a console.
2. **MIT-licensed inference code** a studio could reuse in its own published
   game. xllama is MIT-licensed, so a studio could reuse its inference code in
   its own game. That path isn't built or tested: today xllama is a Developer
   Mode app.

The gamepad chat UI is a **demo / showcase shell** so the stack can be
exercised without a game runtime. It is not the product thesis.

## What exists today

- Dual-backend inference: `*.gguf` through llama.cpp; everything else through
  ONNX Runtime GenAI + DirectML. One resident session
  (`SessionHub` — never 2× model in RAM).
- Opt-in OpenAI-compatible **LAN endpoint** — the current integration probe
  for a PC or tool on the same network. Default OFF. Developer Mode / LAN research
  only. Not a public inbound service.
- Headless benches (tok/s, memory, disk, GPU probes).
- SD-Turbo diffusion on the console. Host PEFT on a workstation. In-app
  training is not walked end to end on a console.

## Series X

Series X is supported from 2026-11-15, by the owner's decision, on this
evidence: an independent tester ran package 1.7.0.1495 on a Series X in
Developer Mode and generated text with `lfm25-350m` and `lfm25-1.2b-instruct`
through the LAN API
([field report](https://github.com/gianlucamazza/xllama/discussions/434),
raw runs in `bench/results/series-x-2026-11-13.csv` and
`series-x-end-to-end-2026-11-13.csv`).

What that covers and what it does not:

- **Covered (independent tester, package 1.7.0.1495):** sideload, model
  download, CPU text generation and the LAN API, for those two models. 143
  timed runs and a one-hour run with no failed generation.
- **Later report from the same tester (package 1.7.0.1626, discussion
  #434):** they also ran the GGUF GPU path, image generation, and on-device
  training on their Series X. Raw files they contributed sit in
  `bench/results/2026-11-16-series-x-422/`. That is still their work, not a
  project re-measure.
- **Not project measurements.** The device is operator-declared (rows are
  labelled `xbox-series-x-declared`) and there is no device receipt from the
  project. The project owns no Series X. The project's own published figures
  are Series S figures. The Series X figures in the README were measured by
  an independent tester (Joshua Henne / Eidon Labs Inc., discussion #434)
  and are labelled as theirs.
- **Game designation (their observation).** The tester reported that, after
  the app was set as a Game, LFM2.5-350M averaged about 95.7 tok/s over the
  LAN API end to end (ten 1024-token generations). Before that change, their
  batches averaged about 35.4 to 38.8 tok/s. On-screen timing of one short
  31-token run was about 103 tok/s. Those are different measures from the
  project's Series S benches; they are not compared here.
- **App designation.** With the App designation the app runs at full speed
  for about ten seconds after launch and then at about 40 % of the Game
  speed: the system caps an App's CPU. The remedy is the Game designation,
  which the sideload guide asks for
  ([#457](https://github.com/gianlucamazza/xllama/issues/457)).

## Honest limits

- **Developer Mode only.** Sideloaded onto a console in Developer Mode.
- **Not Microsoft-affiliated.** Independent research. Microsoft-published
  runtimes (UWP, DirectML, ORT) are used as any third party would.
- **Not a Microsoft game-runtime integration.** xllama is MIT-licensed, so a
  studio could reuse its inference code in its own game. That path isn't
  built or tested: today xllama is a Developer Mode app. Nobody has built
  the core with Microsoft's GDK or run it inside a published game. There is
  no packaged library and no engine integration. No ID@Xbox. Do not claim
  otherwise.
- **LAN is a probe, not a product contract.** Unauthenticated,
  foreground-only, single-slot. See [api-endpoint.md](api-endpoint.md).
- **Numbers are measured.** Published tok/s come from `bench/results/` via
  [benchmarks.md](benchmarks.md). Host timings are not Series S results. Do
  not invent throughput.

## Direction

Prove local inference on the console as an in-house inference device.
The next integration surface is the LAN endpoint, not a richer chat app.
Reuse of the MIT-licensed inference core in a published game is a path a
studio could take; it is not built or tested here.
