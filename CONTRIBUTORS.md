# Contributors

People who have shaped xllama. This list is not exhaustive; see `git log` and
merged pull requests for the full record.

- **Gianluca Mazza** ([@gianlucamazza](https://github.com/gianlucamazza)) —
  creator and maintainer.
- [@duranlisandro](https://github.com/duranlisandro) —
  [#348](https://github.com/gianlucamazza/xllama/pull/348) streamed
  `POST /v1/chat/completions` as SSE when `stream: true`.
  [#358](https://github.com/gianlucamazza/xllama/pull/358) scoped harness
  log evidence to the current run so a validation gate does not treat a
  previous run's lines as evidence.
- [@maxmorrison74](https://github.com/maxmorrison74) —
  [#350](https://github.com/gianlucamazza/xllama/pull/350) added
  process-wide compute admission for GUI, LAN API, and training jobs; that
  work landed via [#354](https://github.com/gianlucamazza/xllama/pull/354).
  Optional debug dashboard idea, from
  [xllama-on-steroids](https://github.com/maxmorrison74/xllama-on-steroids)
  as cited by [#357](https://github.com/gianlucamazza/xllama/pull/357) (and
  closed [#351](https://github.com/gianlucamazza/xllama/pull/351)). Rebuilt
  in-tree as an optional debug dashboard of real runtime readings; the
  fork's rebrand, background, soundtrack, and footer credit were not
  taken.
- [@EgbertJiang](https://github.com/EgbertJiang) — the Gemma 4 chat
  template, from the
  [Xbox One S fork](https://github.com/EgbertJiang/xllama/tree/codex/xbox-one-s-compat).
  Rebuilt in-tree with its own tests; the rest of that fork is reviewed in
  [docs/community-forks.md](docs/community-forks.md#egbertjiang-xbox-one-s).
- [@Chorss](https://github.com/Chorss) —
  [#504](https://github.com/gianlucamazza/xllama/pull/504) opt-in F32 bias ADD
  and [#511](https://github.com/gianlucamazza/xllama/pull/511) opt-in F32 RoPE on the D3D12 GGUF path. The interactive thread override implements Chorss's idea in [#515](https://github.com/gianlucamazza/xllama/pull/515).
- [@tayhill1](https://github.com/tayhill1) made KV benchmark runs honor GPU layer settings and hardware declarations in [#489](https://github.com/gianlucamazza/xllama/pull/489).
- [@vetax-by](https://github.com/vetax-by) designed the SSD expert-streaming experiment and reviewed the portable Reader/cache/lease foundation in [#668](https://github.com/gianlucamazza/xllama/pull/668).
- [@christopherbearman](https://github.com/christopherbearman) contributed the ideas behind the in-tree D3D12 quantized kernels and the optional LCM scheduler. [#513](https://github.com/gianlucamazza/xllama/pull/513) · [#514](https://github.com/gianlucamazza/xllama/pull/514)
- [@bjoern2088](https://github.com/bjoern2088) demonstrated hybrid-cache prefix snapshots. Upstream implemented the design in [#512](https://github.com/gianlucamazza/xllama/pull/512).
- [@SamirCharleston](https://github.com/SamirCharleston) identified the missing CPU model-load warnings. Upstream enabled logging consistently in [#345](https://github.com/gianlucamazza/xllama/pull/345).
