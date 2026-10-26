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
