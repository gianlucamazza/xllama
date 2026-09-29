# ADR 0002: Enforce CI gates on main with a repository ruleset

Status: Accepted (2026-09-29)

Accepted by the owner ("sei autorizzato, completa tu"; sole maintainer).
Applied as ruleset `24213666` (#264).

## Context

Read-only inspection on 2026-09-29, before the change:

- Rulesets: none (`gh api repos/gianlucamazza/xllama/rulesets` returns `[]`).
- Legacy branch protection on `main`: 1 approving review, `enforce_admins`
  false, force pushes and deletions blocked, required status checks **not
  enabled**. The owner therefore merges through the admin exemption, and a red
  or missing check does not block a merge.
- The repository now takes external contributions (#261), so the manual merge
  gate carries real risk.

Checks that run on every PR and on pushes to `main`:

| Workflow      | Check context                                   |
| ------------- | ----------------------------------------------- |
| `build-linux` | `build (linux)` (named explicitly since #278)   |
| `build-uwp`   | `build (unified, true, xllama-appx)`            |
| `build-uwp`   | `build (llamacpp, false, xllama-appx-llamacpp)` |
| `codeql`      | `analyze (c-cpp)`                               |

`asan` (manual) and `store` (Store SKU dispatch) are skipped on normal runs
and must not be required.

## Decision

Add one active ruleset on the default branch:

```json
{
  "name": "main",
  "target": "branch",
  "enforcement": "active",
  "conditions": {
    "ref_name": { "include": ["~DEFAULT_BRANCH"], "exclude": [] }
  },
  "bypass_actors": [
    {
      "actor_id": 5,
      "actor_type": "RepositoryRole",
      "bypass_mode": "pull_request"
    }
  ],
  "rules": [
    { "type": "deletion" },
    { "type": "non_fast_forward" },
    {
      "type": "pull_request",
      "parameters": {
        "required_approving_review_count": 0,
        "dismiss_stale_reviews_on_push": false,
        "require_code_owner_review": false,
        "require_last_push_approval": false,
        "required_review_thread_resolution": false
      }
    },
    {
      "type": "required_status_checks",
      "parameters": {
        "strict_required_status_checks_policy": true,
        "required_status_checks": [
          { "context": "build (linux)", "integration_id": 15368 },
          {
            "context": "build (unified, true, xllama-appx)",
            "integration_id": 15368
          },
          {
            "context": "build (llamacpp, false, xllama-appx-llamacpp)",
            "integration_id": 15368
          },
          { "context": "analyze (c-cpp)", "integration_id": 15368 }
        ]
      }
    }
  ]
}
```

- Every change to `main` goes through a PR with the four checks green on an
  up-to-date head. `integration_id` 15368 (GitHub Actions) pins each context to
  its producer.
- The approval count stays 0 for now: a sole maintainer cannot approve their
  own PR. External PRs still get an explicit owner review before merge.
- Admins may bypass only through a PR (`bypass_mode: pull_request`), which
  leaves an audit record. Direct pushes stay blocked.
- UWP/runtime PRs record the exact SHA, CI package identity and live Xbox
  evidence in the PR body. Hardware evidence stays a review gate. No synthetic
  always-green check stands in for it.
- The legacy protection stays in place until the ruleset has been validated,
  then its review requirement is removed so the two policies do not diverge.

## Rollout and rollback

Applied on 2026-09-29. `evaluate` mode is not available on this plan, so the
ruleset went straight to `active`:

1. Ruleset `24213666` created from the payload above.
2. Legacy `required_pull_request_reviews` removed. Force-push and deletion
   blocks stay on the legacy protection as a second layer.
3. Merge settings: squash only with the PR title and body, branches deleted
   on merge, auto-merge and update-branch enabled. Dependabot security update
   PRs enabled.
4. First gated PR: this ADR update. It reported `BLOCKED` while the required
   checks ran, and merged without `--admin` once they were green.

Emergency rollback: set `enforcement` to `disabled` via
`gh api -X PUT repos/gianlucamazza/xllama/rulesets/<id>`. The owner (repo
admin) is the only authority. Record the reason in the next PR.

## Consequences

A red, missing or stale check blocks the merge. Renaming a required job
requires updating the ruleset in the same PR. `strict` forces a rebase or
update before merge, which costs one extra CI cycle per stale PR.

## Alternatives

- Keep the manual gate: no enforcement, and it relies on memory.
- Require 1 approval: blocks the sole maintainer without an external reviewer.
- Legacy branch protection only: works, but rulesets are the supported
  surface, and bypass through a PR is auditable.
