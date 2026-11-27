# ADR 0011: Standard-First and Upstream-First Architecture

Status: Proposed for maintainer review, 2026-11-27. Applies to new work and changed contracts.

## Context

A single memory-constrained AppContainer serves two inference backends and several callers. Each additional parser, vendor overlay or frontend-specific policy creates another behavior to qualify. Existing console receipts establish useful behavior which a superficially more conventional replacement could regress.

## Decision

1. Start with the applicable specification, upstream contract or reference implementation. Record its version and distinguish formal standards, provider APIs, ecosystem conventions, experiments and proposals. Reuse and compose existing components before adding dependencies. Document the supported subset before extending a public contract. Retain backward compatibility by default and make extensions discoverable.
2. Keep deterministic validation, admission, lifecycle and error handling around probabilistic generation. Give each shared policy one owner. Keep portable decisions in the C++ core and keep WinRT, package and device operations at the boundary. Apply least privilege and propagate failures with actionable errors.
3. Keep upstream pins and their local changes reviewable. An overlay needs a base, reason, regression test and removal gate. Evaluate candidate pins in isolation. Offer general fixes upstream after review.
4. Introduce optimizations or platform-specific implementation only for a measured need. Preserve a known-good route, measure costs and parity, and record the precise hardware, package and configuration before promotion. No speculative dependency or new protocol solely to imitate another runtime.

## Alternatives and consequences

An embedded orchestrator or a dynamic backend loader would add deployment, permission and lifecycle obligations without a demonstrated consumer on Xbox. Use an external client and the current statically composed backends. A blanket ban on local patches would forfeit working AppContainer support. Use the patch inventory and retirement evidence instead. A full OpenAI compatibility claim would hide meaningful wire differences. Test and describe each supported operation. This decision costs maintainer and device time at promotion while permitting host-only proposals earlier.

## Exceptions and review

An exception belongs in the owning contract or relevant ADR. Record the missing upstream capability, alternatives, affected callers and SKU, opt-in or rollback behavior, owner and a measurable removal or review condition. An upstream fix, changed pin, incompatible client behavior or new device evidence reopens the decision.

### Standards & Interoperability Review

At each product release preparation and at least quarterly, the release maintainer adds a dated checkpoint to the existing engineering consolidation tracker (#395 or its successor). Use the documentation ownership map for the current contract. Review changed pins/overlays, public wire fields and clients, shared policy callers, package provenance, threat boundaries and device evidence. Record source and package IDs, observed checks, exceptions, blockers and linked follow-up issues. A PR changing one of those boundaries includes the relevant check in its existing verification section. Compare open branches and issues first. A recurring review does not grant authority to merge, sign, publish or promote a capability.
