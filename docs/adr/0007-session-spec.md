# ADR 0007: One resolved session specification for GUI and API

Status: Accepted (2026-10-14)

Accepted by the owner's delegation. Promotes audit Proposal A
([adr-proposals.md](../audit/excellence/adr-proposals.md)) for finding F03.

## Context

The GUI (`MainPageController::EnsureSession`), LAN chat, embeddings and pull each
turned a catalogue entry into `SessionParams` themselves. Only the GUI resolved
the catalogue `lora`. The hub reused the resident session by model id (plus GPU
layers), and the GUI short-circuited on the id alone. So the first surface to
load a model decided whether its adapter was applied. On the audit's Series S
fixture the same request returned the LoRA marker after a GUI-first load and a
generic answer after an API-first one.

## Decision

`include/xllama/session_spec.h` (WinRT-free, host tested) holds
`resolve_session_params`, the one resolver from catalogue fields to load
parameters:

- catalogue `n_ctx` through `resolve_n_ctx`;
- backend from the catalogue kind;
- GGUF GPU layers through `resolve_gguf_gpu_layers` and the operator knob;
- the adapter path resolved against the model directory, and its scale.

It also holds `same_load`, the load identity: model, backend, context, GPU
layers, adapter path and scale, and the q8 KV flag. Threads, batch sizes and
sampling are not part of it.

Every surface builds its parameters with `CatalogueSessionParams` (uwp, next to
the manifest loader). `SessionHub::ensure_locked` reuses the resident session
only when `same_load` holds; anything else reloads, freeing the old session
first. A different embedding `num_ctx` therefore reloads through the same rule,
not a special case.

A pull of the resident model invalidates it (`invalidate_locked`), since its
files may have changed under the same parameters. Personalize promotion and
training already reset the hub.

## Consequences

The adapter is applied whichever surface loads the model first. Some reuses that
used to be silent become reloads: a GUI turn after an API request with a
different context, and a pull of the model in use. The GUI parses the catalogue
on every turn instead of only on a model switch. That is a small file read,
negligible next to a turn.

## Alternatives

- Copy the GUI's LoRA lines into the API: a smaller patch, but it keeps four
  resolvers and reuse by id only.
- Always reload: correct, but it gives up the resident-session reuse that
  multi-turn latency depends on.
