# Tool proposal budgeting implementation plan

> For agentic workers, use superpowers:executing-plans. The checkboxes track implementation and validation.

## Goal

Keep the accepted tool prompt aligned with its template metadata and give Open WebUI actionable errors for unsupported models and context overflow.

## Architecture

Each budget probe owns the complete render. The budget code retains the selected probe and tokenizes its existing no-tools baseline only on overflow. The LAN adapter maps the result to HTTP and the client controls its tool inventory. The existing SDK string renderer delegates to the same budgeting search.

## Contract

[LAN API](../../api-endpoint.md) and [ADR 0004](../../adr/0004-tool-proposals.md).

- [x] Reproduce the oversized shell argument failure and send gate fixtures to curl over stdin.
- [x] Verify that a failed history probe cannot replace the selected prompt's grammar, stops or preserved tokens. Cover a template that rejects a suffix starting partway through a turn.
- [x] Reuse the template's no-tools comparison, including for a named choice. Preserve unknown measurements and omit the error parameter when the overflow cannot be attributed to one component.
- [x] Reject ONNX tool requests before loading the model, give them a stable error code and document model selection and Open WebUI's optional Tool Search. Exercise refusal, GGUF preservation and plain chat with `MODEL` distinct from `TOOLS_MODEL` so that `/v1/models` distinguishes a loaded model from its startup fallback.
- [x] Run the Linux build, test suite, formatters and coherence check.
- [ ] Run the gate on Series S against the unified artifact for the final PR head.
