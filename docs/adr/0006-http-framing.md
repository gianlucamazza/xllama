# ADR 0006: Exact HTTP framing for the LAN API

Status: Accepted (2026-10-14)

Accepted by the owner's delegation. Promotes audit Proposal B
([adr-proposals.md](../audit/excellence/adr-proposals.md)) for finding F02.

## Context

The LAN API read its requests with a substring search for `content-length:`,
`atoll`, a clamp at 8 MiB and no completion check. On Series S (1.6.0.1218) a
negative length and a body 100 bytes shorter than declared both returned 200 and
appended a preference sample. A stalled body held the connection thread past 40 s
with no answer. The reader was WinRT-only, so the host suite could not test any
of it.

## Decision

Framing decisions live in `include/xllama/http_framing.h` (WinRT-free, host
tested); the UWP transport only moves bytes and enforces deadlines.

- The request line is exactly `METHOD SP target SP HTTP/x.y`, and header names
  are complete RFC 9110 tokens matched case-insensitively. A field name with a
  space before the colon, a folded line or a field with no colon is 400.
- `Content-Length` is plain decimal, with no sign, no list and no overflow.
  Repeats must agree; otherwise the request is 400.
- Any `Transfer-Encoding` is 411, as chunked was before.
- A body over 8 MiB is 413, not clamped. A head over 64 KiB is 431.
- A body that ends before its declared length is 400, and nothing is
  dispatched.
- Deadlines: 10 s for the head and 15 s for the body, each from its start. Past
  either, the request is 408 and the read is cancelled.
  - An 8 MiB body took 1.3–1.4 s from the host to Series S. The body deadline
    is about ten times that, so a ~5 Mbit/s client still fits.
  - Three times the measured value would refuse slower clients that are
    legitimate.

## Consequences

Malformed clients that used to get 200 now get 4xx. The limits are part of the
LAN contract in [api-endpoint.md](../api-endpoint.md).
`validate-api.sh framing` reproduces the audit cases on the console and checks
that a refused POST leaves no sample. On a build without this ADR the gate
itself appends probe samples.

Bounded connection admission (a cap on concurrent connections) is not part of
this decision. Each connection is still one thread-pool work item, bounded in
time now, not in count.

## Alternatives

- Length checks inside the UWP reader only: fixes the wire cases but keeps the
  rules out of the host suite.
- A general HTTP library: a larger AppContainer package for a narrow LAN probe,
  with no evidence that it is needed.
