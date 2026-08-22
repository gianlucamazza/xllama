# Security Policy

## Supported versions

xllama moves fast and only the latest tagged release receives security fixes.

| Version        | Supported |
| -------------- | --------- |
| latest release | ✅        |
| older tags     | ❌        |

## Reporting a vulnerability

Use GitHub's **private vulnerability reporting**:
<https://github.com/gianlucamazza/xllama/security/advisories/new>

Please do not open public issues for suspected vulnerabilities.

Include: affected version/commit, build target (UWP/x64 or Linux), reproduction
steps or PoC, and expected vs actual behaviour.

## Scope

In scope:

- The LAN HTTP endpoint (`SessionHub` surface): request parsing, auth-less
  exposure defaults, path traversal via persisted state files
- MSIX packaging / AppContainer sandbox escapes reachable from app content
- Model-loading paths (GGUF / ONNX) parsing untrusted model files

Out of scope:

- Anything requiring **retail-console** modification or hardware attacks
  (the project ships Dev Mode only)
- Attacks that require an already-compromised dev machine
- Reports about bundled upstream code where the fix belongs upstream
  (`llama.cpp`, ONNX Runtime) — still welcome, but may be routed there

## Notes for reporters

The LAN endpoint is opt-in and default OFF, binds to the local network only,
and requires the app in the foreground (UWP PLM). It carries no authentication
by design — it is a single-operator research tool, not a multi-user service.
Reports demonstrating impact beyond "anyone on the LAN can call a documented
endpoint" are especially valuable.

## Response targets

- Acknowledgement: within ~7 days
- Triage verdict: within ~30 days
- Fix or mitigation: best effort alongside regular releases
