# Upstream contributions

Running xllama on an Xbox Series S in Developer Mode surfaced problems in the
open-source ONNX Runtime projects. These are the pull requests Gianluca Mazza
opened against them, with their status as shown on GitHub:

| Project           | Pull request                                                      | What it does                                                                                      | Status              |
| ----------------- | ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- | ------------------- |
| onnxruntime-genai | [#2280](https://github.com/microsoft/onnxruntime-genai/pull/2280) | Falls back to the system D3D12 runtime when the Agility SDK device factory cannot create a device | Merged, 13 Jul 2026 |
| onnxruntime-genai | [#2300](https://github.com/microsoft/onnxruntime-genai/pull/2300) | Lets a model opt out of DirectML graph capture through a provider option                          | Merged, 20 Jul 2026 |
| onnxruntime-genai | [#2316](https://github.com/microsoft/onnxruntime-genai/pull/2316) | Adds tests and documentation for the #2300 opt-out                                                | Open                |
| onnxruntime       | [#29732](https://github.com/microsoft/onnxruntime/pull/29732)     | Reduces the Windows `ReadFileIntoBuffer` chunk size to 16 MB (AppContainer error 1450)            | Open                |

xllama ships builds of these libraries with its own patches until an official
package includes the fixes it needs; see [`patches/`](../patches/README.md)
and the licence notice in the [README](../README.md#license).

## Reported

Issues filed against the open-source ONNX Runtime project. Both were closed
for inactivity, without a fix:

| Project     | Issue                                                           | Summary                                                                                                                                                                                                                                                                                         | Status                             |
| ----------- | --------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------- |
| onnxruntime | [#29739](https://github.com/microsoft/onnxruntime/issues/29739) | Wrong decoder logits on the DirectML execution provider on Xbox Series S, correct on the CPU execution provider. The issue was opened about attention; a later comment on it traces the fault to the fused RMSNorm (`SimplifiedLayerNormalization` / `SkipSimplifiedLayerNormalization`) nodes. | Closed for inactivity, 18 Sep 2026 |
| onnxruntime | [#29730](https://github.com/microsoft/onnxruntime/issues/29730) | Large `ReadFileIntoBuffer` chunks can fail with `ERROR_NO_SYSTEM_RESOURCES` (1450) inside a Windows AppContainer. Pull request [#29732](https://github.com/microsoft/onnxruntime/pull/29732) proposes the change.                                                                               | Closed for inactivity, 18 Sep 2026 |
