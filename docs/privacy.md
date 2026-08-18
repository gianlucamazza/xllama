# Privacy policy — xllama

**Last updated:** 2026-07-29  
**Applies to:** the xllama UWP application (Dev Mode sideload and any future
Microsoft Store listing of the Store SKU).  
**Operator:** Gianluca Mazza (see the GitHub repository
[gianlucamazza/xllama](https://github.com/gianlucamazza/xllama)).

This document is written for Partner Center / Store submission and for end
users. It is not legal advice.

## Summary

xllama is a **local** LLM chat and optional image-generation app for Xbox
Series S|X. Inference runs **on the device**. There is **no account system**,
**no analytics**, and **no crash-reporting service** built into the app today.

## What stays on the device

Unless you choose otherwise, the following remain in the app sandbox
(`LocalState` and related AppContainer storage):

- Downloaded model weights and configs
- Chat history and conversation settings
- Preference / feedback samples used for optional on-device personalization
- Optional on-device training outputs (merged GGUF under the personalize flow)
- App logs (`xllama.log`) written for debugging

The maintainer does not receive these files automatically.

## What leaves the device

Outbound network use is limited to:

1. **Model catalogue downloads** you start (or that first launch starts for the
   default chat model). Sources are listed in the in-app catalogue
   (`uwp/models/manifest.json`): typically GitHub Releases (`models-v1`) and/or
   Hugging Face model repositories.
2. **Nothing else by default.** The optional LAN HTTP API (Dev Mode / research
   SKU only) speaks only on your local network when you turn it on; it is
   **not** included in the Store SKU and is unauthenticated — use only on a
   trusted LAN.

The app does **not** upload chats, prompts, images, or feedback to the
maintainer or to a cloud inference backend.

## Third parties

When models download from GitHub or Hugging Face, those services process the
download request under **their** privacy policies (IP address, user agent, etc.).
xllama does not control that processing.

Runtime components (e.g. ONNX Runtime, DirectML, llama.cpp) run locally and do
not phone home through xllama.

## Children

xllama can generate open-ended text and images. It is **not** directed at
children. A future Store listing will carry an age rating reflecting generative
content; parents should not treat the app as a supervised kids product.

## Your choices

- Do not download models you do not want stored on the console.
- Clear app data / uninstall to remove LocalState (including history and models).
- Do not enable the LAN API (Dev Mode) on untrusted networks.
- Prefer the Store SKU (when published) if you want research surfaces (LAN API,
  USB paths, headless flags) omitted.

## Changes

Material changes to this policy will be reflected in this file and in the app
version notes (`CHANGELOG.md`). The “Last updated” date will change.

## Contact

- GitHub Issues: [gianlucamazza/xllama](https://github.com/gianlucamazza/xllama/issues)
- Repository owner profile on GitHub for contact details

For a Store listing, a stable HTTPS URL to this document (or a GitHub Pages
mirror) should be used as the privacy policy link in Partner Center.
