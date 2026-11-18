# Read before sideloading

xllama runs only on an Xbox console in **Developer Mode**. These notes are the
full list that used to live on the README. The README keeps a short checklist
and links here.

- **Developer Mode only.** xllama runs only on an Xbox console in Developer
  Mode, Microsoft's program for developing and testing apps. To activate it,
  you need your own Partner Center developer account, and you must accept
  Microsoft's
  [Xbox Developer Mode Activation terms](https://learn.microsoft.com/en-us/legal/windows/agreements/xbox-one-developer-mode-activation)
  yourself. xllama is a developer and research project, not a consumer
  product.
- **Retail games and apps don't run in Developer Mode.** You need to switch
  back to Retail Mode to play. Microsoft notes that required updates may cause
  "diminished or lost functionality" on a Developer Mode console, and a
  console reset deletes local saves that aren't synced to the cloud.
- **Unofficial.** Independent project, not affiliated with or endorsed by
  Microsoft. xllama is not reviewed, certified or supported by Microsoft and
  is not distributed through the Microsoft Store. Sideload it only on a
  console you own, at your own risk. This project makes no statement about
  warranty coverage or about how Microsoft may treat Developer Mode use; read
  Microsoft's terms and warranty for your country.
- **Sideload only from this repository's Releases page** and check the
  SHA-256. The package is signed with a test certificate (`xllama-test.cer`);
  trust it only if you downloaded it from here.
- **The LAN API is off by default and has no authentication.** When it is on,
  any device on your network, and any web page open in a browser on it, can
  send it requests. Enable it only on a network you trust.
- **Models are third-party.** Models are downloaded separately (from Hugging
  Face or this repository's `models-v1` release) and each one is under its
  own licence, not under xllama's MIT licence. Some licences are
  non-commercial or carry attribution requirements. Check each model's
  licence on its upstream model card (LFM models also download their LICENSE
  file alongside the weights) before using it.
- xllama is provided "AS IS", without warranty, under the MIT licence.

Sideload steps: [install-release.md](install-release.md). Product intent and
honest limits: [positioning.md](positioning.md).
