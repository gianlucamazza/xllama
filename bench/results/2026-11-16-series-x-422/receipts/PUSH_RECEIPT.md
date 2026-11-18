# Phi-4-mini push receipt — 2026-11-16

Task: push the verified Phi-4-mini GGUF onto the Xbox Series X for the #422 matrix.
Executed from the tower (ascadmin-1) over the house LAN via the Device Portal,
following the project's own provisioning layout (scripts/provision-models.sh:
catalogue models live at LocalState\models\<catalogue-name>\<file>).

## Source file

- Phi-4-mini-instruct-Q4_K_M.gguf, 2,491,874,272 bytes
- SHA-256: 88c00229914083cd112853aab84ed51b87bdf6b9ce42f532d8c85c7c63b1730a
  (matches the xllama catalogue manifest pin for phi4-mini)
- Downloaded on the tower directly from the catalogue source
  (huggingface.co/unsloth/Phi-4-mini-instruct-GGUF) into ~/phi_stage/ and
  hash-verified on the tower before upload. The tower keeps that staged copy.

## Console-side location

- Package: GianlucaMazza.xllama_1.7.0.1626_x64__pj67f1fcj4n14
- Folder (as the portal names it): \LocalState\models\phi4-mini
  (folder created via the portal folder API; it sits alongside
  lfm25-350m, lfm25-1.2b-instruct, personalized, sd-turbo-fp16)
- File: Phi-4-mini-instruct-Q4_K_M.gguf

## Verification

- Upload: portal POST returned HTTP 200 after the full 2,491,874,272-byte
  file was sent.
- Size: console-side portal listing reports FileSize 2491874272, exact match.
- Hash: the file was downloaded back through the portal to the tower and
  hashed there: 88c00229914083cd112853aab84ed51b87bdf6b9ce42f532d8c85c7c63b1730a,
  an exact match with the pin. The console-side copy is byte-for-byte the
  verified file. (The download-back temp copy was deleted after hashing.)
- API: GET http://192.168.4.54:11434/v1/models now lists phi4-mini alongside
  lfm25-1.2b-instruct, lfm25-350m, and personalized (active). The running app
  picked the model up on its own; NO restart, stop, designation change, or
  package change was made, and nothing was posted anywhere.

## Notes

- Unlike the app-downloaded LFM folders, the phi4-mini folder carries no
  LICENSE file and no .complete marker: the manifest's phi4-mini entry lists
  only the GGUF, and the project's own provision-models.sh uploads exactly
  the manifest files and nothing more. If the app later wants its marker,
  that is an app-side act on first load, not a missing upload step.
- phi4-mini is catalogue-marked CPU-only (kind gguf, routing disabled);
  its matrix cells are CPU rows unless the app says otherwise at run time.
