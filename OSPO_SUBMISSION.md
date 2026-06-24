# OSPO submission package — open-source release request

> Bring this to NVIDIA's internal OSS-release intake ("Open Source @ NVIDIA" portal / OSPO).
> The exact workflow is internal — route via your manager or Legal/IP. This file is the
> *content* a review needs, pre-filled from the repo. Items marked **[legal]** / **[verify]**
> need a human sign-off.

## 1. Project
- **Name:** VoiceGuard
- **Summary:** research code + paper studying cross-modal fusion (an acoustic deepfake
  detector + an LLM scam-intent signal) to detect AI voice-clone phone scams.
- **Reason for release:** defensive publication accompanying a paper — establish prior art,
  help defenders. Not a product; no NVIDIA-proprietary technology.
- **Current location:** `github.com/lamduong0/voiceguard` (PRIVATE).
- **Proposed:** an NVIDIA GitHub org, PUBLIC.
- **Maintainer:** dulam@nvidia.com.

## 2. License
- **Proposed:** MIT (`LICENSE` present, © NVIDIA Corporation).
- **[legal]** A narrow patent disclosure was filed on the core mechanism; MIT-in-public
  grants users that mechanism. Confirm this matches the IP strategy (now
  defensive-publication-primary — see `DISCLOSURE.md` §9, `ERRATA.md`).

## 3. IP / patent
- Filed narrow disclosure (multiplicative acoustic-veto fusion). No NVIDIA-internal source,
  trade secrets, or proprietary tech in the repo — research code over public models/datasets.

## 4. Third-party dependencies (all permissive; no copyleft)
- Core: numpy (BSD), pytest (MIT), openai (Apache-2.0), matplotlib (BSD-style).
- Acoustic/optional (`requirements-acoustic.txt`): torch, torchaudio (BSD-3), transformers,
  datasets, huggingface_hub (Apache-2.0), torchcodec (BSD-3), librosa (ISC), soundfile
  (BSD-3), scikit-learn (BSD-3).
- No GPL/LGPL/AGPL. **[verify]** licenses at the pinned versions during review.

## 5. Datasets — NOT redistributed
- ASVspoof 2019 LA (Edinburgh DataShare; custom license) and In-the-Wild (Apache-2.0),
  accessed via public Hugging Face mirrors.
- The repo commits only **derived per-clip probabilities** (`data/*.json`), never audio. Raw
  audio is gitignored (`*.wav`, `*.flac`). **[legal]** confirm derived-feature release is
  permitted under each dataset's terms.

## 6. Models — NOT redistributed
- Pretrained HF detectors (MelodyMachine, Gustking wav2vec2-xlsr, mo-thecreator, Hemgg) are
  called at inference; weights are not included. **[verify]** each model card's license
  permits this use/citation.

## 7. Secrets / credentials / internal infra
- **Secret scan clean:** no API keys/tokens in any tracked file. The LLM key is read from env
  `VG_LLM_KEY` only.
- **Internal endpoint removed:** the gateway base URL is now env-driven (`VG_LLM_BASE_URL`,
  defaults to OpenAI); no NVIDIA-internal hostnames remain in the code (verified).

## 8. Security / privacy
- No PII. Transcripts are synthetic (LLM-generated, cached in `data/dialogues.json`). No user
  data, no telemetry. Network calls only to user-configured LLM gateway and HF downloads.

## 9. Export control
- Standard ML research code (PyTorch/Hugging Face); no cryptography or controlled technology.
  **[verify]** against NVIDIA export policy.

## 10. Included / excluded
- **Included:** code, derived JSON, figures, docs (paper, disclosure draft, results, errata).
- **Excluded (gitignored):** dataset audio, venv, LaTeX build artifacts, local notes (TODO.md).

## Pre-submission checklist
- [x] Secret scan clean
- [x] Internal endpoint parameterized (no NVIDIA hostnames in code)
- [x] Dataset audio not redistributed (gitignored)
- [x] LICENSE present
- [x] Public-facing numbers corrected (`ERRATA.md`)
- [ ] **[legal]** MIT vs filed-patent sign-off
- [ ] **[legal]** dataset derived-feature redistribution OK
- [ ] **[verify]** model-card licenses
- [ ] Move to NVIDIA org + flip public (after approval)

## How to submit
1. Find NVIDIA's OSS-release intake (internal portal / OSPO; ask manager or Legal-IP).
2. Attach/paste this package + the repo link.
3. Route for license + IP/patent + security/export review.
4. On approval: transfer the repo to the approved NVIDIA org and set it public (I can do the
   `gh` transfer + visibility flip in seconds once you confirm the org).
