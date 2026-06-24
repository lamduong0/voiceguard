# Real paired-audio sourcing plan (§4 validity upgrade)

The biggest validity gap: our benchmark pairs *independent* audio and transcripts, so the
acoustic--content correlation of real calls is absent. The fix is audio where a cloned voice
and scam intent co-occur in the same call (and the hard negative: a genuine voice with scam
intent). No clean public dataset provides exactly this, so below is a concrete acquisition
plan, ranked by feasibility.

## Candidate sources (from a literature/web scan)
1. **PITCH** — "AI-assisted Tagging of Deepfake Audio Calls" (arXiv 2402.18085). ~18,600
   bona-fide + ~1.6M deepfake call samples from 100 users, built for deepfake *call*
   detection. Closest to our setting. Action: check the paper's release/repo for terms and
   availability; if obtainable, it gives real clone-vs-genuine *call* audio.
2. **CallHome (LDC)** — 120 unscripted ~30-min English telephone conversations. Real
   conversational telephone audio (the genuine side / hard negatives). License: LDC (paid /
   institutional). Pair genuine CallHome turns with TTS/clone renditions to build matched
   clone vs genuine pairs of the *same* content.
3. **PartialSpoof / Fake-or-Real (FoR) / Fake Speech in the Wild** — anti-spoofing corpora
   with partial and full spoofs; useful to diversify the acoustic side beyond ASVspoof and
   In-the-Wild and to add a 3rd/4th detector-stress condition.
4. **Scam-baiting / fraud-call recordings (YouTube, FTC complaint audio)** — real scam-call
   audio exists publicly but is unlabeled, consent-ambiguous, and rarely cloned-voice. Use
   only for qualitative validation, not as a labeled benchmark.

## Recommended plan
- **Best case:** obtain PITCH (real deepfake call audio) → re-run the exact
  `experiments/stage2_*` pipeline on it; this directly answers the independence-assumption
  critique. Gate: license/availability.
- **Pragmatic build:** take genuine telephone speech (CallHome or a consented internal set),
  synthesize matched clones of the *same utterances* with a current TTS/voice-clone model,
  and pair both with scam vs benign transcripts. This yields true content-matched
  clone/genuine pairs — the cleanest test of the veto — under controlled, consented data.
- **Diversify detectors** with PartialSpoof/FoR to extend the claim-2 robustness sweep.

## Constraints / notes
- Licensing and consent are the gating issues, not engineering. CallHome and most named
  corpora require LDC or dataset-specific agreements; route through the appropriate data/legal
  process before download or redistribution.
- Do **not** commit any sourced audio to this repo (already gitignored: `*.wav`, `*.flac`).
- Once data is in hand, the existing pipeline (`stage2_roc.py`, `stage2_harden.py`) runs
  on it unchanged — only the acoustic-events JSON source changes.
