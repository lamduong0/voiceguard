# Paper outline

Working title: **"When does fusion help? Multiplicative cross-modal fusion for
voice-clone scam detection, and the limits of off-the-shelf detectors."**

Framing: an honest, empirical paper. The headline is not a new detector but *when and
why* combining an acoustic deepfake detector with conversational intent helps — and the
regime boundaries where it doesn't. Numbers live in `RESULTS.md`.

Venue options: a speech/security workshop (Interspeech/ICASSP satellite, or a fraud/anti-
spoofing workshop) for the focused result; arXiv first as the **defensive publication**
that protects the multiplicative-veto idea regardless of acceptance.

---

## Abstract (draft beats)
- AI voice-clone scams are rising; defenses pair a deepfake-audio detector with other
  signals, but it's unclear when that fusion actually helps.
- We study fusion of an acoustic synthetic-voice detector with an LLM intent signal,
  isolating a **genuine_scam** hard negative (real voice, scam-sounding call) that intent
  alone cannot resolve.
- Finding 1: fusion beats either modality alone. Finding 2: a multiplicative/Bayesian
  veto beats linear fusion **specifically when the detector is informative but imperfect**
  (8/8 seeds, two model×dataset settings); it ties when the detector is near-perfect or
  broken. Finding 3: generalization is the binding constraint and is model-dependent.

## 1. Introduction
- The threat: cheap voice cloning, grandparent/CEO scams, out-of-band-verification advice
  doesn't scale.
- The common architecture (acoustic detector + intent + metadata → risk → alert) and the
  open question: *does fusing intent actually add value, and which fusion?*
- Why the question is non-trivial: intent and clone-status are confounded in naive
  benchmarks (high intent ≈ a clone), making intent look sufficient. The real test needs a
  high-intent genuine negative.
- Contributions (3 claims, mapped to §6). State plainly that the broad system is prior art
  (cite the patents) and our contribution is the mechanism + the evaluation methodology.

## 2. Related work
- Audio deepfake / anti-spoofing detection (ASVspoof line, AASIST, wav2vec2-SSL); the
  generalization gap ("Does Audio Deepfake Detection Generalize?", In-the-Wild).
- Scam/fraud-call detection and intent classification.
- Multimodal/late fusion; Bayesian sequential analysis (SPRT) in detection.
- Prior art / deployed systems: US12284313B1, US10455085B1, US9692885B2, Pindrop, Aurigin
  — position our narrow mechanistic claim against these.

## 3. Method
- Problem setup: per-call decision, positive = clone + scam intent.
- The two signals and four scorers, formally:
  - acoustic_only, intent_only (baselines)
  - parallel (linear late fusion): `a·â + (1−a)·î`
  - **bayesian (multiplicative veto)**: `σ(g·(â − c))·î`, with calibrated center `c`.
- Why the multiplicative form: acoustic evidence can *veto* a high-intent genuine call;
  linear mixing cannot (intent pushes the sum up regardless). This is the mechanistic
  prediction tested in §6.
- Note on Stage 1 (simulation) in an appendix: the temporal evidence-accumulation variant;
  why it does not transfer to per-call real data (acoustic ~static across a clip) — keep
  this honest and brief, as motivation not as a result.

## 4. Benchmark construction (a contribution in itself)
- Datasets: ASVspoof 2019 LA, In-the-Wild; acoustic models m1/m2.
- Intent: LLM-generated + LLM-scored transcripts; archetypes table.
- The **genuine_scam** hard negative and why it's the crux; document the earlier
  intent⟺clone confound and the fix (honesty = credibility).
- Threats to validity (lift from RESULTS.md): constructed pairing, semi-synthetic intent,
  scale.

## 5. Experimental setup
- Matched-FAR (TPR @ 10% FAR) evaluation; AUC; out-of-fold fusion-param fitting; Bayesian
  center calibration; bootstrap CIs; 8 pairing seeds. Emphasize *fair* comparison (equal
  operating point, no leakage) — this is what makes claim 2 trustworthy.

## 6. Results
- **§6.1 Claim 1** — fusion > single modality (Table: 4 scorers × {AUC, TPR@FAR_gs,
  TPR@FAR_all}). The intent_only→0.000 and acoustic_only→0.425 contrast.
- **§6.2 Claim 2** — multiplicative > linear, with the 2×2 regime table (+0.251, +0.136,
  tie, n/a) and the non-overlapping bootstrap CIs. Lead with the boundary characterization.
- **§6.3 Claim 3** — generalization table (mean P-synthetic by class), m2 transfers / m1
  doesn't; fusion can't recover a broken channel.
- Figures: (F1) ROC per method on ASVspoof m1; (F2) bayesian−parallel TPR vs acoustic
  separability across the 4 settings — the "fusion helps when the detector is uncertain"
  curve; (F3) the generalization bar chart.

## 7. Limitations & ethics
- Limitations from §4 threats; single language; no real deployment / latency study.
- Ethics: dual-use; we evaluate *defense*; no new cloning capability; datasets are public
  and consented; we publish to establish prior art and help defenders.

## 8. Conclusion
- Fusion helps, but conditionally; the multiplicative veto is the right form in the regime
  that matters; the real bottleneck is detector generalization — invest there.

---

## Open to-dos before submission
- [ ] F2 curve: sweep an explicit acoustic-degradation axis (needs a short GPU run) to draw
      "fusion gain vs detector quality" cleanly, rather than 4 discrete points.
- [ ] Tighten claim-2 CIs with more clips / a 3rd detector if a reviewer pushes on N.
- [ ] Real paired scam audio (even a small set) would materially strengthen §4.
