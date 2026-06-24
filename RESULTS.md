# VoiceGuard — results memo

Cross-modal fusion for real-time AI voice-clone scam detection. This is the shared
spine for the paper and the invention disclosure: the claims, the numbers behind them,
and the limitations. All numbers are from the runs in `experiments/`; nothing here is
illustrative.

## TL;DR

Three defensible findings:

1. **Cross-modal fusion (acoustic + conversational intent) beats either signal alone.**
   Each modality fails a different hard case; fusion is the only thing strong on both.
2. **Multiplicative/Bayesian fusion beats linear (weighted-sum) fusion in the realistic
   regime where the acoustic detector is informative but imperfect** — 8/8 seeds across
   two independent model×dataset settings. It ties when the detector is near-perfect and
   is moot when the detector is broken.
3. **Acoustic-detector generalization is model-dependent and is the binding constraint.**
   A stronger SSL detector transfers to real-world deepfakes; a weaker one collapses to
   chance. No amount of intent fusion recovers a broken acoustic channel.

What this is *not*: the broad "live + on-device + fuse acoustic/intent/metadata + alert"
system claim is already patented (US12284313B1, US10455085B1, US9692885B2; Pindrop,
Aurigin). The contribution here is narrow, mechanism-specific, and empirical.

## Method

**Signals.** Acoustic = a pretrained anti-spoofing model's P(synthetic) per 1 s window.
Intent = an LLM's per-turn scam-intent score over a call transcript.

**Stage 1 (simulation, `experiments/stage1_*.py`).** Mechanism study on *simulated*
detector outputs, to compare fusion controllers cheaply before spending GPU. Finding:
a naive linear adaptive controller (intent lowers the acoustic threshold) is worse than
a tuned parallel baseline; a Bayesian controller (sequential evidence + intent as a
dynamic prior + multiplicative joint) beats it in the ambiguous regime. **Caveat that
shaped Stage 2:** Stage 1's edge relied on evidence *accumulating over time*; on real
data the per-clip acoustic score is ~constant across the call, so that temporal dynamic
is a simulation artifact. Stage 2 therefore scores each call as a whole.

**Stage 2 (real data, `experiments/stage2_*.py`).**
- Audio: In-the-Wild (real-world deepfakes) and ASVspoof 2019 LA, 300 clips/class/source.
- Acoustic models: `m1 = MelodyMachine/Deepfake-audio-detection-V2`,
  `m2 = Gustking/wav2vec2-large-xlsr-deepfake-audio-classification` (run on a Brev L4).
- Intent: transcripts generated and per-turn-scored by an LLM (45 dialogues, cached).
- **Constructed benchmark** crosses audio (genuine/clone) with transcript intent
  (benign/urgent/scam) into archetypes. The decisive hard negative is **`genuine_scam`**
  — a genuine voice making a scam-sounding call — where intent alone cannot help and only
  acoustic can. Positive = `clone_scam`.
- Evaluation: per-call scores; fusion params fit **out-of-fold** (2-fold, no leakage);
  Bayesian center **calibrated** to data; compared at a **matched false-alarm rate**
  (TPR @ 10% FAR), with bootstrap CIs (`stage2_roc.py`) and across **8 pairing seeds**
  (`stage2_harden.py`).

Four per-call scorers: `acoustic_only` (mean P-synthetic), `intent_only` (max intent),
`parallel` (a·acoustic + (1−a)·intent), `bayesian` (sigmoid(g·(acoustic−c))·intent).

## Claim 1 — fusion > single modality

ASVspoof, detector m1, 600 clips/class/source, 8 pairing seeds (mean ± std):

| method | AUC | TPR@10%FAR vs genuine_scam | TPR@10%FAR vs all-neg |
|---|---|---|---|
| acoustic_only | 0.770 ± 0.006 | 0.679 ± 0.031 | 0.406 ± 0.026 |
| intent_only | 0.894 ± 0.009 | **0.000 ± 0.000** | 0.000 ± 0.000 |
| parallel | 0.933 ± 0.010 | 0.484 ± 0.067 | 0.708 ± 0.064 |
| **bayesian** | **0.950 ± 0.005** | 0.606 ± 0.053 | **0.804 ± 0.035** |

`intent_only` → 0.000 against genuine_scam (cannot tell a clone-scam from a real-voice
scam); `acoustic_only` → 0.406 against the broad negative set. Fusion is the only thing
strong on both axes.

## Claim 2 — multiplicative > linear fusion (the patent-relevant mechanism)

bayesian − parallel, TPR@10%FAR vs genuine_scam, 8 seeds (600 clips/class/source for
m1/m2; 300 for m3/m4):

| setting | acoustic regime | bayesian − parallel | seeds bayesian wins |
|---|---|---|---|
| m1 × ASVspoof | informative, imperfect | **+0.122** | **8/8** |
| m2 × In-the-Wild | informative, imperfect | **+0.135** | **8/8** |
| m2 × ASVspoof | near-perfect | −0.010 | 1/8 (tie) |
| m4 × ASVspoof | near-perfect (clean) | −0.018 | 2/8 (tie) |
| m4 × In-the-Wild | informative (clean) | −0.031 | 3/8 (no win) |
| m1 × In-the-Wild | broken | +0.033 | 8/8 (both ≈ 0) |

Where it appears (m1×ASV, m2×ITW) the multiplicative veto wins across all 8 seeds and the
bootstrap CIs do not overlap (m1×ASV at 600 clips: bayesian [0.604, 0.707] vs parallel
[0.422, 0.533]) — acoustic evidence vetoes a high-intent genuine call where linear mixing
dilutes it.

**But the scaled-up run (4 detectors, 600 clips) makes the effect modest and clearly
*not universal* — this is the honest headline:**
- **Magnitude shrank with more data.** m1×ASV was +0.251 at 300 clips; at 600 it is +0.122.
  The small-sample estimate was inflated. m2×ITW is stable (+0.136 → +0.135).
- **Only 2 of 4 detectors show a clean win.** A 4th detector (m4, Hemgg) that separates both
  datasets and degrades *gracefully* still shows **no** win — near-perfect/tie on clean
  (both sources, −0.018 / −0.031) and ≈0 gain across its whole degradation sweep. A 3rd (m3,
  mo-thecreator) collapses under noise (both classes → P_fake ≈ 0, no graded signal) and
  never wins. AUC level alone does not predict where the veto helps.

Net: claim 2 holds where demonstrated (real, seed-robust, mechanistically motivated, CIs
separate) but is **detector/setting-specific and modest (~+0.12–0.14)**, not a general
property of multiplicative fusion. The honest contribution is "it can help, here is the
mechanism and where it shows up," not "it always helps."

## Claim 3 — generalization is the binding constraint (and model-dependent)

Mean P(synthetic) by class (sanity check that the detector separates the data):

| model | ASVspoof genuine → clone | In-the-Wild genuine → clone |
|---|---|---|
| m1 (MelodyMachine) | 0.026 → 0.401 ✓ | 0.339 → 0.211 ✗ (inverted; AUC 0.519) |
| m2 (wav2vec2-xlsr) | 0.353 → 0.896 ✓ | 0.212 → 0.634 ✓ |

The weaker detector fails to generalize to real-world (In-the-Wild) deepfakes; the
stronger SSL detector transfers. When acoustic is broken (m1 × ITW), every method scores
~0 against genuine_scam — fusion cannot manufacture a signal that isn't there.

## Threats to validity

- **Constructed benchmark.** Pairing independent audio with independent transcripts
  removes natural acoustic–content correlation. An over-coupled first version made intent
  trivially sufficient; adding the `genuine_scam` hard negative fixed that. Real paired
  scam recordings would be stronger evidence.
- **Semi-synthetic intent.** Transcripts are LLM-generated and LLM-scored, so the intent
  signal is somewhat idealized vs. a real ASR→classifier pipeline.
- **Scale.** Two acoustic models, two datasets, 300 clips/class/source; genuine_scam is
  ~25% of negatives, so per-setting CIs are non-trivial (hence 8-seed reporting).
- **Off-GPU intent / no streaming.** The streaming/time dimension was dropped because
  real per-clip acoustic is ~static; the temporal-accumulation result from Stage 1 does
  not transfer and is not claimed.

## Reproduce

```
# Stage 1 (CPU): stage1_mechanism.py, stage1_robustness.py
# Stage 2 (acoustic scored on GPU -> data/acoustic_events_m{1,2}.json):
PYTHONPATH=. .venv/bin/python experiments/stage2_roc.py           # iso-FAR + bootstrap CIs
PYTHONPATH=. .venv/bin/python experiments/stage2_harden.py data/acoustic_events_m1.json 8
PYTHONPATH=. .venv/bin/python experiments/stage2_harden.py data/acoustic_events_m2.json 8
```
