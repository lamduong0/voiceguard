# When does fusion help? Multiplicative cross-modal fusion for voice-clone scam detection

*Draft. Numbers trace to `experiments/` and `RESULTS.md`. §6.4 (degradation sweep) is
pending an in-flight run; its placeholder is marked.*

## Abstract

AI voice cloning has made impersonation scams cheap and convincing, and proposed defenses
increasingly pair an acoustic synthetic-voice detector with a second signal such as
conversational intent. It is not obvious that such fusion helps, nor which fusion rule to
use. We study the fusion of an acoustic deepfake detector with an LLM-derived scam-intent
score for the per-call decision *is this a voice-clone scam?* The key to a meaningful
evaluation is a hard negative we call **genuine_scam** — a real human voice making a
scam-sounding call — which intent alone cannot resolve and on which only the acoustic
channel carries information. On a constructed benchmark over two public audio datasets and
four pretrained detectors, evaluated at a matched false-alarm rate with out-of-fold fusion
fitting and bootstrap confidence intervals, we find: (1) fusion beats either modality
alone; (2) a *multiplicative* fusion rule, in which the acoustic posterior can veto a
high-intent call, beats linear (weighted-sum) fusion in two of the detector×dataset
settings tested (8/8 and 6/8 random seeds), but the benefit is modest (≈0.12–0.13 TPR)
and detector-specific — two other detectors show no gain, and it does not win when the
detector is near-perfect or broken. A logistic-regression fusion with an acoustic×intent
interaction term and a min (AND) rule match it, so the gain belongs to conjunctive fusion,
not to the multiplicative form; and (3) detector generalization is
the binding constraint and is strongly model-dependent. We release the benchmark
construction and analysis to support reproducible, honest evaluation of such defenses.

## 1. Introduction

Consumer-facing voice-clone fraud — the "grandparent" scam, CEO-impersonation, fake-kidnap
calls — has grown with the availability of few-second voice cloning. The standard defensive
recipe combines an acoustic synthetic-voice detector with auxiliary signals (conversational
content, telephony metadata) into a risk score and an in-call alert. Whether adding a
conversational-intent signal actually improves detection, and how best to combine it, is
under-examined.

The question is subtle because of a confound. In naive benchmarks, high scam-intent is
almost perfectly correlated with the call being a clone (the only scam-y transcripts belong
to clone audio), so an intent classifier alone appears to solve the task and fusion looks
pointless. The acoustic channel only earns its keep when intent is *not* sufficient — that
is, on calls with high scam-intent but a genuine voice: a panicked relative whose words
match a scam script, or a human scammer using their own voice. We make this hard negative,
`genuine_scam`, central to the evaluation.

Our contribution is deliberately narrow. The broad system — live, on-device fusion of
acoustic, intent, and metadata signals into a during-call risk and alert — is prior art
(§2). What we contribute is (i) a fusion *mechanism*, a multiplicative acoustic veto, and an
analysis of *when* it helps; and (ii) an evaluation methodology built around the
`genuine_scam` hard negative. Concretely:

- **Claim 1.** Cross-modal fusion beats either modality alone (§6.1).
- **Claim 2.** Multiplicative fusion beats linear fusion when the acoustic detector is
  informative but imperfect — the realistic regime (§6.2, §6.4).
- **Claim 3.** Detector generalization is the binding constraint and is model-dependent;
  fusion cannot recover a broken acoustic channel (§6.3).

## 2. Related work

**Audio deepfake / anti-spoofing detection.** The ASVspoof challenge series and models such
as AASIST and wav2vec2-based SSL detectors define the field. A recurring result is poor
*generalization*: detectors trained on text-to-speech attacks degrade sharply on
real-world deepfakes (the In-the-Wild dataset). Our claim 3 reproduces and quantifies this.

**Scam/fraud-call detection and intent.** Prior work classifies fraudulent calls from
transcript intent (payment requests, urgency, threats). We use such a signal but treat it as
one input to fusion, not the decision.

**Fusion and sequential analysis.** Late/early/hybrid multimodal fusion is well studied;
sequential probability ratio testing appears in detection, including a deepfake patent
(US12347238B2). We isolate the *form* of the fusion rule (multiplicative vs. linear) as the
object of study.

**Prior art / deployed systems.** US12284313B1 (live cloned-voice detection + transcript
intent + risk + alert, parallel fusion), US10455085B1 and US9692885B2 (during-call risk and
alerts), and commercial systems (Pindrop, Aurigin) establish the broad architecture. We
position our multiplicative-veto mechanism against the parallel fusion these teach.

## 3. Method

We make a per-call decision; the positive class is `clone_scam` (cloned voice **and** scam
intent). Two signals feed the decision:

- **Acoustic** â ∈ [0,1]: a pretrained anti-spoofing model's probability that the call audio
  is synthetic (mean over 1 s windows for the per-call score).
- **Intent** î ∈ [0,1]: an LLM's scam-intent score over the call transcript (max over turns).

We compare four per-call scorers:

- `acoustic_only` = â,  `intent_only` = î  (baselines),
- **parallel** (linear late fusion): a·â + (1−a)·î,
- **bayesian** (multiplicative veto): σ( g·(â − c) ) · î,

where c is a decision center calibrated to the operating population (the midpoint of
genuine vs. clone acoustic scores on held-out data) and g is a gain. Fusion parameters
(a, g) and the center c are fit out-of-fold.

**Why multiplicative.** Because the score is a product, strong genuine-voice evidence
(small â, hence a small first factor) drives risk toward zero *regardless of intent*,
vetoing the high-intent genuine call. A weighted sum cannot: a high î always lifts the sum,
so to hold the false-alarm rate on `genuine_scam` the linear rule must raise its threshold
and loses true positives. The calibrated center makes the veto robust to detectors whose
scores are not centered at 0.5. This mechanistic prediction is exactly what §6.2 tests.

**Note on a non-result.** An earlier simulation favored a controller that accumulated
acoustic evidence *over time*. On real data the per-clip acoustic score is approximately
constant across a call, so that temporal advantage does not transfer; we therefore score
per-call and do not claim a streaming/early-detection benefit. (Details in Appendix A.)

## 4. Benchmark construction

No public dataset pairs cloned-voice audio with scam-intent dialogue in the same call, so we
construct one by crossing audio (genuine vs. clone) with transcript intent (benign, urgent,
scam) into call archetypes: `clone_scam` (positive), `clone_benign`, `genuine_benign`,
`genuine_urgent`, and the decisive **`genuine_scam`** — genuine audio paired with a scam
transcript. `genuine_scam` is the hard negative on which intent is uninformative and only
acoustic can separate clone from genuine.

- **Audio:** ASVspoof 2019 LA and In-the-Wild, 300 clips per class per source.
- **Intent:** transcripts generated and per-turn scored by an LLM (scam, benign, and
  genuine-urgent scenarios), aligned to the call timeline.

We note a methodological pitfall we hit and fixed: an initial pairing tied scam-intent only
to clone audio, making intent trivially sufficient (AUC 0.99, fusion irrelevant). Adding
`genuine_scam` removes the confound and is what makes the comparison meaningful. We report
this for transparency; it is the central design choice of the benchmark.

## 5. Experimental setup

Detectors: m1 = `MelodyMachine/Deepfake-audio-detection-V2`, m2 =
`Gustking/wav2vec2-large-xlsr-deepfake-audio-classification`. Evaluation: per-call scores;
fusion parameters and center fit **out-of-fold** (2-fold, no leakage); comparison at a
**matched false-alarm rate** (TPR @ 10% FAR) against `genuine_scam` and against all
negatives; AUC; bootstrap 95% CIs; and reporting over **8 random pairing seeds** to separate
pairing randomness from estimator noise.

## 6. Results

### 6.1 Fusion beats single modality

ASVspoof, detector m1, 600 clips/class/source, 8 seeds (mean ± std):

| method | AUC | TPR@10%FAR vs genuine_scam | TPR@10%FAR vs all-neg |
|---|---|---|---|
| acoustic_only | 0.770 ± 0.006 | 0.679 ± 0.031 | 0.406 ± 0.026 |
| intent_only | 0.894 ± 0.009 | 0.098 ± 0.003 | 0.502 ± 0.047 |
| parallel | 0.933 ± 0.010 | 0.484 ± 0.067 | 0.708 ± 0.064 |
| bayesian | 0.950 ± 0.005 | 0.606 ± 0.054 | 0.803 ± 0.032 |

Intent alone is at chance against `genuine_scam` (0.098 at a 10% FAR); acoustic alone is weak against the
broad negative set (0.406). Only fusion is strong on both axes. Figure F1
(`figures/F1_roc.png`) shows the full ROC: the multiplicative rule dominates.

### 6.2 Multiplicative beats linear fusion (claim 2)

bayesian − parallel, TPR@10%FAR vs `genuine_scam`, 8 seeds (600 clips for m1/m2; 300 for m3/m4):

| setting | acoustic regime | bayesian − parallel | seeds won |
|---|---|---|---|
| m1 × ASVspoof | informative, imperfect | +0.123 | 8/8 |
| m2 × In-the-Wild | informative, imperfect | +0.127 | 6/8 |
| m2 × ASVspoof | near-perfect | −0.030 | no win (0/8) |
| m4 × ASVspoof | near-perfect (clean) | −0.047 | no win (1/8) |
| m4 × In-the-Wild | informative (clean) | −0.024 | no win (3/8) |
| m1 × In-the-Wild | broken | −0.067 | both ≈ chance (0/8) |

Where it appears (m1×ASVspoof, m2×In-the-Wild) the multiplicative veto wins in 8/8 and 6/8
seeds. The bootstrap CIs separate for m1×ASVspoof (600 clips: bayesian [0.600, 0.705] vs
parallel [0.422, 0.533]); for m2×In-the-Wild the single-seed CIs overlap (bayesian
[0.491, 0.762] vs parallel [0.411, 0.534]), so that setting rests on the seed count. But the scaled-up run (4 detectors, 600 clips)
shows the effect is **modest and not universal** — the honest headline: (i) the m1×ASVspoof
gain shrank from +0.251 (300 clips) to +0.123 (600), so the small-sample estimate was
inflated; (ii) only 2 of 4 detectors show a clean win — a 3rd (mo-thecreator) collapses both
classes to P(synthetic)≈0 under noise (negative gain at every SNR), and a 4th (Hemgg) that
*does* degrade gracefully still shows no win in its informative band. The veto helps in
specific detector×regime combinations with a clear mechanism, but is not a general property
of multiplicative fusion, and AUC level alone does not predict it.

**Stronger baselines and label noise.** Two more fusion rules, fit out-of-fold the same way,
test whether the *multiplicative form* matters: logistic regression on (â, î, â·î) and an
AND rule min(σ(g(â−c)), î). In the two winning settings the multiplicative veto beats the
logistic-interaction fusion by only +0.011 (m1×ASVspoof, 7/8 seeds) and +0.016
(m2×In-the-Wild, 7/8), and the AND rule beats the product by 0.015–0.018. Dropping the 3 of
15 "scam" dialogues that the generating LLM itself rated benign (peak intent 0.1–0.5) moves
the gain over linear fusion to +0.093 and +0.104, adds a win for m4×ASVspoof (+0.048, 8/8),
and leaves every conjunctive rule within ±0.04 of the others. The gain over a weighted sum
is therefore the gain of having an interaction term at all (RESULTS.md, "Claim 2
robustness").

### 6.3 Generalization is the binding constraint (claim 3)

Mean P(synthetic) by class:

| model | ASVspoof genuine → clone | In-the-Wild genuine → clone |
|---|---|---|
| m1 | 0.026 → 0.401 | 0.339 → 0.211 (inverted; AUC 0.519) |
| m2 | 0.353 → 0.896 | 0.212 → 0.634 |

The weaker detector collapses to chance on real-world deepfakes; the stronger SSL detector
transfers. When acoustic is broken, every method is at or below chance (≲0.10) against `genuine_scam`.

### 6.4 Fusion gain vs. detector quality (degradation sweep)

We add white noise at decreasing SNR (clean → 0 dB), sweeping detector m2 from informative
to chance, and measure the multiplicative−linear gain (bayesian − parallel, TPR@10%FAR vs
`genuine_scam`) against detector quality (acoustic_only AUC). 5 seeds, `stage2_degrade.py`.

| source | SNR | acoustic AUC | bayesian TPR_gs | linear TPR_gs | gain (b−l) |
|---|---|---|---|---|---|
| ASVspoof | clean | 0.875 | 0.909 | 0.935 | −0.026 |
| ASVspoof | 20 dB | 0.860 | 0.816 | 0.707 | +0.109 |
| ASVspoof | 15 dB | 0.796 | 0.551 | 0.392 | **+0.159** |
| ASVspoof | 10 dB | 0.680 | 0.212 | 0.242 | −0.030 |
| ASVspoof | 5 dB | 0.539 | 0.088 | 0.097 | −0.009 |
| ASVspoof | 0 dB | 0.474 | 0.068 | 0.100 | −0.033 |
| In-the-Wild | clean | 0.805 | 0.680 | 0.586 | +0.094 |
| In-the-Wild | 20 dB | 0.752 | 0.501 | 0.481 | +0.020 |
| In-the-Wild | 15 dB | 0.728 | 0.455 | 0.396 | +0.059 |
| In-the-Wild | 10 dB | 0.674 | 0.341 | 0.339 | +0.001 |
| In-the-Wild | 5 dB | 0.623 | 0.257 | 0.231 | +0.027 |
| In-the-Wild | 0 dB | 0.583 | 0.164 | 0.173 | −0.009 |

The gain is largest in the **informative-but-imperfect band** (acoustic AUC ≈ 0.73–0.86):
it peaks at +0.159 on ASVspoof (15 dB) and is non-negative across In-the-Wild's informative
range (≈0 at 10 dB). It turns slightly negative when the detector is near-perfect (ASVspoof
clean, AUC 0.875, −0.026), and there is a small negative dip at 10 dB (AUC 0.68, −0.030).
Below that the detector is at chance (AUC ≤ 0.54): both rules sit near chance (TPR ≈ 0.1)
and the gain is ≈0 or slightly negative. The overall shape —
rising from the high-quality end and peaking where the detector is uncertain but not useless
— matches the §6.2 prediction. Figure F2 (`figures/F2_gain_vs_quality.png`) plots gain
against acoustic AUC.

## 7. Limitations & ethics

The benchmark pairs independent audio and transcripts, removing natural acoustic–content
correlation; real paired scam recordings would strengthen the evidence. The intent signal is
LLM-generated and LLM-scored, so it is somewhat idealized relative to an ASR→classifier
pipeline. We evaluate two detectors, two datasets, and one language, at modest scale (hence
multi-seed reporting). We study *defense* only, introduce no new cloning capability, use
public consented datasets, and release the methodology to help defenders and to establish
prior art.

## 8. Conclusion

Fusing conversational intent with an acoustic deepfake detector helps for voice-clone scam
detection — but conditionally and modestly. A multiplicative acoustic veto can beat linear
fusion when the detector is informative but imperfect, with a clear mechanism (acoustic
evidence vetoes a high-intent genuine call); across four detectors it delivered a clean,
gain in two settings (≈0.12–0.13 TPR; 8/8 and 6/8 seeds) and nothing in the other two, so it is a
useful tool in the toolbox, not a universal rule — and any conjunctive rule (an AND, or a
logistic fusion with an interaction term) does as well as the multiplicative one. The dominant lever is acoustic-detector
generalization: when it fails, no fusion recovers it. Defenders should invest first in
detectors that transfer to real-world deepfakes, and treat multiplicative fusion as a
mechanism to evaluate per detector rather than assume.

## Appendix A — the simulation non-result

A pre-registration-style simulation (Appendix, `experiments/stage1_*.py`) favored a
controller accumulating acoustic evidence over the call. We include it to document that the
temporal advantage is a simulation artifact: real per-clip acoustic scores are ~constant
across a call, so per-call scoring is appropriate and we make no streaming claim.
