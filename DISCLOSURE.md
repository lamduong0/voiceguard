# Invention disclosure (draft)

> Internal draft for the NVIDIA invention-disclosure process. Scope is deliberately
> narrow — the broad system is prior art (see §3). Fill in inventor names before filing.

**Title:** Multiplicative cross-modal fusion with an acoustic veto for distinguishing
voice-clone scams from genuine high-urgency calls.

**Inventor(s):** (dulam@nvidia.com)

**Date:** 2026-06-23.

---

## 1. Problem
Defenses against AI voice-clone phone scams combine an acoustic synthetic-voice detector
with a conversational scam-intent signal. The hard case is the **high-intent genuine
call** — a real person (panicked relative, or a human scammer using their own voice)
whose words pattern-match a scam. Intent alone flags it (false alarm); a system that
over-weights intent will repeatedly cry wolf on real family emergencies, which is the
single fastest way to get a consumer defense disabled.

## 2. What is needed
A fusion rule that uses the conversational intent signal to *raise* sensitivity to scams
**without** flagging high-intent genuine calls — i.e., that lets acoustic evidence of a
genuine voice *override* a high intent score, while still catching clones whose acoustic
evidence is only moderate.

## 3. Closest prior art (acknowledged)
- **US12284313B1** — real-time call segmentation, AI cloned-voice detection, continuous
  risk score, transcript intent, alert. Uses **parallel/late fusion**.
- **US10455085B1**, **US9692885B2** — transcript-intent / voice-sample risk scoring during
  a call with alerts.
- **US12347238B2** — SPRT / cumulative-log-likelihood deepfake detection (audio+video).
- Commercial: Pindrop, Aurigin (real-time acoustic deepfake detection).

These teach combining acoustic + intent signals and producing a during-call risk/alert.
**None teaches the specific fusion rule below, nor its regime-dependent benefit.**

## 4. The invention (summary)
Combine the calibrated acoustic synthetic-voice posterior and the conversational
scam-intent score **multiplicatively**, so that the acoustic term acts as a **veto**:

    risk = sigmoid( g · ( â − c ) ) · î

- `â` = acoustic P(synthetic-voice) for the call (or running estimate),
- `c` = a **calibrated decision center** estimated from the operating population
  (midpoint of genuine vs. clone acoustic scores), making the rule robust to detectors
  whose scores are not centered at 0.5,
- `g` = a gain, `î` = conversational scam-intent (e.g., max over turns).

Because risk is a **product**, a low acoustic posterior (strong genuine-voice evidence)
drives risk toward zero **regardless of how high the intent is** — vetoing the high-intent
genuine call. A weighted-sum (parallel) rule cannot do this: high intent always lifts the
sum. The alert/intervention then triggers on `risk ≥ threshold` during the call.

## 5. Key novel points (candidate claims)
1. **Multiplicative acoustic-veto fusion** of an acoustic synthetic-voice posterior and a
   conversational scam-intent score for an in-call clone-scam decision (independent claim).
2. **Calibrated center `c`** estimated from operating-population acoustic scores, so the
   veto works with uncalibrated/domain-shifted detectors (dependent).
3. Triggering an **in-call intervention** (warning / verification challenge) when the
   multiplicative risk crosses a threshold (dependent).
4. **Gating by detector confidence regime**: applying the multiplicative veto when the
   acoustic detector is informative-but-imperfect, falling back otherwise (dependent;
   supported by the regime-dependent result in §7).
5. Extending the product to additional veto factors (e.g., telephony/caller-ID
   route-mismatch posterior) as further multiplicative terms (dependent).

## 6. Advantages
- Rejects the high-intent genuine call (the dominant false-alarm source) that linear
  fusion flags.
- Strong on **both** axes simultaneously — clone discrimination *and* the broad negative
  set — where each single modality and linear fusion sacrifice one.
- Robust to uncalibrated detector outputs via the calibrated center.

## 7. Evidence of reduction to practice
Implemented and evaluated (`experiments/stage2_*.py`); see `RESULTS.md`. Per-call,
matched-FAR (10%), out-of-fold fusion fitting, 8 pairing seeds, two pretrained detectors,
two datasets. Against the **genuine_scam** hard negative, multiplicative beats linear:

| setting | acoustic regime | Δ TPR@10%FAR (mult − linear) | seeds won |
|---|---|---|---|
| MelodyMachine × ASVspoof | informative, imperfect | +0.251 | 8/8 |
| wav2vec2-xlsr × In-the-Wild | informative, imperfect | +0.136 | 8/8 |
| wav2vec2-xlsr × ASVspoof | near-perfect | −0.021 | tie |
| MelodyMachine × In-the-Wild | broken | +0.001 | n/a |

The regime-dependent gain (helps precisely when the detector is uncertain) is the
**non-obviousness** argument: a person of ordinary skill combining known acoustic + intent
fusion would not predict that a multiplicative veto specifically recovers the high-intent
genuine case while linear fusion fails it.

## 8. Alternatives / variations
- `î` as max, mean, or last-turn intent; `â` as whole-call or streaming estimate.
- `c` calibrated globally, per-caller, or per-device.
- Learned monotone gating g(·) instead of the sigmoid form.
- On-device vs. server execution; provider-agnostic intent model.

## 9. Filing recommendation
File as a **narrow** disclosure on claims 1–2 (the mechanism + calibrated center). Pair
with an **arXiv defensive publication** of `RESULTS.md`/the paper so the idea is protected
even if the patent is not pursued or is narrowed on obviousness. Do **not** assert the
broad live-fusion-alert system (prior art).
