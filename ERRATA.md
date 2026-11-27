# Errata / changelog — reconciling the first filed/posted version with current

Three corrections landed after the initial invention disclosure / preprint. Anything already
public (arXiv, the filed disclosure) should be reconciled to the values below. The repo docs
(RESULTS.md, PAPER.md, paper/main.tex, DISCLOSURE.md, figures) are already updated.

## Correction 1 — AUC tie-handling bug (2026-06-24, from code review)
`auc()` used sequential sort-order ranks instead of tie-averaged ranks, understating AUC
where scores tie.
- **Effect:** `intent_only` AUC **0.828 → 0.889** (it has many tied scores).
- **Unchanged:** acoustic/parallel/bayesian AUCs, every TPR, and the claim-2 comparison
  (those scores are near-continuous). Fixed with averaged ranks + a regression test.

## Correction 2 — scale-up (600 clips/class/source + 4 detectors)
Doubling clips and adding two detectors revised the central claim-2 result **downward**:
- m1×ASVspoof gain **+0.251 (300 clips) → +0.122 (600)** — the small-sample estimate was
  inflated. m2×In-the-Wild stable (+0.136 → +0.135).
- Claim-1 table (m1×ASVspoof, now 600 clips): acoustic_only AUC 0.839→0.770, bayesian
  TPR@10%FAR-vs-genuine_scam 0.768→0.606, etc.
- Two more detectors tested: a 3rd (mo-thecreator) collapses under noise; a 4th (Hemgg)
  degrades gracefully but still shows **no** win. Net: a clean win in **2 of 4 detectors**.
- Claim 2 reframed: **real where it appears (8/8 seeds, CIs separate) but modest (~+0.12–0.14)
  and detector-specific, not universal.** F1 figure regenerated (intent AUC label 0.823→0.886).

## Correction 3 — TPR tie handling + acoustic-center calibration (2026-11-27, from code review)
Two evaluation bugs in `experiments/stage2_roc.py`, each fixed with a regression test:
- **`tpr_at_far()` dropped tied scores.** It counted positives strictly above the
  90th-percentile negative score. 77% of scam calls clip to intent_max = 1.0, so more than
  10% of negatives tied at the threshold and every tied positive was dropped. Ties at the
  threshold are now split (randomized tie-break, in expectation), so the achieved FAR equals
  the target.
  - `intent_only` TPR@10%FAR: vs genuine_scam **0.000 → 0.098** (chance); vs all negatives
    **0.000 → 0.502**. Claim 1 holds (intent alone is at chance on genuine_scam), but
    intent alone is not useless on the broad negative set.
  - Linear fusion was hit the same way wherever its fitted weight collapsed onto intent:
    m1×In-the-Wild gain +0.033 → −0.067 (both rules at chance). The low-SNR
    degradation-sweep "gains" (m2×ASVspoof 5/0 dB: +0.088/+0.068) were artifacts and are now
    ≈0 or slightly negative.
- **Acoustic center calibrated on the wrong labels.** `oof_scores()` took the midpoint of
  clone_scam vs *all* negatives, which include clone_benign (clone voices). The paper and
  disclosure define `c` as the genuine-vs-clone midpoint, and the code now computes that.
  At seed 0, the m1×ASVspoof center moves 0.244 → 0.203 and m2×In-the-Wild 0.455 → 0.408.

Effect on claim 2 (bayesian − parallel, TPR@10%FAR vs genuine_scam, 8 seeds):

| setting | before | after |
|---|---|---|
| m1 × ASVspoof | +0.122 (8/8) | +0.123 (8/8) |
| m2 × In-the-Wild | +0.135 (8/8) | **+0.127 (6/8)** |
| m2 × ASVspoof | −0.010 (1/8) | −0.030 (0/8) |
| m4 × ASVspoof | −0.018 (2/8) | −0.047 (1/8) |
| m4 × In-the-Wild | −0.031 (3/8) | −0.024 (3/8) |
| m1 × In-the-Wild | +0.033 (8/8) | −0.067 (0/8) |

The degradation-sweep peak moves from +0.171 to +0.159 (m2×ASVspoof, 15 dB). The docs also
said bootstrap CIs "do not overlap" in both winning settings. That held only for m1×ASVspoof
(bayesian [0.600, 0.705] vs parallel [0.422, 0.533]). For m2×In-the-Wild the single-seed
CIs overlapped both before the fix ([0.490, 0.754] vs [0.411, 0.534]) and after it
([0.491, 0.762] vs [0.411, 0.534]), so the docs no longer claim separation there.

**Net:** claim 2 keeps its scope (a gain of about +0.12–0.13 in 2 of 4 detectors), but
m2×In-the-Wild is now 6/8 seeds rather than 8/8, with overlapping CIs. Figures F1 and F2
have been regenerated.

## Addendum A — stronger baselines and label noise (2026-11-27; new analysis, not a bug)
No published number was wrong. Two robustness checks narrow claim 2, and anything public
that presents the *multiplicative form* as the advance should be qualified:
- **Stronger baselines.** In the two winning settings, a logistic-regression fusion with an
  acoustic×intent interaction term trails the multiplicative veto by only 0.011
  (m1×ASVspoof) and 0.016 (m2×In-the-Wild) TPR, and a min (AND) rule beats it by
  0.015–0.018. The gain over linear fusion is the gain of any conjunctive rule.
- **Label noise.** 3 of 15 cached "scam" dialogues self-rate as benign (peak intent
  0.1–0.5). Without them the gain over linear fusion is +0.093 (m1×ASVspoof) and +0.104
  (m2×In-the-Wild), plus +0.048 for m4×ASVspoof.
- **Model identities.** m3 = `mo-thecreator/Deepfake-audio-detection@e4d9874b4933` and m4 =
  `Hemgg/Deepfake-audio-detection@0d75271368ef`, confirmed by re-scoring (RESULTS.md, Method).

## Ready-to-paste arXiv v2 comment
> v2: corrected an AUC tie-handling bug (intent-only baseline 0.83→0.89, other metrics
> unchanged); expanded the evaluation to four detectors and 600 clips/source, which revised
> the multiplicative-vs-linear fusion gain from ~+0.25 to ~+0.12 and showed it holds in 2 of
> 4 detectors. The effect is real and seed-robust where it appears but modest and
> detector-specific; conclusions are qualified accordingly.

## Disclosure amendment note
If the filed disclosure cited +0.251 / "two detectors": the reduction-to-practice magnitude
is **+0.122** (600 clips) and the effect is **detector-specific (2 of 4)**. Update the
evidence and the non-obviousness framing in any continuation/amendment (see DISCLOSURE.md §7,
§9).

## Ready-to-paste comment for Correction 3
> Fixed tie handling in TPR@FAR (intent-only baseline vs genuine_scam 0.00→0.10, vs all
> negatives 0.00→0.50) and calibrated the acoustic center on genuine-vs-clone labels as the
> paper describes. Multiplicative-vs-linear gain: +0.123 (m1×ASVspoof, 8/8 seeds) and +0.127
> (m2×In-the-Wild, 6/8 seeds); scope unchanged (2 of 4 detectors), and CIs separate only for
> m1×ASVspoof.

## Ready-to-paste comment for Addendum A
> Added baselines: a logistic fusion with an acoustic×intent interaction term comes within
> 0.01–0.02 TPR of the multiplicative veto, and a min (AND) rule matches or beats it. The
> advantage over linear fusion is that of conjunctive fusion in general, not of the
> multiplicative form.
