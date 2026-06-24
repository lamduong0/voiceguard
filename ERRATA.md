# Errata / changelog — reconciling the first filed/posted version with current

Two corrections landed after the initial invention disclosure / preprint. Anything already
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
