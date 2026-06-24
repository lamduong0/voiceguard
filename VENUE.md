# Venue plan

## Recommendation: Interspeech (primary), DLS workshop (fallback)

**Primary target: Interspeech.** It is the home of the anti-spoofing community (the
ASVspoof line, AASIST, the In-the-Wild "does it generalize?" conversation). Our claim 3
(model-dependent generalization) and the ASVspoof/In-the-Wild datasets land directly in
their wheelhouse, and an Interspeech paper meaningfully strengthens the arXiv defensive
publication. Format: 4 pages of content + 1 page of references; Interspeech LaTeX template.

**Fallback / security-framed alternative: DLS (Deep Learning and Security Workshop, co-located
with IEEE S&P).** Better fit for the *cautionary* half of the result (fusion is conditional,
off-the-shelf detectors don't transfer) and for a defensive-publication posture; workshop
reviewers reward honest boundary characterizations. Shorter, faster turnaround. Use this if
the Interspeech deadline is missed or a reviewer-friendly home for the negative findings is
preferred.

(ICASSP is a reasonable third option — same 4-page signal-processing format — if timing
aligns better.)

## Tailoring for Interspeech (4 + 1 pages)
- **Swap** `\documentclass{article}` for the Interspeech template (`interspeech.sty`).
- **Keep:** the three claims, the `genuine_scam` hard negative, F1 (ROC) and F2 (gain vs
  detector quality). These are the paper.
- **Tighten:** related work to one dense paragraph; method to the four scorers + the
  multiplicative-veto rationale; merge setup into results.
- **Cut to appendix/supplement (or drop):** the Stage 1 simulation non-result (one or two
  sentences in-text is enough); the per-SNR table can become the F2 figure alone.
- **Abstract:** already ~200 words; trim to the three findings.
- **Add (small experiments):** the 3rd-detector / degradation-sweep results once finalized,
  to firm claim 2; cite the constructed-benchmark release.

## Open decisions for the author
- Confirm Interspeech vs DLS given your timeline (deadlines move yearly — check the current
  CFP).
- Author list / affiliation (currently placeholder in `paper/main.tex`).
- Whether to release the benchmark construction publicly with the paper (recommended; aligns
  with the defensive-publication goal).
