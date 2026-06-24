# VoiceGuard — open items

The experimental arc, paper + figures, filed disclosure, and CLI prototype are all done.
This is what remains, in recommended order.

## Housekeeping — done in the initial commit
- [x] git init + .gitignore
- [x] complete requirements (`requirements.txt` core + `requirements-acoustic.txt`)

## Next — cheap, high value
- [x] Real-audio smoke test: verified on real ASVspoof clips — clone â=1.000 (flagged),
      genuine â=0.000 (vetoed). `scripts/fetch_samples.py` pulls samples; audio gitignored.
- [x] Wire LLM intent into the live CLI trace — `cli.py --llm` (uses the gateway per turn).

## Paper packaging
- [ ] arXiv-ready LaTeX build of `PAPER.md` (sections, F1/F2, bibliography for the cited
      patents and datasets, abstract/formatting).
- [ ] Pick a target venue; tailor scope and length to its CFP / page limit.

## Stronger evidence — larger
- [ ] Real paired scam audio for §4 (biggest validity gain; replaces the constructed
      benchmark's independence assumption).
- [ ] Tighten claim-2 CIs: a 3rd detector and/or more clips (needs a fresh GPU instance).

## Optional
- [ ] Interactive call-screen widget (drag acoustic/intent, watch the veto flip the
      decision in real time), driven by `detect.py:fuse`.
