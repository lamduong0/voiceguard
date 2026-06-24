# VoiceGuard — open items

The experimental arc, paper + figures, filed disclosure, and CLI prototype are all done.
This is what remains, in recommended order.

## Housekeeping — done in the initial commit
- [x] git init + .gitignore
- [x] complete requirements (`requirements.txt` core + `requirements-acoustic.txt`)

## Next — cheap, high value
- [ ] Real-audio smoke test: run `cli.py --audio call.wav --transcript turns.json` on a
      real clip end-to-end (needs `requirements-acoustic.txt` + one `.wav`). Verifies the
      only untested code path (`detect.py:AcousticScorer`).
- [ ] Wire LLM intent into the live CLI trace (currently keyword-only per turn; the LLM
      path exists in `detect.py` but isn't used by `cli.py`).

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
