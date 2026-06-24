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
- [x] arXiv-ready LaTeX source: `paper/main.tex` + `references.bib`. Compiles clean to
      `paper/main.pdf` via `tectonic main.tex` (PDF gitignored as a build artifact).
- [x] Venue picked: see `VENUE.md` (Interspeech primary, DLS fallback) + tailoring plan.
      Still needs your final call + author list.

## GitHub
- [x] LICENSE added (MIT, (c) NVIDIA Corporation — change holder if personal).
- [x] Pushed to github.com/lamduong0/voiceguard (PRIVATE). Flip public or transfer to an
      NVIDIA org after legal/OSPO sign-off.

## Stronger evidence — larger
- [~] Real paired scam audio for §4 — sourcing plan in `DATA_SOURCING.md` (PITCH / CallHome+TTS
      / PartialSpoof). Actual data gated by licensing/consent, not engineering.
- [x] 3rd detector (mo-thecreator), scored locally on CPU (no GPU). Did NOT simply confirm
      claim 2: near-perfect on clean ASVspoof (tie) and collapses under noise, exposing that
      the veto needs a detector that degrades *gracefully*. Recorded in RESULTS.md / paper.
      (More clips / a 4th graceful detector could still tighten the win-regime CIs.)

## Optional
- [x] Interactive widget: `demo/voiceguard_demo.html` (standalone; drag acoustic/intent or
      use presets, watch the veto flip the decision; same fuse math + calibration).
