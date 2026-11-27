# VoiceGuard — cross-modal fusion for voice-clone scam detection

> A multiplicative **acoustic veto** that flags AI voice-clone scams *without* crying wolf on
> a real, panicked relative whose words happen to sound like one. Honest finding: the
> mechanism helps, but modestly (≈+0.12–0.13 TPR@10%FAR) and in only 2 of 4 detectors tested.

Research code for a paper + invention disclosure on combining an **acoustic deepfake
detector** with an **LLM scam-intent** signal to flag AI voice-clone phone scams. The broad
"live, fuse acoustic/intent/metadata, alert during call" system is already prior art
(US12284313B1, US10455085B1, US9692885B2; Pindrop, Aurigin); the contribution here is a
specific fusion *mechanism* and an honest evaluation methodology.

## The mechanism

```
risk_score = sigmoid(g · (â − c)) · î   # â = acoustic score, î = scam-intent, c = fitted center
```

A **multiplicative acoustic veto** uses genuine-voice evidence (small `â`) to reduce the
combined score even when intent is high. It targets the hard case of a panicked relative whose
words sound like a scam (`genuine_scam`) — where linear (weighted-sum) fusion cries wolf.
(An earlier *additive/adaptive-threshold* variant was tested and failed; see RESULTS.md.)

## Findings (see [RESULTS.md](RESULTS.md) for numbers + CIs)

1. **Fusion beats either modality alone.** Intent alone can't tell a clone-scam from a
   real-voice scam (chance level, ≈0.10 TPR@10%FAR vs `genuine_scam`); acoustic alone is weak on the broad
   negative set. Only fusion is strong on both axes.
2. **Multiplicative beats linear fusion in specific settings.** A gain of
   ≈+0.12–0.13 TPR at nominal 10% FAR appears in **2 of 4 detectors tested**
   (8/8 seeds with separated bootstrap CIs for m1×ASVspoof, 6/8 seeds for
   m2×In-the-Wild). It helps in specific
   detector×regime combinations with a clear mechanism, not as a general law.
3. **Detector generalization is the binding constraint** and is model-dependent: off-the-
   shelf detectors range from chance to strong on real-world (In-the-Wild) deepfakes, and no
   fusion recovers a broken acoustic channel.

## Status

The committed score experiments are reproducible. Drafted: [PAPER.md](PAPER.md) /
[paper/main.tex](paper/main.tex) (compiles to PDF via `tectonic`), [DISCLOSURE.md](DISCLOSURE.md)
(narrow invention disclosure), [VENUE.md](VENUE.md), [DATA_SOURCING.md](DATA_SOURCING.md),
figures F1/F2. Open: real paired scam audio (licensing-gated) and a final venue choice.

The evaluation scripts report the expected false-alarm rate with randomized tie handling
alongside the nominal 10% FAR target. `stage2_roc.py` also reports the threshold, tie weight,
and strict-threshold FAR. Finite samples can put FAR slightly above the target.
`stage2_roc.py` also reports paired bootstrap
confidence intervals for the multiplicative-minus-linear TPR difference, conditional on
the fitted scores. Speaker and script holdouts require a new paired-call evaluation.

## Layout

- `voiceguard/` — core: `fusion.py`, `evaluate.py`, `simulate.py` (Stage 1 sim),
  `realsignals.py`, `intent.py`, `detect.py` (demo core), `assess.py` (1-click pipeline)
- `experiments/` — `stage1_*` (simulation), `stage2_*` (real-data: roc/harden/degrade/run),
  `make_figures.py`
- `scripts/`, `remote/` — dataset fetch + GPU/CPU scoring
- `check.py` (1-click CLI), `app.py` (drag-drop web app), `cli.py`, `demo.py`,
  `demo/voiceguard_demo.html` — runnable prototypes
- `pyproject.toml` — package + dependency extras; `data/` — derived per-clip scores (JSON);
  `figures/` — F1/F2

## Install
```bash
python3.11 -m venv .venv
.venv/bin/pip install -e .                 # core (CPU): sim, analysis, offline demo, LLM intent
.venv/bin/pip install -e ".[acoustic]"     # + real acoustic scoring & ASR (torch/transformers)
.venv/bin/pip install -e ".[app]"          # + the drag-drop web app (gradio)
.venv/bin/pip install -e ".[figures]"      # + matplotlib, for experiments/make_figures.py
.venv/bin/pip install -e ".[acoustic,app,figures,dev]"   # everything, incl. pytest
```
Dependencies are declared in `pyproject.toml` (single source); the `requirements*.txt` files
are thin shims that map to these extras. An editable install puts `voiceguard` on the path,
so **`PYTHONPATH=.` is no longer needed** — the commands below assume it's installed; from a
bare, uninstalled checkout, prefix them with `PYTHONPATH=.` instead.

To reproduce the **exact** environment the RESULTS.md / paper numbers were produced in, use
the pinned provenance freeze `requirements.lock` (macOS/arm64, CPython 3.11 — see its header).
For normal use, prefer the version floors above.

## Run
```bash
.venv/bin/python -m pytest tests -q
.venv/bin/python -m experiments.stage1_mechanism                         # Stage 1 sim
.venv/bin/python -m experiments.stage2_harden data/events_m1_600.json 8  # claim 2
```
Run experiments as modules (`-m`) from the repo root: `experiments/` isn't part of the
installed package, and `-m` puts the repo root on the path.

## Demo
End-to-end prototype: audio + transcript → multiplicative-veto risk → in-call intervention.
Core logic in `voiceguard/detect.py`; interactive widget at `demo/voiceguard_demo.html`.

**1-click check** — hand it a recording, get a verdict (transcribe → score voice + intent →
risk). Backed by the m2 detector that generalizes to real-world audio (see RESULTS.md claim 3);
long calls are chunk-pooled so a localized clone isn't averaged away.
```bash
.venv/bin/python check.py call.mp3          # CLI: prints verdict + transcript
.venv/bin/pip install -e ".[app]"           # then, for the web app:
.venv/bin/python app.py                     # drag-drop upload in the browser
```
Shared core in `voiceguard/assess.py`; needs the `acoustic` extra (`pip install -e ".[acoustic]"`).
LLM intent runs when `VG_LLM_KEY` is set and sends the transcript to `VG_LLM_BASE_URL`
(OpenAI by default). The result identifies the intent backend. With the key unset,
intent uses the local keyword scorer.

The checker decodes once for ASR and the acoustic detector. Resampling is always
librosa's, like the calibration pipeline's; PyAV only decodes containers such as M4A at
their native rate, since FFmpeg's resampler shifted detector scores. Multi-channel audio is
averaged to mono. ASR uses voice activity detection, and the acoustic score uses only the
one-second windows that overlap detected speech. Audio below the detector's 0.5-second
minimum, missing speech, or missing acoustic evidence produces `INSUFFICIENT_EVIDENCE`,
with `null` for unavailable scores and a reason in the output.

The displayed values are experimental **scores**. The combined value measures voice-clone
scam risk, and scam intent is displayed separately.

**Codec-matched calibration.** Lossy codecs shift m2's scores, so the checker uses the
scaling fit on In-the-Wild clips passed through the recording's codec (`codec` in the
output). `scripts/score_codecs.py` produced `data/codec_m2_itw.json`: the 1,200 clips
behind the uncompressed calibration, each encoded and decoded through the checker's
loader. Mean score genuine / clone, and genuine clips flagged HIGH (intent at its
maximum) with the uncompressed calibration vs the codec's own:

| codec (as fit) | genuine / clone | HIGH on genuine: uncompressed cal. → own |
|---|---|---|
| uncompressed (WAV, FLAC, ALAC) | 0.20 / 0.62 | 7.5% |
| AAC 64 kb/s | 0.19 / 0.57 | 5.3% → 7.0% |
| MP3 64 kb/s | 0.20 / 0.61 | 6.7% → 7.2% |
| Opus 24 kb/s | 0.30 / 0.66 | 20.2% → 15.0% |
| AMR-NB 12.2 kb/s | 0.23 / 0.70 | 9.2% → 5.3% |

Opus still over-alarms about 2× at HIGH. Other codecs (e.g. Vorbis) fall back to the
uncompressed calibration with a reason, and other bit rates may shift further. (A
one-clip TTS test once suggested AAC raises scores; on real speech it lowers both classes
slightly.)

**Long calls.** A long call is judged by its most synthetic 30-second chunk, and more
chunks give a genuine call more chances to cross the center c. Simulating calls from the
per-second In-the-Wild scores (`experiments/fit_pooling.py`; two speakers taking turns,
each one clip's scores resampled), a genuine call crosses c 9.8% of the time per clip but
12.4% at 2 minutes of speech and 16.9% at 16 minutes. The checker raises c by the fitted
shift (0 up to 1 minute, +0.027 at 2, +0.061 at 8, +0.086 at 16; `center_shift` in the
output), which holds it at 9.8% and lowers clone-scam detection from 0.69–0.77 to
0.65–0.68. Real calls drift more than one resampled clip, so treat this as a floor.

**Dual-channel calls** are mixed to mono. Per-channel scoring was tried and removed: m2
scores a quiet line's noise floor as synthetic (0.65–0.79 at −60 to −40 dBFS), and in the
same simulation it caught no more clones at an equal false-alarm rate.

**Alert thresholds.** On uncompressed In-the-Wild clips with intent at its maximum (a
genuine voice saying scam-like words), MEDIUM (≥ 0.30) flags 20.5% of genuine and 84% of
clone clips, and HIGH (≥ 0.60) flags 7.5% and 67%. These are clip-level operating points;
probability calibration and validation on paired telephone calls are still open.

Score-scaling parameters ship in `voiceguard/calibration.json`, with a version, source
data hash, and pinned runtime detector revision. Default loading is independent of the
working directory. Explicit calibration files must be valid and contain both classes.
JSON output includes `clone_score`; the legacy `p_clone` field aliases that score.

Turn-by-turn "live call" player (risk evolves as the scam escalates; intervention fires on
threshold crossing):
```bash
.venv/bin/python cli.py --scenario clone_scam     # AI impostor -> flagged
.venv/bin/python cli.py --scenario genuine_scam   # real relative, same words -> vetoed
.venv/bin/python cli.py --scenario benign
.venv/bin/python cli.py --audio call.wav --transcript turns.json   # real model
.venv/bin/python cli.py --audio call.mp3 --asr                      # transcribe (ASR) + score real words
.venv/bin/python cli.py --scenario clone_scam --json               # machine-readable
```
**Transcribe your own audio** (mp3/wav/m4a) into a `turns.json` you can inspect or edit
before scoring — `--asr` above does this inline; `scripts/transcribe.py` saves it to a file:
```bash
.venv/bin/python scripts/transcribe.py call.mp3 --out turns.json --txt transcript.txt
.venv/bin/python cli.py --audio call.mp3 --transcript turns.json
```
ASR needs the `acoustic` extra (`pip install -e ".[acoustic]"`, faster-whisper). Caveat: the
bundled `cli.py` acoustic detector (m1) is near-chance on real-world/phone audio — see RESULTS.md
claim 3. (The 1-click `check.py`/`app.py` use the stronger m2 by default.)
One-shot scorer (three scenarios side by side):
```bash
.venv/bin/python demo.py --example
```
LLM-scored intent (`cli.py --llm`, or regenerating `data/dialogues.json`) reads `VG_LLM_KEY`
and optionally `VG_LLM_BASE_URL` / `VG_LLM_MODEL` (any OpenAI-compatible endpoint). The
offline demo uses a keyword scorer and needs none.
