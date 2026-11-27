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
.venv/bin/pip install -e ".[acoustic,app,dev]"   # everything, incl. pytest
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

The checker shares one decoded waveform between ASR and the acoustic detector. It uses
the calibration pipeline's soundfile/librosa decoding and resampling, with PyAV for
containers such as M4A. ASR uses voice activity detection. Audio below the detector's
0.5-second minimum, missing speech, or missing acoustic evidence produces
`INSUFFICIENT_EVIDENCE`, with `null` for unavailable scores and a reason in the output.

The displayed values are experimental **scores**. The combined value measures voice-clone
scam risk, and scam intent is displayed separately. The sigmoid and the 0.30/0.60 alert
thresholds still need probability calibration and validation on paired telephone calls.
Long-call max pooling also needs a false-alarm evaluation across recording lengths.

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
