# VoiceGuard — cross-modal fusion for voice-clone scam detection

Research code for a paper + invention disclosure on combining an **acoustic deepfake
detector** with an **LLM scam-intent** signal to flag AI voice-clone phone scams. The broad
"live, fuse acoustic/intent/metadata, alert during call" system is already prior art
(US12284313B1, US10455085B1, US9692885B2; Pindrop, Aurigin); the contribution here is a
specific fusion *mechanism* and an honest evaluation methodology.

## The mechanism

```
risk = sigmoid(g · (â − c)) · î          # â = P(synthetic voice), î = scam-intent, c = calibrated center
```

A **multiplicative acoustic veto**: strong genuine-voice evidence (small `â`) drives risk to
zero *regardless of intent*, so it rejects the hard case — a real, panicked relative whose
words sound like a scam (`genuine_scam`) — where linear (weighted-sum) fusion cries wolf.
(An earlier *additive/adaptive-threshold* variant was tested and failed; see RESULTS.md.)

## Findings (see [RESULTS.md](RESULTS.md) for numbers + CIs)

1. **Fusion beats either modality alone.** Intent alone can't tell a clone-scam from a
   real-voice scam (0.00 TPR vs `genuine_scam`); acoustic alone is weak on the broad
   negative set. Only fusion is strong on both axes.
2. **Multiplicative beats linear fusion — but modestly and not universally.** A clean,
   seed-robust win (≈+0.12–0.14 TPR@10%FAR, non-overlapping bootstrap CIs) in **2 of 4
   detector×dataset settings**; the other two show no gain. It helps in specific
   detector×regime combinations with a clear mechanism, not as a general law.
3. **Detector generalization is the binding constraint** and is model-dependent: off-the-
   shelf detectors range from chance to strong on real-world (In-the-Wild) deepfakes, and no
   fusion recovers a broken acoustic channel.

## Status

Experimental arc complete and code-reviewed. Drafted: [PAPER.md](PAPER.md) /
[paper/main.tex](paper/main.tex) (compiles to PDF via `tectonic`), [DISCLOSURE.md](DISCLOSURE.md)
(narrow invention disclosure), [VENUE.md](VENUE.md), [DATA_SOURCING.md](DATA_SOURCING.md),
figures F1/F2. Open: real paired scam audio (licensing-gated) and a final venue choice.

## Layout

- `voiceguard/` — core: `fusion.py`, `evaluate.py`, `simulate.py` (Stage 1 sim),
  `realsignals.py`, `intent.py`, `detect.py` (demo core)
- `experiments/` — `stage1_*` (simulation), `stage2_*` (real-data: roc/harden/degrade/run),
  `make_figures.py`
- `scripts/`, `remote/` — dataset fetch + GPU/CPU scoring
- `cli.py`, `demo.py`, `demo/voiceguard_demo.html` — runnable prototypes
- `data/` — derived per-clip scores (JSON); `figures/` — F1/F2

## Run
```bash
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt   # core (CPU)
PYTHONPATH=. .venv/bin/python -m pytest tests -q
PYTHONPATH=. .venv/bin/python experiments/stage1_mechanism.py           # Stage 1 sim
PYTHONPATH=. .venv/bin/python experiments/stage2_harden.py data/events_m1_600.json 8   # claim 2
# real acoustic scoring needs: pip install -r requirements-acoustic.txt
```

## Demo
End-to-end prototype: audio + transcript → multiplicative-veto risk → in-call intervention.
Core logic in `voiceguard/detect.py`; interactive widget at `demo/voiceguard_demo.html`.

Turn-by-turn "live call" player (risk evolves as the scam escalates; intervention fires on
threshold crossing):
```bash
PYTHONPATH=. .venv/bin/python cli.py --scenario clone_scam     # AI impostor -> flagged
PYTHONPATH=. .venv/bin/python cli.py --scenario genuine_scam   # real relative, same words -> vetoed
PYTHONPATH=. .venv/bin/python cli.py --scenario benign
PYTHONPATH=. .venv/bin/python cli.py --audio call.wav --transcript turns.json   # real model
PYTHONPATH=. .venv/bin/python cli.py --audio call.mp3 --asr                      # transcribe (ASR) + score real words
PYTHONPATH=. .venv/bin/python cli.py --scenario clone_scam --json               # machine-readable
```
One-shot scorer (three scenarios side by side):
```bash
PYTHONPATH=. .venv/bin/python demo.py --example
```
LLM-scored intent (`cli.py --llm`, or regenerating `data/dialogues.json`) reads `VG_LLM_KEY`
and optionally `VG_LLM_BASE_URL` / `VG_LLM_MODEL` (any OpenAI-compatible endpoint). The
offline demo uses a keyword scorer and needs none.
