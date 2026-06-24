# VoiceGuard — cross-modal adaptive fusion study

Real-time defense against AI voice-clone phone scams. The broad "live + on-device +
fuse acoustic/intent/metadata + alert" claim is already patented (US12284313B1,
US10455085B1, US9692885B2; Pindrop, Aurigin). This project tests the one mechanism
that might still be novel and is worth a paper either way:

> **Cross-modal adaptive fusion** — rising scam-*intent* dynamically lowers the
> acoustic deepfake detector's decision threshold *during the call*, instead of
> combining the two signals with fixed weights (parallel fusion, the prior-art baseline).

## Stages
- **Stage 1 (this code, CPU):** mechanism study on *simulated* detector outputs.
  Does adaptive fusion detect clone+scam calls earlier, at equal false-alarm rate,
  vs a tuned parallel baseline — and does its edge grow as acoustic audio gets
  ambiguous? Numbers here judge a *model of* the detectors, not real audio.
- **Stage 2 (later, GPU):** real acoustic model + LLM intent on ASVspoof / In-the-Wild
  audio paired with scam/benign transcripts. The paper's real numbers.

## Run
```bash
python3.11 -m venv .venv && .venv/bin/pip install -q numpy pytest
PYTHONPATH=. .venv/bin/python -m pytest tests -q
PYTHONPATH=. .venv/bin/python experiments/stage1_mechanism.py
```

## Demo
End-to-end prototype: audio + transcript -> multiplicative-veto risk -> in-call intervention.
Core logic in `voiceguard/detect.py`.

Turn-by-turn "live call" player (risk evolves as the scam escalates; intervention fires on
threshold crossing):
```bash
PYTHONPATH=. .venv/bin/python cli.py --scenario clone_scam     # AI impostor -> flagged
PYTHONPATH=. .venv/bin/python cli.py --scenario genuine_scam   # real relative, same words -> vetoed
PYTHONPATH=. .venv/bin/python cli.py --scenario benign
PYTHONPATH=. .venv/bin/python cli.py --audio call.wav --transcript turns.json   # real model
PYTHONPATH=. .venv/bin/python cli.py --scenario clone_scam --json               # machine-readable
```
One-shot scorer (three scenarios side by side):
```bash
PYTHONPATH=. .venv/bin/python demo.py --example
```

