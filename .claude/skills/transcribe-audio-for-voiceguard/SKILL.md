---
name: transcribe-audio-for-voiceguard
description: Use when given an audio file (mp3/wav/m4a) and asked to transcribe it, build a transcript or turns.json, or check whether a real call recording is an AI voice clone or scam with VoiceGuard. Covers ASR plus running the acoustic-and-intent detector and its known pitfalls.
---

# Transcribe audio & run VoiceGuard on a real call

## Overview
VoiceGuard scores a call as `risk = P(synthetic voice) × scam-intent`. To run it on a real
recording you need two things from the audio: the **acoustic** score (model reads the audio
directly) and a **transcript** for the intent score. This skill produces the transcript and
runs the full pipeline.

Run from the repo root with `PYTHONPATH=. .venv/bin/python`. Needs
`pip install -r requirements-acoustic.txt` (faster-whisper + torch + transformers).

## Steps

**One-shot (transcribe + score):**
```bash
PYTHONPATH=. .venv/bin/python cli.py --audio call.mp3 --asr --json
```

**Or transcribe to a file first, then score (lets you inspect/edit the transcript):**
```bash
PYTHONPATH=. .venv/bin/python scripts/transcribe.py call.mp3 --out turns.json --txt transcript.txt
PYTHONPATH=. .venv/bin/python cli.py --audio call.mp3 --transcript turns.json --json
```

`turns.json` is a list of `{"t","text"}` (segment start-second + words) — one entry per
speech segment, which VoiceGuard treats as a turn. mp3, wav, m4a all decode. Bigger
`--model` (e.g. `medium`) = more accurate, slower; `base`/`small` is fine for short clips
on CPU.

## Reading the result
- `final_level` LOW / MEDIUM / HIGH and `final_risk = p_clone × intent`.
- `acoustic` is mean P(synthetic) over 1s windows; `p_clone` is its calibrated sigmoid.
- A per-turn `trace` shows where intent rose.

## Gotchas (these bite — verified on real audio)
- **Default acoustic model (m1, MelodyMachine) is near-chance on real-world audio** (RESULTS.md
  claim 3, AUC 0.519, scores can invert). A real phone call is out-of-distribution. For a
  trustworthy acoustic read, score with **m2** (`Gustking/wav2vec2-large-xlsr-deepfake-audio-classification`)
  via `scripts/score_local.py` and compare. Don't quote m1's `p_clone` as fact.
- **Keyword intent is coarse** (single-word cues → easy false positives; a `\bwon\b`-matches-
  "won't" bug was fixed, but the failure class remains). For real transcripts prefer `--llm`
  (needs `VG_LLM_KEY`) over the keyword scorer.
- **MP3 artifacts** can spike a window or two; a lossless WAV gives a cleaner per-window read.
  Cross-check the timestamps where acoustic spikes against the transcript — a telephony "this
  number is disconnected" message is genuinely synthetic and *should* score high.
- **No transcript ≠ no answer:** acoustic alone answers "is the voice synthetic"; intent needs
  words. Without `--asr`/`--transcript` the CLI uses a placeholder turn and intent is meaningless.

## When NOT to use
Synthetic scenarios (`--scenario clone_scam` etc.) need no audio. Reproducing paper numbers
uses `experiments/` on the bundled `data/*.json`, not live ASR.
