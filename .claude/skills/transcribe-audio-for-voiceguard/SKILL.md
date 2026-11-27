---
name: transcribe-audio-for-voiceguard
description: Use when given an audio file (mp3/wav/m4a) and asked to transcribe it, build a transcript or turns.json, or check whether a real call recording is an AI voice clone or scam with VoiceGuard. Covers ASR plus running the acoustic-and-intent detector and its known pitfalls.
---

# Transcribe audio & run VoiceGuard on a real call

## Overview
VoiceGuard scores a call as `risk = clone score × scam-intent`. To run it on a real
recording you need two things from the audio: the **acoustic** score (model reads the audio
directly) and a **transcript** for the intent score. This skill produces the transcript and
runs the full pipeline.

Run from the repo root after `pip install -e ".[acoustic]"` (faster-whisper + torch +
transformers). Commands below use `.venv/bin/python`.

## Steps

**One-shot verdict (recommended — m2 detector, the one that transfers to real-world audio):**
```bash
.venv/bin/python check.py call.m4a --json
```

**Transcript only, to inspect or edit it:**
```bash
.venv/bin/python scripts/transcribe.py call.m4a --out turns.json --txt transcript.txt
```

**Turn-by-turn intent trace** (`cli.py` uses the m1 detector, near chance on real audio —
use it to see where intent rises, not for the voice verdict):
```bash
.venv/bin/python cli.py --audio call.m4a --transcript turns.json --json
```

`turns.json` is a list of `{"t", "end", "text"}` (segment start/end seconds + words), one
entry per speech segment. wav, mp3 and m4a all decode. Bigger `--model` (e.g. `medium`) =
more accurate, slower; `base`/`small` is fine for short clips on CPU.

## Reading the `check.py` result
- `level`: LOW / MEDIUM / HIGH, or `INSUFFICIENT_EVIDENCE` with the cause in `reasons`
  (no speech, too short, undecodable, or a failed stage).
- `risk = clone_score × intent`. `acoustic` is the detector's mean synthetic-voice score over
  the one-second windows that contain speech (max over 30 s chunks on long calls);
  `clone_score` is its sigmoid scaling. `intent_source` says `llm` or `keyword`.
- `peak_segment` (long calls) says where the most synthetic 30 s are.

## Gotchas (these bite — verified on real audio)
- **Codecs shift the voice score.** The checker detects the codec and uses a matching
  calibration (`codec` in the JSON): AAC, MP3, Opus, AMR-NB, or uncompressed. Opus still
  over-alarms about 2×; an uncalibrated codec falls back with a note in `reasons`. MP3
  artifacts can spike single windows — cross-check where acoustic spikes against the
  transcript; a telephony "this number is disconnected" message is genuinely synthetic and
  *should* score high.
- **Dual-channel recordings are mixed to mono.** Per-channel scoring was tried and removed:
  m2 scores a quiet line's noise floor as synthetic (~0.7), and at equal false alarms it
  caught no more clones in simulation.
- **Long calls**: the verdict uses the most synthetic 30 s, and `center_shift` (in the
  JSON) raises the center with call length so long genuine calls don't over-alarm.
- **Keyword intent is coarse** (single-word cues → easy false positives). For real
  transcripts set `VG_LLM_KEY` to use the LLM scorer; the transcript is then sent to
  `VG_LLM_BASE_URL` (OpenAI by default).
- **No transcript ≠ no answer:** the voice evidence (`clone_score`) is still reported, but
  the combined `risk` needs words, so the level is `INSUFFICIENT_EVIDENCE`.

## When NOT to use
Synthetic scenarios (`--scenario clone_scam` etc.) need no audio. Reproducing paper numbers
uses `experiments/` on the bundled `data/*.json`, not live ASR.
