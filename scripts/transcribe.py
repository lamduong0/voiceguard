"""Create a VoiceGuard transcript from an audio file (offline ASR, CPU-friendly).

  PYTHONPATH=. .venv/bin/python scripts/transcribe.py call.mp3
  PYTHONPATH=. .venv/bin/python scripts/transcribe.py call.mp3 --out turns.json --txt transcript.txt --model small

Writes turns.json in the format `cli.py --transcript` expects (a list of {"t","text"}),
plus an optional human-readable, timestamped transcript you can inspect or edit. Then
score the real call:

  PYTHONPATH=. .venv/bin/python cli.py --audio call.mp3 --transcript turns.json

(or `cli.py --audio call.mp3 --asr` to transcribe + score in one step). Needs
`pip install -r requirements-acoustic.txt` (faster-whisper).
"""
import json, argparse
from voiceguard.detect import transcribe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--out", default="turns.json")
    ap.add_argument("--txt", help="also write a human-readable, timestamped transcript here")
    ap.add_argument("--model", default="base",
                    help="faster-whisper size: tiny/base/small/medium/large-v3 (bigger = better, slower)")
    a = ap.parse_args()
    turns = transcribe(a.audio, model_size=a.model)
    json.dump(turns, open(a.out, "w"), indent=2)
    lines = [f"[{t['t']:6.1f}]  {t['text']}" for t in turns]
    if a.txt:
        open(a.txt, "w").write("\n".join(lines) + "\n")
    print(f"turns={len(turns)}  -> {a.out}" + (f", {a.txt}" if a.txt else ""))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
