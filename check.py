"""1-click risk check: hand it an audio file, get a verdict. No flags to remember.

  PYTHONPATH=. .venv/bin/python check.py call.mp3
  PYTHONPATH=. .venv/bin/python check.py call.mp3 --json   # machine-readable

Transcribes the call, scores voice (m2) + scam-intent, and rates the risk. See
voiceguard/assess.py for the core. Needs `pip install -r requirements-acoustic.txt`.
"""
import sys, json
from voiceguard.assess import assess
from voiceguard.detect import format_score


def bar(x, n=22):
    if x is None:
        return "unavailable"
    k = max(0, min(n, round(x * n)))
    return "#" * k + "." * (n - k)


def main():
    args = sys.argv[1:]
    paths = [a for a in args if not a.startswith("--")]
    if not paths:
        print("usage: python check.py <audio-file> [--json]")
        sys.exit(1)
    r = assess(paths[0])
    if "--json" in args:
        print(json.dumps(r, indent=2))
        return
    print(f"\nVoiceGuard  |  {r['audio']}")
    print(f"  detector: {r['detector'].split('/')[-1]}   intent: {r['intent_source']}\n")
    print("  transcript:")
    for t in r["transcript"]:
        print(f"    [{t['t']:5.1f}s] {t['text']}")
    if not r["transcript"]:
        print("    (no speech detected)")
    print(f"\n  acoustic score {format_score(r['acoustic'])}  ->  "
          f"clone score {format_score(r['clone_score'])}   ({r['pooling']})")
    if r["peak_segment"]:
        ps = r["peak_segment"]
        print(f"    most-synthetic segment {ps['start']}-{ps['end']}s   "
              f"(whole-file mean {r['acoustic_overall']:.2f}, "
              f"{int(r['frac_synthetic'] * 100)}% of scored windows above 0.5)")
    print(f"  scam-intent           {format_score(r['intent'])}")
    print(f"  voice-clone scam score [{bar(r['risk'])}] {format_score(r['risk'])}   [{r['level']}]\n")
    print(f"  {r['advice']}")
    for reason in r["reasons"]:
        print(f"  {reason}")
    print(f"  note: {r['caveat']}\n")


if __name__ == "__main__":
    main()
