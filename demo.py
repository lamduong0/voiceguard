"""VoiceGuard demo: audio + transcript -> multiplicative-veto risk -> in-call intervention.

  python demo.py --example                      # offline, uses real precomputed acoustic scores
  python demo.py --audio call.wav --transcript turns.json   # real model on your audio

The --example run shows the mechanism: a clone+scam call and a GENUINE panicked-relative
call with the SAME scam-sounding words get opposite decisions, because the acoustic veto
separates them where intent alone cannot.
"""
import json, argparse
from voiceguard.detect import load_calibration, fuse, score_intent, decide, AcousticScorer

SCAM = [{"text": "Grandma, it's me, please don't tell mom and dad."},
        {"text": "I got arrested after an accident, I need bail money right now."},
        {"text": "Just buy gift cards and read me the codes, it's urgent."}]
BENIGN = [{"text": "Hi grandma, just calling to say hi."},
          {"text": "How was your week?"},
          {"text": "Let's plan dinner this weekend."}]


def report(name, a, turns, cal):
    i, src = score_intent(turns)
    risk, p_clone = fuse(a, i, cal)
    lvl, msg = decide(risk)
    print(f"# {name}")
    print(f"  acoustic P(synthetic) a = {a:.3f}   -> P(clone) = {p_clone:.3f}")
    print(f"  intent ({src})         i = {i:.3f}")
    print(f"  RISK = {risk:.3f}   ->   [{lvl}] {msg}\n")


def example():
    cal = load_calibration()
    data = [r for r in json.load(open("data/acoustic_events_m1.json")) if r["source"] == "asv"]
    mean_a = lambda r: sum(p for _, p in r["events"]) / len(r["events"])
    clone_a = mean_a(max((r for r in data if r["is_clone"]), key=mean_a))
    genuine_a = mean_a(min((r for r in data if not r["is_clone"]), key=mean_a))
    print(f"calibration (real ASVspoof data): center c={cal['c']}, gain g={cal['g']}, "
          f"genuine~{cal['mean_genuine']}, clone~{cal['mean_clone']}\n")
    report("clone + scam  (AI impostor)", clone_a, SCAM, cal)
    report("genuine + scam-sounding  (real panicked relative, SAME words)", genuine_a, SCAM, cal)
    report("genuine + benign", genuine_a, BENIGN, cal)
    print("Note: rows 1 and 2 have identical intent; only the acoustic veto separates them.")


def real(audio, transcript):
    cal = load_calibration()
    turns = json.load(open(transcript)) if transcript else [{"text": "(no transcript)"}]
    a = AcousticScorer().score(audio)
    report(f"audio={audio}", a, turns, cal)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio")
    ap.add_argument("--transcript")
    ap.add_argument("--example", action="store_true")
    args = ap.parse_args()
    if args.audio:
        real(args.audio, args.transcript)
    else:
        example()
