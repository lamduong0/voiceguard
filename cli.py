"""VoiceGuard CLI prototype — play a call turn-by-turn, watch risk evolve, intervene.

  python cli.py --scenario clone_scam        # AI impostor
  python cli.py --scenario genuine_scam      # real relative, same scam-sounding words
  python cli.py --scenario benign
  python cli.py --audio call.wav --transcript turns.json   # real model on your audio
  python cli.py --scenario clone_scam --json               # machine-readable

Risk = P(clone | acoustic) x scam-intent (the multiplicative veto). Intent is recomputed
over the turns heard so far, so the risk bar rises as a scam escalates; a genuine voice is
vetoed even when the words sound identical.
"""
import json, argparse
from voiceguard.detect import (load_calibration, fuse, decide, keyword_intent,
                               score_intent, AcousticScorer, transcribe)

SCAM_TURNS = [
    {"text": "Grandma, it's me!"},
    {"text": "I'm in trouble, please don't tell mom and dad."},
    {"text": "I got arrested after a car accident, I need bail money right now."},
    {"text": "Just buy gift cards and read me the codes — hurry, it's urgent."},
]
BENIGN_TURNS = [
    {"text": "Hi grandma, just calling to say hi!"},
    {"text": "How was your week?"},
    {"text": "We were thinking of coming over for dinner this weekend."},
    {"text": "Can I bring anything?"},
]


def acoustic_from_data(is_clone, path="data/acoustic_events_m1.json"):
    data = [r for r in json.load(open(path)) if r["source"] == "asv" and r["is_clone"] == is_clone]
    mean_a = lambda r: sum(p for _, p in r["events"]) / len(r["events"])
    pick = max(data, key=mean_a) if is_clone else min(data, key=mean_a)
    return mean_a(pick)


SCENARIOS = {
    "clone_scam": (True, SCAM_TURNS),
    "genuine_scam": (False, SCAM_TURNS),
    "benign": (False, BENIGN_TURNS),
}


def bar(x, width=22):
    n = max(0, min(width, int(round(x * width))))
    return "#" * n + "." * (width - n)


def play(acoustic, turns, cal, json_out=False, use_llm=False):
    trace, fired = [], None
    intent_src = "keyword"
    if use_llm and not json_out:
        print("(scoring intent via LLM gateway)\n")
    for k, turn in enumerate(turns):
        if use_llm:
            intent, intent_src = score_intent(turns[:k + 1])
        else:
            intent = keyword_intent(turns[:k + 1])
        risk, p_clone = fuse(acoustic, intent, cal)
        lvl, msg = decide(risk)
        trace.append({"turn": k + 1, "text": turn["text"], "intent": round(intent, 3),
                      "p_clone": round(p_clone, 3), "risk": round(risk, 3), "level": lvl})
        if not json_out:
            print(f"[t{k + 1}] caller: {turn['text']}")
            print(f"      risk [{bar(risk)}] {risk:.2f}   P_clone {p_clone:.2f} x intent {intent:.2f}   {lvl}")
        if fired is None and lvl == "HIGH":
            fired = k + 1
            if not json_out:
                print(f"      >>> INTERVENTION (turn {k + 1}): {msg}")
    final_risk, _ = fuse(acoustic, keyword_intent(turns), cal)
    final_lvl, final_msg = decide(final_risk)
    if json_out:
        print(json.dumps({"acoustic": round(acoustic, 3), "intent_source": intent_src,
                          "final_risk": round(final_risk, 3), "final_level": final_lvl,
                          "fired_at_turn": fired, "trace": trace}, indent=2))
    else:
        tail = f"  (first flagged at turn {fired})" if fired else "  (never flagged)"
        print(f"\nVerdict: [{final_lvl}] {final_msg}{tail}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=list(SCENARIOS), default="clone_scam")
    ap.add_argument("--audio")
    ap.add_argument("--transcript")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--llm", action="store_true",
                    help="score intent via the LLM gateway (needs VG_LLM_KEY) instead of keywords")
    ap.add_argument("--asr", action="store_true",
                    help="transcribe the --audio file (faster-whisper) and score its real words")
    a = ap.parse_args()
    cal = load_calibration()
    if a.audio:
        acoustic = AcousticScorer().score(a.audio)
        if a.asr:
            turns = transcribe(a.audio)
            if not a.json:
                print("(transcript from ASR of the audio)\n")
        elif a.transcript:
            turns = json.load(open(a.transcript))
        else:
            turns = [{"text": "(no transcript)"}]
    else:
        is_clone, turns = SCENARIOS[a.scenario]
        acoustic = acoustic_from_data(is_clone)
    if not a.json:
        print(f"VoiceGuard  |  calibration c={cal['c']} g={cal['g']}  |  acoustic P(synthetic)={acoustic:.3f}\n")
    play(acoustic, turns, cal, json_out=a.json, use_llm=a.llm)


if __name__ == "__main__":
    main()
