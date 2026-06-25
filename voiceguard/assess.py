"""One-shot assessment: an audio file in, a risk verdict out. Shared core for the
zero-flag CLI (`check.py`) and the drag-drop web app (`app.py`).

Defaults to m2 (`wav2vec2-large-xlsr`), the detector that generalizes to real-world
audio (RESULTS.md claim 3), calibrated on its In-the-Wild scores — a real call
recording is the In-the-Wild regime, not the studio ASVspoof one. Intent uses the LLM
gateway when VG_LLM_KEY is set, else the keyword scorer.
"""
from voiceguard.detect import (AcousticScorer, transcribe, load_calibration,
                               fuse, decide, score_intent)

M2 = "Gustking/wav2vec2-large-xlsr-deepfake-audio-classification"
CAL_DATA, CAL_SOURCE = "data/events_m2_600.json", "itw"
CAVEAT = ("Indicative, not forensic: one open detector on out-of-distribution phone "
          "audio. Don't rely on this alone for a real decision.")

_scorer = None


def _get_scorer():
    """Lazy, cached — the web app reuses one loaded model across requests."""
    global _scorer
    if _scorer is None:
        _scorer = AcousticScorer(model=M2)
    return _scorer


def assess(audio_path, model_size="base"):
    """Transcribe -> acoustic score (m2) -> intent -> multiplicative-veto risk."""
    turns = transcribe(audio_path, model_size=model_size)
    acoustic = _get_scorer().score(audio_path)
    cal = load_calibration(path=CAL_DATA, source=CAL_SOURCE)
    intent, intent_src = score_intent(turns) if turns else (0.0, "none")
    risk, p_clone = fuse(acoustic, intent, cal)
    level, advice = decide(risk)
    return {"audio": audio_path, "detector": M2,
            "acoustic": round(acoustic, 3), "p_clone": round(p_clone, 3),
            "intent": round(intent, 3), "intent_source": intent_src,
            "risk": round(risk, 3), "level": level, "advice": advice,
            "caveat": CAVEAT, "transcript": turns}
