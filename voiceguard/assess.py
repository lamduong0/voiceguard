"""One-shot assessment: an audio file in, a risk verdict out. Shared core for the
zero-flag CLI (`check.py`) and the drag-drop web app (`app.py`).

Defaults to m2 (`wav2vec2-large-xlsr`), the detector that generalizes to real-world
audio (RESULTS.md claim 3), calibrated on its In-the-Wild scores — a real call
recording is the In-the-Wild regime, not the studio ASVspoof one. Intent uses the LLM
gateway when VG_LLM_KEY is set, else the keyword scorer.

Long-file handling: the acoustic detector is scored per second over the whole file, then
pooled by CHUNK over ~CHUNK_S-second segments. The verdict uses the *most synthetic*
chunk, so a short clone inside a long genuine call surfaces instead of being averaged
away. Each chunk's mean is the same statistic the calibration center was fit on, so the
sigmoid stays valid. For files <= CHUNK_S this reduces exactly to the whole-file mean.
"""
from voiceguard.detect import (AcousticScorer, transcribe, load_calibration,
                               fuse, decide, score_intent)

M2 = "Gustking/wav2vec2-large-xlsr-deepfake-audio-classification"
CAL_DATA, CAL_SOURCE = "data/events_m2_600.json", "itw"
CHUNK_S, STRIDE_S = 30, 15
CAVEAT = ("Indicative, not forensic: one open detector on out-of-distribution phone "
          "audio. Don't rely on this alone for a real decision.")

_scorer = None


def _get_scorer():
    """Lazy, cached — the web app reuses one loaded model across requests."""
    global _scorer
    if _scorer is None:
        _scorer = AcousticScorer(model=M2)
    return _scorer


def _pool(windows, cal):
    """Slide ~CHUNK_S-second chunks (stride STRIDE_S), take the most-synthetic chunk.
    Returns (peak_start_s, peak_end_s, peak_mean, peak_p_clone, overall_mean, frac_high)."""
    probs = [p for _, p in windows]
    overall = sum(probs) / len(probs)
    frac_high = sum(1 for p in probs if p > 0.5) / len(probs)
    peak, i = None, 0
    while i < len(probs):
        seg = probs[i:i + CHUNK_S]
        if not seg:
            break
        m = sum(seg) / len(seg)
        _, pc = fuse(m, 1.0, cal)                      # calibrated P(clone) for this chunk
        end_s = windows[min(i + CHUNK_S, len(windows)) - 1][0] + 1
        if peak is None or m > peak[2]:
            peak = (windows[i][0], end_s, m, pc)
        if i + CHUNK_S >= len(probs):
            break
        i += STRIDE_S
    return (*peak, overall, frac_high)


def assess(audio_path, model_size="base"):
    """Transcribe -> per-second acoustic (m2) -> chunk-pool -> multiplicative-veto risk."""
    turns = transcribe(audio_path, model_size=model_size)
    windows = _get_scorer().score_windows(audio_path)
    cal = load_calibration(path=CAL_DATA, source=CAL_SOURCE)
    intent, intent_src = score_intent(turns) if turns else (0.0, "none")
    if windows:
        start_s, end_s, acoustic, p_clone, overall, frac_high = _pool(windows, cal)
    else:
        acoustic, overall, frac_high, start_s, end_s = 0.5, 0.5, 0.0, 0, 0
        _, p_clone = fuse(acoustic, 1.0, cal)
    risk, _ = fuse(acoustic, intent, cal)              # risk = P(clone) * intent
    level, advice = decide(risk)
    long_file = len(windows) > CHUNK_S
    return {"audio": audio_path, "detector": M2,
            "acoustic": round(acoustic, 3), "acoustic_overall": round(overall, 3),
            "p_clone": round(p_clone, 3), "intent": round(intent, 3),
            "intent_source": intent_src, "risk": round(risk, 3),
            "level": level, "advice": advice,
            "duration_s": len(windows), "frac_synthetic": round(frac_high, 2),
            "pooling": (f"max of {CHUNK_S}s chunks (stride {STRIDE_S}s)"
                        if long_file else "whole-file mean"),
            "peak_segment": ({"start": start_s, "end": end_s} if long_file else None),
            "caveat": CAVEAT, "transcript": turns}
