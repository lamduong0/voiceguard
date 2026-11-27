"""Shared recording assessment for check.py and app.py.

The m2 detector uses packaged score scaling from In-the-Wild clips. Long calls
use the largest mean over overlapping 30-second chunks. This pooling rule and
the alert thresholds still need validation on paired telephone recordings.
"""
from voiceguard.detect import (AcousticScorer, transcribe, load_audio, load_calibration,
                               fuse, decide, score_intent, SAMPLE_RATE, MIN_SAMPLES)

M2 = "Gustking/wav2vec2-large-xlsr-deepfake-audio-classification"
CHUNK_S, STRIDE_S = 30, 15
CAVEAT = ("Experimental voice-clone scam scores. Probabilities and alert thresholds "
          "need validation on telephone calls, including long recordings. "
          "Review scam intent separately when verifying a suspicious request.")

_scorer = None


def _get_scorer():
    """Lazy, cached — the web app reuses one loaded model across requests."""
    global _scorer
    if _scorer is None:
        _scorer = AcousticScorer(model=M2)
    return _scorer


def _pool(windows, cal):
    """Return peak bounds, peak scores, overall mean, and fraction above 0.5."""
    probs = [p for _, p in windows]
    overall = sum(probs) / len(probs)
    frac_high = sum(1 for p in probs if p > 0.5) / len(probs)
    peak, i = None, 0
    while i < len(probs):
        seg = probs[i:i + CHUNK_S]
        m = sum(seg) / len(seg)
        _, pc = fuse(m, 1.0, cal)
        end_s = windows[min(i + CHUNK_S, len(windows)) - 1][0] + 1
        if peak is None or m > peak[2]:
            peak = (windows[i][0], end_s, m, pc)
        if i + CHUNK_S >= len(probs):
            break
        i += STRIDE_S
    return (*peak, overall, frac_high)


def speech_windows(windows, turns):
    """Keep the one-second windows that overlap an ASR speech segment. The detector was
    calibrated on speech; scoring silence and line noise only adds noise. Without segment
    ends (no ASR, or transcripts from older versions) every window is kept."""
    spans = [(t["t"], t["end"]) for t in turns if "end" in t]
    if not spans:
        return windows
    kept = [(s, p) for s, p in windows if any(a < s + 1 and s < b for a, b in spans)]
    return kept or windows


def assess(audio_path, model_size="base"):
    """Decode once and combine available acoustic and transcript evidence."""
    cal = load_calibration(model=M2, source="itw")
    level, advice = decide(None)
    result = {
        "audio": str(audio_path), "detector": M2, "calibration": cal,
        "acoustic": None, "acoustic_overall": None, "clone_score": None,
        "p_clone": None,  # Compatibility alias for clone_score.
        "intent": None, "intent_source": "none", "risk": None,
        "level": level, "advice": advice, "reasons": [],
        "duration_s": None, "frac_synthetic": None,
        "pooling": "whole-file mean", "peak_segment": None,
        "caveat": CAVEAT, "transcript": [],
    }
    try:
        audio = load_audio(audio_path, mono=False)   # channels kept for per-speaker scoring
    except (ValueError, RuntimeError, OSError, EOFError) as error:
        result["reasons"].append(f"Audio could not be decoded ({error}).")
        return result
    waveform = audio if audio.ndim == 1 else audio.mean(0)   # ASR hears the mix
    result["duration_s"] = len(waveform) / SAMPLE_RATE
    if len(waveform) < MIN_SAMPLES:
        result["reasons"].append("Audio is shorter than the detector's 0.5-second minimum.")
        return result

    try:
        result["transcript"] = transcribe(waveform, model_size=model_size)
    except (ValueError, RuntimeError, OSError) as error:
        result["reasons"].append(f"Transcription failed ({error}).")
    intent, intent_src = score_intent(result["transcript"])
    result["intent"] = round(intent, 3) if intent is not None else None
    result["intent_source"] = intent_src
    if intent is None:
        result["reasons"].append("No usable speech transcript is available.")

    windows = []
    try:
        windows = speech_windows(_get_scorer().score_windows(audio), result["transcript"])
    except (ValueError, RuntimeError, OSError) as error:
        result["reasons"].append(f"Acoustic scoring failed ({error}).")
    if not windows:
        result["reasons"].append("No usable acoustic windows are available.")
        return result

    start, end, acoustic, clone_score, overall, frac_high = _pool(windows, cal)
    risk, _ = fuse(acoustic, intent, cal)
    result.update(
        acoustic=round(acoustic, 3), acoustic_overall=round(overall, 3),
        clone_score=round(clone_score, 3), p_clone=round(clone_score, 3),
        risk=round(risk, 3) if risk is not None else None,
        frac_synthetic=round(frac_high, 2),
    )
    result["level"], result["advice"] = decide(risk)
    if len(windows) > CHUNK_S:
        result["pooling"] = f"max of {CHUNK_S}s chunks (stride {STRIDE_S}s)"
        result["peak_segment"] = {"start": start, "end": min(end, result["duration_s"])}
    return result
