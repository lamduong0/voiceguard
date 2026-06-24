"""VoiceGuard demo core: acoustic + intent -> multiplicative-veto risk -> intervention.

The fusion + intervention (the novel part) runs with zero heavy deps. The acoustic scorer
uses the real pretrained model when torch/transformers + an audio file are available (lazy
import); the bundled demo uses real precomputed per-window scores so it runs offline.
"""
import os, json, math, re, statistics as st

DEFAULT_MODEL = "MelodyMachine/Deepfake-audio-detection-V2"


def load_calibration(path="data/acoustic_events_m1.json", source="asv"):
    """Center c and gain g for the acoustic veto, calibrated on real data from the regime
    where the detector works (ASVspoof). Falls back to sane defaults if data is absent."""
    try:
        data = [r for r in json.load(open(path)) if r["source"] == source]
        mean_a = lambda r: sum(p for _, p in r["events"]) / len(r["events"])
        mg = st.mean([mean_a(r) for r in data if not r["is_clone"]])
        mc = st.mean([mean_a(r) for r in data if r["is_clone"]])
        c = (mg + mc) / 2
        g = math.log(4) / max(mc - c, 1e-3)        # maps clone mean ~0.8, genuine ~0.2
        return {"c": round(c, 4), "g": round(g, 3), "mean_genuine": round(mg, 3),
                "mean_clone": round(mc, 3)}
    except Exception:
        return {"c": 0.21, "g": 7.3, "mean_genuine": 0.03, "mean_clone": 0.40}


def fuse(acoustic, intent, cal):
    """Multiplicative acoustic veto: risk = P(clone) * intent."""
    p_clone = 1 / (1 + math.exp(-cal["g"] * (acoustic - cal["c"])))
    return p_clone * intent, p_clone


SCAM_CUES = [r"gift card", r"wire", r"\bbail\b", r"\barrested\b", r"urgent", r"right now",
             r"don'?t tell", r"do not tell", r"secret", r"transfer", r"bitcoin", r"crypto",
             r"social security", r"\bIRS\b", r"verify your account", r"send money",
             r"western union", r"\bgift\b", r"\bwon\b(?!['’])", r"\bfee\b"]  # \bwon\b excludes "won't"


def keyword_intent(turns):
    best = 0.0
    for t in turns:
        txt = (t["text"] if isinstance(t, dict) else t).lower()
        hits = sum(1 for c in SCAM_CUES if re.search(c, txt))
        best = max(best, min(1.0, 0.25 * hits + (0.4 if hits else 0.05)))
    return best


def llm_intent(turns):
    key = os.environ.get("VG_LLM_KEY")
    if not key:
        return None
    try:
        from openai import OpenAI
        cl = OpenAI(api_key=key, base_url=os.environ.get("VG_LLM_BASE_URL") or None)
        text = "\n".join((t["text"] if isinstance(t, dict) else t) for t in turns)
        r = cl.chat.completions.create(
            model=os.environ.get("VG_LLM_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content":
                       "Rate 0..1 how strongly this call transcript indicates a scam/fraud "
                       "attempt (money/gift-card/wire request, urgency, secrecy, "
                       "impersonation). Reply with ONLY the number.\n\n" + text}],
            max_tokens=8, temperature=0)
        m = re.search(r"[01](?:\.\d+)?", r.choices[0].message.content)
        return float(m.group(0)) if m else None
    except Exception:
        return None


def score_intent(turns):
    v = llm_intent(turns)
    return (v, "llm") if v is not None else (keyword_intent(turns), "keyword")


def decide(risk):
    if risk >= 0.60:
        return ("HIGH", "Likely AI-cloned voice in a scam. Issue the safe-word challenge; "
                        "do not send money or codes.")
    if risk >= 0.30:
        return ("MEDIUM", "Suspicious. Hang up and call the person back on a known number.")
    return ("LOW", "No strong clone+scam signal. Proceed with normal caution.")


def transcribe(audio_path, model_size="base"):
    """ASR an audio file into timestamped turns via faster-whisper (CPU). Returns
    [{"t": start_s, "text": ...}]; pass these as the intent transcript."""
    import librosa
    from faster_whisper import WhisperModel
    wav, _ = librosa.load(audio_path, sr=16000, mono=True)
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(wav)   # auto-detect language (don't force English)
    return [{"t": float(s.start), "text": s.text.strip()} for s in segments if s.text.strip()]


class AcousticScorer:
    """Real pretrained anti-spoofing model; per-1s-window P(synthetic), averaged."""

    def __init__(self, model=DEFAULT_MODEL):
        import torch
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
        self.torch = torch
        self.fe = AutoFeatureExtractor.from_pretrained(model)
        self.m = AutoModelForAudioClassification.from_pretrained(model).eval()
        self.fake_idx = [i for i, l in self.m.config.id2label.items()
                         if str(l).lower() in ("fake", "spoof", "synthetic")][0]

    def score(self, wav_path):
        import soundfile as sf, librosa, numpy as np
        wav, sr = sf.read(wav_path, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(1)
        if sr != 16000:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        probs = []
        for i in range(0, len(wav), 16000):
            ch = wav[i:i + 16000]
            if len(ch) < 8000:
                break
            inp = self.fe(ch, sampling_rate=16000, return_tensors="pt")
            with self.torch.no_grad():
                probs.append(self.torch.softmax(self.m(**inp).logits, -1)[0, self.fake_idx].item())
        return sum(probs) / len(probs) if probs else 0.5
