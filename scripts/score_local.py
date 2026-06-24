"""Local (CPU) acoustic scoring on ASVspoof 2019 LA via the HF mirror — no GPU needed.
Used to add a 3rd detector for the claim-2 robustness check.

  PYTHONPATH=. .venv/bin/python scripts/score_local.py --model <hf_id> --out data/x.json --n 300
"""
import io, json, argparse, statistics as st
import soundfile as sf, librosa, numpy as np, torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from datasets import load_dataset, Audio

SR, WIN, MAXW = 16000, 16000, 30


def load_model(mid):
    fe = AutoFeatureExtractor.from_pretrained(mid)
    m = AutoModelForAudioClassification.from_pretrained(mid).eval()
    fl = [i for i, l in m.config.id2label.items()
          if str(l).lower() in ("fake", "spoof", "spoofed", "fake_audio", "ai", "synthetic")]
    fi = fl[0] if fl else 0
    print("MODEL", mid, m.config.id2label, "fake_idx", fi, flush=True)
    return fe, m, fi


@torch.no_grad()
def score(fe, m, fi, wav):
    wins, times, i = [], [], 0
    while i < len(wav) and len(wins) < MAXW:
        ch = wav[i:i + WIN]
        if len(ch) < WIN // 2:
            break
        wins.append(ch); times.append(i / SR); i += WIN
    if not wins:
        wins, times = [wav], [0.0]
    probs = []
    for s in range(0, len(wins), 16):
        inp = fe(wins[s:s + 16], sampling_rate=SR, return_tensors="pt", padding=True)
        probs.extend(torch.softmax(m(**inp).logits, -1)[:, fi].numpy().tolist())
    return list(zip(times, probs))


def degrade(wav, snr_db, rng):
    if snr_db is None:
        return wav
    p_sig = float(np.mean(wav ** 2)) + 1e-12
    noise = rng.normal(0, np.sqrt(p_sig / (10 ** (snr_db / 10))), len(wav)).astype(np.float32)
    return np.clip(wav + noise, -1, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--levels", default="clean", help="comma list, e.g. clean,15,10,5")
    a = ap.parse_args()
    levels = [None if x == "clean" else float(x) for x in a.levels.split(",")]
    rng = np.random.default_rng(0)
    fe, m, fi = load_model(a.model)
    ds = load_dataset("Bisher/ASVspoof_2019_LA", split="train", streaming=True
                      ).cast_column("audio", Audio(decode=False))
    bona, spoof = [], []
    for ex in ds:
        b = ex["audio"]["bytes"]
        if b is None:
            continue
        if ex["system_id"].strip() != "-" and len(spoof) < a.n:
            spoof.append(b)
        elif ex["system_id"].strip() == "-" and len(bona) < a.n:
            bona.append(b)
        if len(bona) >= a.n and len(spoof) >= a.n:
            break
    clips = [(False, b) for b in bona] + [(True, b) for b in spoof]
    out = []
    for j, (is_clone, b) in enumerate(clips):
        wav, sr = sf.read(io.BytesIO(b), dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(1)
        if sr != SR:
            wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=SR)
        wav = wav.astype(np.float32)
        for lv in levels:
            ev = score(fe, m, fi, degrade(wav, lv, rng))
            out.append({"source": "asv", "is_clone": bool(is_clone),
                        "snr": ("clean" if lv is None else lv), "duration": len(wav) / SR,
                        "events": [[float(t), float(p)] for t, p in ev]})
        if j % 100 == 0:
            print("progress", j, "/", len(clips), flush=True)
    json.dump(out, open(a.out, "w"))
    print("WROTE", a.out, len(out), flush=True)
    for lv in levels:
        key = "clean" if lv is None else lv
        for cl in [False, True]:
            ps = [p for r in out if r["snr"] == key and r["is_clone"] == cl for _, p in r["events"]]
            if ps:
                print("MEAN_PFAKE", "snr", key, "clone" if cl else "genuine",
                      round(st.mean(ps), 3), flush=True)
    print("LOCAL_SCORE_DONE", flush=True)


if __name__ == "__main__":
    main()
