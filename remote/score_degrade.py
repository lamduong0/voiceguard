"""Remote (GPU) acoustic scoring across a DEGRADATION sweep.

Scores each clip at several additive-noise SNR levels, so locally we can plot the
multiplicative-vs-linear fusion gain against detector quality (acoustic AUC). As audio
degrades, the acoustic detector goes from strong -> chance; the hypothesis is that the
multiplicative veto's advantage peaks in the imperfect-but-informative middle.

Output: list of {source, is_clone, snr, duration, events:[[t,p_fake]...]}.
"""
import os, io, json, random, argparse, csv, statistics as st
import numpy as np, soundfile as sf, librosa, torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from datasets import load_dataset, Audio

SR, WIN, MAX_WIN = 16000, 16000, 30


def load_model(mid):
    fe = AutoFeatureExtractor.from_pretrained(mid)
    m = AutoModelForAudioClassification.from_pretrained(mid).to("cuda").eval()
    fl = [i for i, l in m.config.id2label.items()
          if str(l).lower() in ("fake", "spoof", "spoofed", "fake_audio", "ai", "synthetic")]
    fi = fl[0] if fl else 0
    print("MODEL", mid, m.config.id2label, "fake_idx", fi, flush=True)
    return fe, m, fi


@torch.no_grad()
def score_windows(fe, m, fi, wav):
    wins, times, i = [], [], 0
    while i < len(wav) and len(wins) < MAX_WIN:
        c = wav[i:i + WIN]
        if len(c) < WIN // 2:
            break
        wins.append(c); times.append(i / SR); i += WIN
    if not wins:
        wins, times = [wav], [0.0]
    probs = []
    for s in range(0, len(wins), 32):
        inp = fe(wins[s:s + 32], sampling_rate=SR, return_tensors="pt", padding=True)
        inp = {k: v.to("cuda") for k, v in inp.items()}
        probs.extend(torch.softmax(m(**inp).logits, dim=-1)[:, fi].cpu().numpy().tolist())
    return list(zip(times, probs))


def load_wav(path=None, data_bytes=None):
    w, sr = sf.read(io.BytesIO(data_bytes) if data_bytes is not None else path, dtype="float32")
    if w.ndim > 1:
        w = w.mean(axis=1)
    if sr != SR:
        w = librosa.resample(w.astype(np.float32), orig_sr=sr, target_sr=SR)
    return w.astype(np.float32)


def degrade(wav, snr_db, rng):
    if snr_db is None:
        return wav
    p_sig = float(np.mean(wav ** 2)) + 1e-12
    p_noise = p_sig / (10 ** (snr_db / 10))
    noise = rng.normal(0, np.sqrt(p_noise), len(wav)).astype(np.float32)
    return np.clip(wav + noise, -1, 1)


def itw_clips(n_per, seed=0):
    base = os.path.expanduser("~/data/release_in_the_wild")
    rows = list(csv.DictReader(open(os.path.join(base, "meta.csv"))))
    rng = random.Random(seed)
    bona = [r for r in rows if r["label"].strip() == "bona-fide"]
    spoof = [r for r in rows if r["label"].strip() == "spoof"]
    rng.shuffle(bona); rng.shuffle(spoof)
    return ([("itw", False, os.path.join(base, r["file"])) for r in bona[:n_per]] +
            [("itw", True, os.path.join(base, r["file"])) for r in spoof[:n_per]])


def asv_clips(n_per):
    ds = load_dataset("Bisher/ASVspoof_2019_LA", split="train", streaming=True
                      ).cast_column("audio", Audio(decode=False))
    bona, spoof = [], []
    for ex in ds:
        b = ex["audio"]["bytes"]
        if b is None:
            continue
        if ex["system_id"].strip() != "-" and len(spoof) < n_per:
            spoof.append(b)
        elif ex["system_id"].strip() == "-" and len(bona) < n_per:
            bona.append(b)
        if len(bona) >= n_per and len(spoof) >= n_per:
            break
    return [("asv", False, b) for b in bona] + [("asv", True, b) for b in spoof]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--levels", default="clean,20,15,10,5,0")
    a = ap.parse_args()
    levels = [None if x == "clean" else float(x) for x in a.levels.split(",")]
    fe, m, fi = load_model(a.model)
    clips = itw_clips(a.n) + asv_clips(a.n)
    print("clips", len(clips), "levels", a.levels, flush=True)
    rng = np.random.default_rng(0)
    out = []
    for j, (src, is_clone, ref) in enumerate(clips):
        try:
            wav = load_wav(data_bytes=ref) if src == "asv" else load_wav(path=ref)
        except Exception as e:
            print("SKIP", src, j, str(e)[:60], flush=True); continue
        for lv in levels:
            ev = score_windows(fe, m, fi, degrade(wav, lv, rng))
            out.append({"source": src, "is_clone": bool(is_clone),
                        "snr": ("clean" if lv is None else lv), "duration": len(wav) / SR,
                        "events": [[float(t), float(p)] for t, p in ev]})
        if j % 100 == 0:
            print("progress", j, "/", len(clips), flush=True)
    json.dump(out, open(a.out, "w"))
    print("WROTE", a.out, "records", len(out))
    for lv in levels:
        key = "clean" if lv is None else lv
        for cl in [False, True]:
            ps = [p for r in out if r["snr"] == key and r["is_clone"] == cl for _, p in r["events"]]
            if ps:
                print("MEAN_PFAKE", "snr", key, "clone" if cl else "genuine", round(st.mean(ps), 3), flush=True)
    print("DEGRADE_DONE", flush=True)


if __name__ == "__main__":
    main()
