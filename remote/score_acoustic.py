"""Remote (GPU) acoustic scoring for VoiceGuard Stage 2.

For each clip from In-the-Wild and ASVspoof 2019 LA, chunk into 1s windows and score
each with a pretrained anti-spoofing model -> per-window P(synthetic). This is the
*acoustic signal* the fusion controllers consume; intent is added later off-GPU.

Output: acoustic_events.json = [{source, is_clone, duration, events:[[t_s, p_fake]...]}].
Also prints mean P(fake) per class as a sanity check that the model separates the data.
"""
import os, io, json, random, argparse, csv, statistics as st
import numpy as np, soundfile as sf, librosa, torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from datasets import load_dataset, Audio

SR = 16000
WIN = 16000          # 1s window
MAX_WIN = 30         # cap windows/clip so long clips don't dominate compute
DEFAULT_MID = "MelodyMachine/Deepfake-audio-detection-V2"


def load_model(mid):
    fe = AutoFeatureExtractor.from_pretrained(mid)
    m = AutoModelForAudioClassification.from_pretrained(mid).to("cuda").eval()
    fake_lbls = [i for i, l in m.config.id2label.items()
                 if str(l).lower() in ("fake", "spoof", "spoofed", "fake_audio", "ai", "synthetic")]
    fake_idx = fake_lbls[0] if fake_lbls else 0
    print("MODEL", mid, "id2label", m.config.id2label, "fake_idx", fake_idx, flush=True)
    return fe, m, fake_idx


@torch.no_grad()
def score_windows(fe, m, fake_idx, wav):
    wins, times, i = [], [], 0
    while i < len(wav) and len(wins) < MAX_WIN:
        chunk = wav[i:i + WIN]
        if len(chunk) < WIN // 2:
            break
        wins.append(chunk); times.append(i / SR); i += WIN
    if not wins:
        wins, times = [wav], [0.0]
    probs = []
    for s in range(0, len(wins), 32):
        batch = wins[s:s + 32]
        inp = fe(batch, sampling_rate=SR, return_tensors="pt", padding=True)
        inp = {k: v.to("cuda") for k, v in inp.items()}
        p = torch.softmax(m(**inp).logits, dim=-1)[:, fake_idx].cpu().numpy()
        probs.extend(p.tolist())
    return list(zip(times, probs))


def load_wav(path=None, data_bytes=None):
    src = io.BytesIO(data_bytes) if data_bytes is not None else path
    wav, sr = sf.read(src, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SR:
        wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=SR)
    return wav.astype(np.float32)


def itw_clips(n_per, seed=0):
    base = os.path.expanduser("~/data/release_in_the_wild")
    rows = list(csv.DictReader(open(os.path.join(base, "meta.csv"))))
    rng = random.Random(seed)
    bona = [r for r in rows if r["label"].strip() == "bona-fide"]
    spoof = [r for r in rows if r["label"].strip() == "spoof"]
    rng.shuffle(bona); rng.shuffle(spoof)
    out = [("itw", False, os.path.join(base, r["file"])) for r in bona[:n_per]]
    out += [("itw", True, os.path.join(base, r["file"])) for r in spoof[:n_per]]
    return out


def asv_clips(n_per):
    ds = load_dataset("Bisher/ASVspoof_2019_LA", split="train", streaming=True
                      ).cast_column("audio", Audio(decode=False))
    bona, spoof = [], []
    for ex in ds:
        is_spoof = ex["system_id"].strip() != "-"
        b = ex["audio"]["bytes"]
        if b is None:
            continue
        if is_spoof and len(spoof) < n_per:
            spoof.append(b)
        elif not is_spoof and len(bona) < n_per:
            bona.append(b)
        if len(bona) >= n_per and len(spoof) >= n_per:
            break
    return [("asv", False, b) for b in bona] + [("asv", True, b) for b in spoof]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--out", default="acoustic_events.json")
    ap.add_argument("--model", default=DEFAULT_MID)
    a = ap.parse_args()
    fe, m, fake_idx = load_model(a.model)
    clips = itw_clips(a.n) + asv_clips(a.n)
    print("total clips", len(clips), flush=True)
    results = []
    for j, (src, is_clone, ref) in enumerate(clips):
        try:
            wav = load_wav(data_bytes=ref) if src == "asv" else load_wav(path=ref)
            ev = score_windows(fe, m, fake_idx, wav)
            results.append({"source": src, "is_clone": bool(is_clone),
                            "duration": len(wav) / SR,
                            "events": [[float(t), float(p)] for t, p in ev]})
        except Exception as e:
            print("SKIP", src, j, str(e)[:80], flush=True)
        if j % 100 == 0:
            print("progress", j, "/", len(clips), flush=True)
    json.dump(results, open(a.out, "w"))
    print("WROTE", a.out, "n=", len(results))
    for src in ["itw", "asv"]:
        for cl in [False, True]:
            ps = [p for r in results if r["source"] == src and r["is_clone"] == cl for _, p in r["events"]]
            nc = sum(1 for r in results if r["source"] == src and r["is_clone"] == cl)
            if ps:
                print("MEAN_PFAKE", src, "clone" if cl else "genuine",
                      round(st.mean(ps), 3), "n_clips", nc, flush=True)


if __name__ == "__main__":
    main()
