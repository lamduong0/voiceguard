"""Score In-the-Wild clips through common call-recording codecs with the m2 detector, so
the checker can use a codec-matched calibration: lossy audio raises the synthetic-voice
score (README), and a calibration fit on uncompressed clips then over-alarms.

Fetches only the selected clips from the dataset zip with HTTP range requests (no 8 GB
download). The selection mirrors remote/score_acoustic.py's itw_clips (meta.csv,
Random(0) shuffle, first N per class), so the "pcm" rows re-score the clips behind
data/events_m2_600.json. Every variant goes through the checker's own decode path
(voiceguard.detect.load_audio) and scorer, capped at 30 one-second windows like the
calibration data.

  .venv/bin/python scripts/score_codecs.py --n 600 --out data/codec_m2_itw.json
"""
import argparse, csv, io, json, os, random, struct, tempfile, time, urllib.request, zipfile, zlib
from concurrent.futures import ThreadPoolExecutor
import numpy as np, soundfile as sf, librosa, av
from voiceguard.detect import load_audio, AcousticScorer, SAMPLE_RATE
from voiceguard.assess import M2

URL = "https://huggingface.co/datasets/mueller91/In-The-Wild/resolve/main/release_in_the_wild.zip"
ZIP_SIZE = 8161489023
MAX_WIN = 30
CODECS = {  # name: (container, encoder, sample rate, bit rate)
    "aac": ("mp4", "aac", 16000, 64000),                  # voice-memo / call-recorder apps
    "mp3": ("mp3", "libmp3lame", 16000, 64000),           # call-recorder apps
    "opus": ("ogg", "libopus", 16000, 24000),             # VoIP / messaging apps
    "amr_nb": ("amr", "libopencore_amrnb", 8000, 12200),  # narrowband phone network
}


def fetch(start, end):
    req = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{end}"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except OSError:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


class RangeFile(io.RawIOBase):
    """Just enough of a seekable file for zipfile to read the central directory."""

    def __init__(self):
        self.pos = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        self.pos = {0: offset, 1: self.pos + offset, 2: ZIP_SIZE + offset}[whence]
        return self.pos

    def tell(self):
        return self.pos

    def readinto(self, buf):
        if self.pos >= ZIP_SIZE or not len(buf):
            return 0
        data = fetch(self.pos, min(self.pos + len(buf), ZIP_SIZE) - 1)
        buf[:len(data)] = data
        self.pos += len(data)
        return len(data)


def member(info):
    """One range request per member: local header + data (extra field guessed <= 256 B)."""
    name_len = len(info.filename.encode())
    blob = fetch(info.header_offset, info.header_offset + 30 + name_len + 256 + info.compress_size)
    n, m = struct.unpack("<HH", blob[26:30])
    raw = blob[30 + n + m: 30 + n + m + info.compress_size]
    return raw if info.compress_type == zipfile.ZIP_STORED else zlib.decompress(raw, -15)


def itw_selection(zf, n_per, seed=0):
    rows = list(csv.DictReader(io.TextIOWrapper(io.BytesIO(member(zf.getinfo("release_in_the_wild/meta.csv"))))))
    rng = random.Random(seed)
    bona = [r for r in rows if r["label"].strip() == "bona-fide"]
    spoof = [r for r in rows if r["label"].strip() == "spoof"]
    rng.shuffle(bona); rng.shuffle(spoof)
    return [(False, r["file"]) for r in bona[:n_per]] + [(True, r["file"]) for r in spoof[:n_per]]


def encode(wav, sr, codec, path):
    fmt, encoder, rate, bit_rate = CODECS[codec]
    if sr != rate:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=rate)
    pcm = (np.clip(wav, -1, 1) * 32767).astype(np.int16)[None, :]
    with av.open(path, "w", format=fmt) as out:
        stream = out.add_stream(encoder, rate=rate, layout="mono")
        stream.bit_rate = bit_rate
        frame = av.AudioFrame.from_ndarray(pcm, format="s16", layout="mono")
        frame.rate = rate
        for packet in list(stream.encode(frame)) + list(stream.encode(None)):
            out.mux(packet)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--out", default="data/codec_m2_itw.json")
    ap.add_argument("--codecs", default="pcm," + ",".join(CODECS))
    ap.add_argument("--device", default="cpu", help="torch device for the detector, e.g. mps or cuda")
    a = ap.parse_args()
    codecs = a.codecs.split(",")
    zf = zipfile.ZipFile(RangeFile())
    clips = itw_selection(zf, a.n)
    infos = [zf.getinfo(f"release_in_the_wild/{f}") for _, f in clips]
    print("clips", len(clips), "codecs", codecs, flush=True)
    scorer = AcousticScorer(model=M2)
    if a.device != "cpu":
        fe, scorer.m = scorer.fe, scorer.m.to(a.device)
        scorer.fe = lambda chunks, **kw: fe(chunks, **kw).to(a.device)
    out, t0 = [], time.time()
    with tempfile.TemporaryDirectory() as tmp, ThreadPoolExecutor(8) as pool:
        for j, ((is_clone, name), raw) in enumerate(zip(clips, pool.map(member, infos))):
            wav, sr = sf.read(io.BytesIO(raw), dtype="float32")
            wav = (wav.mean(1) if wav.ndim > 1 else wav)[: MAX_WIN * sr]
            src = os.path.join(tmp, "clip.wav")
            sf.write(src, wav, sr, subtype="PCM_16")
            for codec in codecs:
                path = src if codec == "pcm" else os.path.join(tmp, f"clip.{CODECS[codec][0]}")
                if codec != "pcm":
                    encode(wav, sr, codec, path)
                audio = load_audio(path)[: MAX_WIN * SAMPLE_RATE]
                events = scorer.score_windows(audio)
                out.append({"source": "itw", "codec": codec, "is_clone": is_clone, "file": name,
                            "duration": len(wav) / sr, "events": [[float(t), float(p)] for t, p in events]})
            if j % 50 == 0:
                print(f"progress {j}/{len(clips)}  {time.time() - t0:.0f}s", flush=True)
    with open(a.out, "w") as f:
        json.dump(out, f)
    print("WROTE", a.out, len(out))
    for codec in codecs:
        for cl in (False, True):
            means = [np.mean([p for _, p in r["events"]]) for r in out
                     if r["codec"] == codec and r["is_clone"] == cl and r["events"]]
            print(f"MEAN {codec:>7} {'clone' if cl else 'genuine':>8} {np.mean(means):.3f}  n={len(means)}")


if __name__ == "__main__":
    main()
