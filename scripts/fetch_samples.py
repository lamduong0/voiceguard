"""Fetch one genuine + one clone clip from the ASVspoof 2019 LA HF mirror and save as WAV
for the real-audio smoke test. Audio is gitignored (do not redistribute dataset files).

  PYTHONPATH=. .venv/bin/python scripts/fetch_samples.py
"""
import io, os, soundfile as sf
from datasets import load_dataset, Audio

os.makedirs("data", exist_ok=True)
ds = load_dataset("Bisher/ASVspoof_2019_LA", split="train", streaming=True
                  ).cast_column("audio", Audio(decode=False))
got = {}
for ex in ds:
    key = "clone" if ex["system_id"].strip() != "-" else "genuine"
    if key in got:
        continue
    b = ex["audio"]["bytes"]
    if b is None:
        continue
    wav, sr = sf.read(io.BytesIO(b), dtype="float32")
    path = f"data/sample_{key}.wav"
    sf.write(path, wav, sr)
    got[key] = path
    print("saved", path, "sr", sr, "samples", len(wav))
    if len(got) == 2:
        break
print("done:", got)
