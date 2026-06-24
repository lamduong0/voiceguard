"""Stage 2 real-data result: parallel vs bayesian on real acoustic (deepfake model on
ASVspoof / In-the-Wild) paired with real intent (LLM-scored transcripts).

Pairing recreates the four Stage 1 archetypes from real signals:
  clone  -> clone_scam (p=0.85) | clone_benign
  genuine-> genuine_benign (p=0.66) | genuine_urgent (hard negative)
Positive = clone_scam. Same tune/metrics/controllers as Stage 1; tuned to equal
hard-negative FAR. Reported per source.
"""
import json, random, numpy as np
from voiceguard.simulate import Dataset
from voiceguard.evaluate import tune, evaluate, running_mean
from voiceguard.realsignals import align_to_timeline
from experiments.stage1_mechanism import METHODS, FAR_BUDGET

T = 30
CT = {("scam", True): "clone_scam", ("benign", True): "clone_benign",
      ("benign", False): "genuine_benign", ("urgent", False): "genuine_urgent",
      ("scam", False): "genuine_scam"}


def build_dataset(clips, pools, seed=0, intent_noise=0.05):
    rng = random.Random(seed)
    nprng = np.random.default_rng(seed)
    n = len(clips)
    acoustic = np.empty((n, T))
    intent = np.empty((n, T))
    is_clone, is_scam, call_type = [], [], []
    for i, c in enumerate(clips):
        dur = max(float(c["duration"]), 1.0)
        acoustic[i] = align_to_timeline([(t, p) for t, p in c["events"]], T, dur, fill=0.5)
        clone = bool(c["is_clone"])
        if clone:
            arch = "scam" if rng.random() < 0.7 else "benign"
        else:
            r = rng.random()
            arch = "benign" if r < 0.5 else ("urgent" if r < 0.75 else "scam")
        dia = rng.choice(pools[arch])
        iv = align_to_timeline([(f * dur, v) for f, v in dia], T, dur, fill=0.05)
        intent[i] = np.clip(iv + nprng.normal(0, intent_noise, T), 0, 1)
        is_clone.append(clone); is_scam.append(arch == "scam"); call_type.append(CT[(arch, clone)])
    return Dataset(acoustic=acoustic, intent=intent, is_clone=np.array(is_clone),
                   is_scam=np.array(is_scam), call_type=np.array(call_type))


def run(source, pools):
    data = [c for c in json.load(open("data/acoustic_events.json")) if c["source"] == source]
    rng = random.Random(0); rng.shuffle(data)
    half = len(data) // 2
    train = build_dataset(data[:half], pools, seed=1)
    test = build_dataset(data[half:], pools, seed=2)
    acc = running_mean(test.acoustic)
    npos = int(test.positive.sum()); nhard = int((test.call_type == "genuine_urgent").sum())
    print(f"\n### source={source}  test: {len(data)-half} calls, {npos} clone_scam pos, {nhard} hard-neg")
    cols = f"{'method':>9} | {'det_rate':>8} | {'FAR':>5} | {'FAR_hard':>8} | {'median_TTD':>10}"
    print(cols); print("-" * len(cols))
    for name, fire_fn, grid in METHODS:
        best = tune(fire_fn, grid, train, FAR_BUDGET, constraint_key="far_hard")
        if not best:
            print(f"{name:>9} | (no feasible config)"); continue
        params, _ = best
        m = evaluate(fire_fn(acc, test.intent, **params), test)
        print(f"{name:>9} | {m['det_rate']:>8.3f} | {m['far']:>5.3f} | {m['far_hard']:>8.3f} | {m['median_ttd']:>10.1f}")
    print("-" * len(cols))


def main():
    from voiceguard.intent import build_pools
    pools = build_pools()
    print("intent pools:", {k: len(v) for k, v in pools.items()})
    for src in ["asv", "itw"]:
        run(src, pools)


if __name__ == "__main__":
    main()
