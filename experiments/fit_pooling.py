"""Fit the checker's center shift for long calls and dual-channel recordings.

The packaged center c for m2 (data/events_m2_600.json, In-the-Wild) is fit on clip means;
a genuine clip's mean exceeds c 9.8% of the time. On a real call the checker's statistic
is different: the most synthetic 30-second chunk (stride 15), and for dual-channel audio
the more synthetic of the two parties. A max over more chunks or channels exceeds c more
often. This simulates genuine and clone-scam calls from the per-second In-the-Wild
scores and finds, per (channels, speech seconds), the center shift that keeps a genuine
call's exceedance at the clip rate. Shifts are never negative.

Call model (an approximation; README "Long and dual-channel calls"): two speakers take
turns second by second, the caller speaking 60% of the time. Each speaker is one
In-the-Wild clip whose per-second scores are resampled, which keeps a speaker's level
steady through the call; real calls drift more. Mono puts both in one track, stereo one
per channel (scored only while that party talks, as the checker does).

  python -m experiments.fit_pooling        # table of FAR/TPR before and after the shift
"""
import hashlib, json, numpy as np
from voiceguard.assess import M2, CHUNK_S, STRIDE_S
from voiceguard.detect import load_calibration

DATA = "data/events_m2_600.json"
LENGTHS = [10, 30, 60, 120, 240, 480, 960]     # speech seconds in the call
CALLER_SHARE, N_SIM, SEED = 0.6, 2000, 0


def peak_mean(p):
    """Most synthetic chunk mean, exactly as voiceguard.assess._pool computes it."""
    p = np.asarray(p, float)
    starts = [0]
    while starts[-1] + CHUNK_S < len(p):
        starts.append(starts[-1] + STRIDE_S)
    starts = np.array(starts)
    ends = np.minimum(starts + CHUNK_S, len(p))
    csum = np.concatenate([[0.0], np.cumsum(p)])
    return float(np.max((csum[ends] - csum[starts]) / (ends - starts)))


def clip_scores(path=DATA):
    rows = [r for r in json.load(open(path)) if r["source"] == "itw"]
    per_clip = lambda clone: [np.array([p for _, p in r["events"]]) for r in rows if r["is_clone"] == clone]
    return per_clip(False), per_clip(True)


def call_stat(rng, caller, other, seconds, channels):
    talks = rng.uniform(size=seconds) < CALLER_SHARE
    a, b = rng.choice(caller, talks.sum()), rng.choice(other, (~talks).sum())
    if channels == 2:
        return max(peak_mean(x) for x in (a, b) if len(x))
    mix = np.empty(seconds)
    mix[talks], mix[~talks] = a, b
    return peak_mean(mix)


def fit(path=DATA):
    cal = load_calibration(model=M2, source="itw")
    genuine, clone = clip_scores(path)
    target = float(np.mean([g.mean() > cal["c"] for g in genuine]))
    rng = np.random.default_rng(SEED)
    pick = lambda pool: pool[rng.integers(len(pool))]
    shifts, report = {"1": {}, "2": {}}, []
    for channels in (1, 2):
        for seconds in LENGTHS:
            g = np.array([call_stat(rng, pick(genuine), pick(genuine), seconds, channels) for _ in range(N_SIM)])
            c = np.array([call_stat(rng, pick(clone), pick(genuine), seconds, channels) for _ in range(N_SIM)])
            shift = max(0.0, round(float(np.quantile(g, 1 - target)) - cal["c"], 4))
            shifts[str(channels)][str(seconds)] = shift
            report.append((channels, seconds, (g > cal["c"]).mean(), (c > cal["c"]).mean(), shift,
                           (g > cal["c"] + shift).mean(), (c > cal["c"] + shift).mean()))
    with open(path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    table = {"model": M2, "source": "itw", "data_path": path, "data_sha256": digest,
             "statistic": "center shift keeping a simulated genuine call's peak-chunk "
                          "exceedance of c at the genuine clip rate",
             "target_exceedance": round(target, 4), "caller_share": CALLER_SHARE,
             "n_sim": N_SIM, "seed": SEED, "shifts": shifts["1"]}   # the checker mixes to mono
    return table, report


def main():
    table, report = fit()
    print(f"target genuine exceedance (clip level): {table['target_exceedance']:.3f}")
    print("channels | speech s | genuine>c | clone>c | shift  | genuine>c+shift | clone>c+shift")
    for ch, s, g0, c0, d, g1, c1 in report:
        print(f"{ch:>8} | {s:>8} | {g0:>9.3f} | {c0:>7.3f} | {d:.4f} | {g1:>15.3f} | {c1:>13.3f}")
    print("packaged (mono) shifts:", json.dumps(table["shifts"]))


if __name__ == "__main__":
    main()
