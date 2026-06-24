"""Local analysis of the degradation sweep -> the F2 curve.

For each (source, SNR level): build the paired per-call benchmark, run the iso-FAR
out-of-fold comparison over several seeds, and report detector quality (acoustic_only
AUC) alongside the multiplicative-vs-linear fusion gain (bayesian - parallel TPR@10%FAR
vs genuine_scam). The story: gain peaks where the detector is imperfect-but-informative.

Usage: stage2_degrade.py <degrade_events.json> [n_seeds]
"""
import sys, json, numpy as np
from voiceguard.intent import build_pools
from experiments.stage2_run import build_dataset
from experiments.stage2_roc import auc, tpr_at_far, oof_scores, FAR


def metrics_one_seed(clips, pools, seed):
    ds = build_dataset(clips, pools, seed=seed)
    a_mean, i_max = ds.acoustic.mean(1), ds.intent.max(1)
    pos, gscam = ds.positive, ds.call_type == "genuine_scam"
    y = np.where(pos, 1, np.where(~pos, 0, -1))
    folds = np.array_split(np.random.default_rng(seed).permutation(len(y)), 2)
    sc = oof_scores(a_mean, i_max, y, folds)
    aco_auc = auc(sc["acoustic_only"], y == 1, y == 0)
    bg = tpr_at_far(sc["bayesian"], y == 1, gscam, FAR)
    pg = tpr_at_far(sc["parallel"], y == 1, gscam, FAR)
    return aco_auc, bg, pg


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data/acoustic_degrade_m2.json"
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    data = json.load(open(path))
    pools = build_pools()
    levels = sorted({r["snr"] for r in data}, key=lambda x: (x == "clean", x if x != "clean" else 0), reverse=True)
    # order: clean first, then high->low SNR
    levels = (["clean"] if "clean" in levels else []) + sorted([l for l in levels if l != "clean"], reverse=True)
    print(f"{path}  ({n_seeds} seeds)\n")
    cols = f"{'source':>7} | {'SNR':>6} | {'acoustic_AUC':>13} | {'bayes TPR_gs':>13} | {'linear TPR_gs':>14} | {'gain(b-l)':>10}"
    print(cols); print("-" * len(cols))
    for src in ["asv", "itw"]:
        for lv in levels:
            clips = [r for r in data if r["source"] == src and r["snr"] == lv]
            if not clips:
                continue
            A, BG, PG = [], [], []
            for s in range(n_seeds):
                a, b, p = metrics_one_seed(clips, pools, s)
                A.append(a); BG.append(b); PG.append(p)
            A, BG, PG = np.array(A), np.array(BG), np.array(PG)
            gain = BG - PG
            print(f"{src:>7} | {str(lv):>6} | {A.mean():>13.3f} | {BG.mean():>13.3f} | "
                  f"{PG.mean():>14.3f} | {gain.mean():>+10.3f}")
        print("-" * len(cols))


if __name__ == "__main__":
    main()
