"""Harden claim 2 (Bayesian/multiplicative > linear fusion on high-intent genuine calls).

Runs the iso-FAR per-call comparison across MULTIPLE pairing seeds (separating
pairing randomness from bootstrap noise) for a given acoustic-events file, so we can
report robustness across seeds and across acoustic models. Headline metric:
TPR@10%FAR against the genuine_scam hard negative, parallel vs bayesian.

Usage: stage2_harden.py <events.json> [n_seeds]
"""
import sys, json, numpy as np
from voiceguard.intent import build_pools
from experiments.stage2_run import build_dataset
from experiments.stage2_roc import auc, tpr_at_far, oof_scores, FAR

METHODS = ["acoustic_only", "intent_only", "parallel", "bayesian"]


def one_seed(clips, pools, seed):
    ds = build_dataset(clips, pools, seed=seed)
    a_mean, i_max = ds.acoustic.mean(1), ds.intent.max(1)
    pos, gscam = ds.positive, ds.call_type == "genuine_scam"
    y = np.where(pos, 1, np.where(~pos, 0, -1))
    folds = np.array_split(np.random.default_rng(seed).permutation(len(y)), 2)
    sc = oof_scores(a_mean, i_max, y, folds)
    out = {}
    for k, s in sc.items():
        out[k] = (auc(s, y == 1, y == 0),
                  tpr_at_far(s, y == 1, gscam, FAR),
                  tpr_at_far(s, y == 1, y == 0, FAR))
    return out


def run(events_path, source, pools, n_seeds):
    clips = [c for c in json.load(open(events_path)) if c["source"] == source]
    if not clips:
        return
    rows = {k: {"auc": [], "tpr_gs": [], "tpr_all": []} for k in METHODS}
    for s in range(n_seeds):
        r = one_seed(clips, pools, s)
        for k in METHODS:
            rows[k]["auc"].append(r[k][0])
            rows[k]["tpr_gs"].append(r[k][1])
            rows[k]["tpr_all"].append(r[k][2])
    print(f"\n### {events_path}  source={source}  ({n_seeds} pairing seeds, mean +/- std)")
    cols = f"{'method':>14} | {'AUC':>15} | {'TPR@FAR_genuine_scam':>22} | {'TPR@FAR_all':>15}"
    print(cols); print("-" * len(cols))
    for k in METHODS:
        a = np.array(rows[k]["auc"]); g = np.array(rows[k]["tpr_gs"]); al = np.array(rows[k]["tpr_all"])
        print(f"{k:>14} | {a.mean():.3f} +/- {a.std():.3f} | {g.mean():.3f} +/- {g.std():.3f}        | {al.mean():.3f} +/- {al.std():.3f}")
    # claim-2 head-to-head
    bg = np.array(rows["bayesian"]["tpr_gs"]); pg = np.array(rows["parallel"]["tpr_gs"])
    diff = bg - pg
    print(f"  claim2  bayesian-parallel TPR@FAR_genuine_scam: mean {diff.mean():+.3f}, "
          f"bayesian>parallel in {int((diff > 0).sum())}/{n_seeds} seeds")


def main():
    events = sys.argv[1] if len(sys.argv) > 1 else "data/acoustic_events.json"
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    pools = build_pools()
    for src in ["asv", "itw"]:
        run(events, src, pools, n_seeds)


if __name__ == "__main__":
    main()
