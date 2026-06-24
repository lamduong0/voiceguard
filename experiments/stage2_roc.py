"""Stage 2 rigorous re-analysis (no GPU).

The streaming/TTD dimension is degenerate on real data (per-clip acoustic is ~static),
so we score each CALL as a whole and compare methods fairly:
  - acoustic_only : mean P(synthetic)
  - intent_only   : max scam-intent
  - parallel      : a*acoustic + (1-a)*intent      (a fit out-of-fold)
  - bayesian      : sigmoid(g*(acoustic-c))*intent  (c calibrated to data, g fit out-of-fold)

Fair comparison: out-of-fold scoring (2-fold, no leakage), then on the pooled
out-of-fold scores report AUC and TPR at a MATCHED false-alarm rate (on the
genuine_scam hard negatives, and on all negatives), each with bootstrap 95% CIs.
"""
import json, numpy as np
from voiceguard.intent import build_pools
from experiments.stage2_run import build_dataset

FAR = 0.10
B = 2000
RNG = np.random.default_rng(0)


def _avg_ranks(x):
    """1-indexed ranks with ties averaged (required for a correct Mann-Whitney/AUC;
    plain sort-order ranks bias AUC low whenever scores tie)."""
    order = np.argsort(x, kind="mergesort")
    sx = x[order]
    ranks_sorted = np.empty(len(x), float)
    i = 0
    while i < len(x):
        j = i
        while j < len(x) and sx[j] == sx[i]:
            j += 1
        ranks_sorted[i:j] = (i + j + 1) / 2.0   # mean of ranks (i+1 .. j)
        i = j
    out = np.empty(len(x), float)
    out[order] = ranks_sorted
    return out


def auc(scores, pos, neg):
    sp, sn = scores[pos], scores[neg]
    if not len(sp) or not len(sn):
        return float("nan")
    ranks = _avg_ranks(np.concatenate([sp, sn]))
    r_pos = ranks[:len(sp)].sum()
    return (r_pos - len(sp) * (len(sp) + 1) / 2) / (len(sp) * len(sn))


def tpr_at_far(scores, pos, neg, far):
    if not np.any(neg):
        return float("nan")
    thr = np.quantile(scores[neg], 1 - far)
    return (scores[pos] > thr).mean()


def fit_parallel(a, i, y):
    best_alpha, best = 0.5, -1
    for alpha in np.linspace(0, 1, 21):
        s = alpha * a + (1 - alpha) * i
        v = auc(s, y == 1, y == 0)
        if v > best:
            best, best_alpha = v, alpha
    return best_alpha


def fit_bayesian(a, i, y, c):
    best_g, best = 1.0, -1
    for g in np.linspace(0.5, 25, 25):
        s = (1 / (1 + np.exp(-g * (a - c)))) * i
        v = auc(s, y == 1, y == 0)
        if v > best:
            best, best_g = v, g
    return best_g


def oof_scores(a_mean, i_max, y, idx_folds):
    """Out-of-fold per-call scores for each method (no train/test leakage)."""
    out = {k: np.zeros(len(y)) for k in ["acoustic_only", "intent_only", "parallel", "bayesian"]}
    for test_idx in idx_folds:
        tr = np.setdiff1d(np.arange(len(y)), test_idx)
        c = (a_mean[tr][y[tr] == 1].mean() + a_mean[tr][y[tr] == 0].mean()) / 2  # calibrated center
        alpha = fit_parallel(a_mean[tr], i_max[tr], y[tr])
        g = fit_bayesian(a_mean[tr], i_max[tr], y[tr], c)
        out["acoustic_only"][test_idx] = a_mean[test_idx]
        out["intent_only"][test_idx] = i_max[test_idx]
        out["parallel"][test_idx] = alpha * a_mean[test_idx] + (1 - alpha) * i_max[test_idx]
        out["bayesian"][test_idx] = (1 / (1 + np.exp(-g * (a_mean[test_idx] - c)))) * i_max[test_idx]
    return out


def boot(fn, n):
    vals = []
    for _ in range(B):
        bi = RNG.integers(0, n, n)
        vals.append(fn(bi))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return np.mean(vals), lo, hi


def run(source, pools, path="data/acoustic_events.json"):
    data = [c for c in json.load(open(path)) if c["source"] == source]
    ds = build_dataset(data, pools, seed=0)
    a_mean = ds.acoustic.mean(1)
    i_max = ds.intent.max(1)
    pos = ds.positive
    gscam = ds.call_type == "genuine_scam"   # decisive hard neg: high intent, genuine voice
    y = np.where(pos, 1, np.where(~pos, 0, -1))  # 1=clone_scam, 0=negative

    perm = RNG.permutation(len(y))
    folds = np.array_split(perm, 2)
    scores = oof_scores(a_mean, i_max, y, folds)

    print(f"\n### source={source}  n={len(y)}  pos(clone_scam)={int(pos.sum())}  genuine_scam={int(gscam.sum())}")
    cols = f"{'method':>14} | {'AUC':>18} | {'TPR@10%FAR_genuine_scam':>30} | {'TPR@10%FAR_all':>24}"
    print(cols); print("-" * len(cols))
    for name, s in scores.items():
        auc_m = boot(lambda bi, s=s: auc(s[bi], y[bi] == 1, y[bi] == 0), len(y))
        tg = boot(lambda bi, s=s: tpr_at_far(s[bi], y[bi] == 1, gscam[bi], FAR), len(y))
        ta = boot(lambda bi, s=s: tpr_at_far(s[bi], y[bi] == 1, y[bi] == 0, FAR), len(y))
        print(f"{name:>14} | {auc_m[0]:.3f} [{auc_m[1]:.3f},{auc_m[2]:.3f}] | "
              f"{tg[0]:.3f} [{tg[1]:.3f},{tg[2]:.3f}]          | "
              f"{ta[0]:.3f} [{ta[1]:.3f},{ta[2]:.3f}]")
    print("-" * len(cols))


def main():
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/acoustic_events.json"
    pools = build_pools()
    for src in ["asv", "itw"]:
        run(src, pools, path)


if __name__ == "__main__":
    main()
