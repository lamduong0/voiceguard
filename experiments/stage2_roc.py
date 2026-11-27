"""Stage 2 rigorous re-analysis (no GPU).

The streaming/TTD dimension is degenerate on real data (per-clip acoustic is ~static),
so we score each CALL as a whole and compare methods fairly:
  - acoustic_only : mean P(synthetic)
  - intent_only   : max scam-intent
  - parallel      : a*acoustic + (1-a)*intent      (a fit out-of-fold)
  - bayesian      : sigmoid(g*(acoustic-c))*intent  (c calibrated to data, g fit out-of-fold)

Out-of-fold scoring (2-fold), followed by AUC and TPR at a nominal false-alarm
target on pooled scores. Report achieved FAR and thresholds because ties and
finite samples affect the operating point. Bootstrap CIs condition on these
fitted scores, with paired resampling for the difference between fusion methods.
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
    return operating_point(scores, pos, neg, far)["tpr"]


def operating_point(scores, pos, neg, far):
    """Quantile threshold with randomized ties and the expected achieved FAR.

    Tied negatives and positives receive the same selection weight. Report the
    strict-threshold FAR too, since deploying tie randomization is a separate choice.
    Finite-sample quantiles can put FAR slightly above the nominal target.
    """
    if not np.any(neg):
        return dict(tpr=float("nan"), far=float("nan"), threshold=float("nan"),
                    strict_far=float("nan"), tie_weight=float("nan"))
    sn, sp = scores[neg], scores[pos]
    thr = np.quantile(sn, 1 - far)
    above, at = (sn > thr).mean(), (sn == thr).mean()
    weight = min(1.0, max(0.0, (far - above) / at)) if at else 0.0
    tpr = (sp > thr).mean() + weight * (sp == thr).mean() if len(sp) else float("nan")
    return dict(tpr=tpr, far=above + weight * at, threshold=thr,
                strict_far=above, tie_weight=weight)


def fit_parallel(a, i, y):
    best_alpha, best = 0.5, -1
    for alpha in np.linspace(0, 1, 21):
        s = alpha * a + (1 - alpha) * i
        v = auc(s, y == 1, y == 0)
        if v > best:
            best, best_alpha = v, alpha
    return best_alpha


def fit_bayesian(a, i, y, c, combine=np.multiply):
    """Gain g of the acoustic posterior sigmoid(g*(a-c)), chosen by AUC; `combine` joins
    it with intent (product = the multiplicative veto, np.minimum = an AND rule)."""
    best_g, best = 1.0, -1
    for g in np.linspace(0.5, 25, 25):
        s = combine(1 / (1 + np.exp(-g * (a - c))), i)
        v = auc(s, y == 1, y == 0)
        if v > best:
            best, best_g = v, g
    return best_g


def fit_logistic(X, y, iters=100, l2=1e-3):
    """Logistic regression by Newton/IRLS (numpy only; a small ridge keeps separable
    folds finite). The reviewer-standard learned fusion baseline."""
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X @ w, -30, 30)))
        H = X.T @ (X * (p * (1 - p))[:, None]) + l2 * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - p) - l2 * w)
        w += step
        if np.abs(step).max() < 1e-8:
            break
    return w


def _features(a, i):
    return np.column_stack([np.ones_like(a), a, i, a * i])


def oof_scores(a_mean, i_max, y, idx_folds, clone):
    """Out-of-fold per-call scores for each method (no train/test leakage). `clone` is
    the per-call voice label: the acoustic center splits clone vs genuine voices (not
    y, whose negatives include clone_benign calls)."""
    out = {k: np.zeros(len(y)) for k in ["acoustic_only", "intent_only", "parallel", "bayesian",
                                         "logistic", "and_rule"]}
    for test_idx in idx_folds:
        tr = np.setdiff1d(np.arange(len(y)), test_idx)
        c = (a_mean[tr][clone[tr]].mean() + a_mean[tr][~clone[tr]].mean()) / 2  # calibrated center
        alpha = fit_parallel(a_mean[tr], i_max[tr], y[tr])
        g = fit_bayesian(a_mean[tr], i_max[tr], y[tr], c)
        g_and = fit_bayesian(a_mean[tr], i_max[tr], y[tr], c, combine=np.minimum)
        w = fit_logistic(_features(a_mean[tr], i_max[tr]), (y[tr] == 1).astype(float))
        a_te, i_te = a_mean[test_idx], i_max[test_idx]
        out["acoustic_only"][test_idx] = a_te
        out["intent_only"][test_idx] = i_te
        out["parallel"][test_idx] = alpha * a_te + (1 - alpha) * i_te
        out["bayesian"][test_idx] = (1 / (1 + np.exp(-g * (a_te - c)))) * i_te
        out["logistic"][test_idx] = _features(a_te, i_te) @ w        # logit with a*i interaction
        out["and_rule"][test_idx] = np.minimum(1 / (1 + np.exp(-g_and * (a_te - c))), i_te)
    return out


def boot(fn, n, rng=None):
    rng = RNG if rng is None else rng
    vals = []
    for _ in range(B):
        bi = rng.integers(0, n, n)
        vals.append(fn(bi))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return np.mean(vals), lo, hi


def paired_tpr_gain(first, second, pos, neg, far=FAR):
    """Bootstrap the TPR difference using the same sampled calls for both methods."""
    return boot(
        lambda bi: tpr_at_far(first[bi], pos[bi], neg[bi], far)
        - tpr_at_far(second[bi], pos[bi], neg[bi], far),
        len(first), rng=np.random.default_rng(0),
    )


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
    scores = oof_scores(a_mean, i_max, y, folds, ds.is_clone)

    print(f"\n### source={source}  n={len(y)}  pos(clone_scam)={int(pos.sum())}  genuine_scam={int(gscam.sum())}")
    cols = f"{'method':>14} | {'AUC':>18} | {'TPR target10% genuine_scam':>30} | {'TPR target10% all':>24}"
    print(cols); print("-" * len(cols))
    for name, s in scores.items():
        auc_m = boot(lambda bi, s=s: auc(s[bi], y[bi] == 1, y[bi] == 0), len(y))
        tg = boot(lambda bi, s=s: tpr_at_far(s[bi], y[bi] == 1, gscam[bi], FAR), len(y))
        ta = boot(lambda bi, s=s: tpr_at_far(s[bi], y[bi] == 1, y[bi] == 0, FAR), len(y))
        print(f"{name:>14} | {auc_m[0]:.3f} [{auc_m[1]:.3f},{auc_m[2]:.3f}] | "
              f"{tg[0]:.3f} [{tg[1]:.3f},{tg[2]:.3f}]          | "
              f"{ta[0]:.3f} [{ta[1]:.3f},{ta[2]:.3f}]")
        for label, negatives in [("genuine_scam", gscam), ("all", y == 0)]:
            point = operating_point(s, pos, negatives, FAR)
            print(f"    {label} expected FAR={point['far']:.4f}  "
                  f"strict FAR={point['strict_far']:.4f}  tie weight={point['tie_weight']:.4f}  "
                  f"threshold={point['threshold']:.4f}  expected TPR={point['tpr']:.4f}")
    print("-" * len(cols))
    for other in ["parallel", "logistic", "and_rule"]:
        for label, negatives in [("genuine_scam", gscam), ("all", y == 0)]:
            gain = tpr_at_far(scores["bayesian"], pos, negatives, FAR) - tpr_at_far(
                scores[other], pos, negatives, FAR)
            _, lo, hi = paired_tpr_gain(scores["bayesian"], scores[other], pos, negatives)
            print(f"  multiplicative - {other} ({label}) observed gain={gain:+.3f}  "
                  f"paired bootstrap 95% CI=[{lo:+.3f}, {hi:+.3f}]")


def main():
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/acoustic_events.json"
    pools = build_pools()
    for src in ["asv", "itw"]:
        run(src, pools, path)


if __name__ == "__main__":
    main()
