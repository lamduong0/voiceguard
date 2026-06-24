"""Fair evaluation: tune each controller on train to a shared false-alarm budget,
then compare detection rate and time-to-detection on a held-out test set."""
import itertools
import numpy as np
from .fusion import running_mean


def first_true(grid):
    """Per row, the first column index that is True. Returns (detected[N], idx[N]);
    idx is 0 for all-False rows, so always gate on `detected`."""
    detected = grid.any(axis=1)
    idx = grid.argmax(axis=1)
    return detected, idx


def evaluate(grid, ds):
    detected, t_det = first_true(grid)
    pos = ds.positive
    neg = ~pos
    hard = ds.call_type == "genuine_urgent"
    det_rate = detected[pos].mean() if pos.any() else 0.0
    far = detected[neg].mean() if neg.any() else 0.0
    far_hard = detected[hard].mean() if hard.any() else 0.0
    ttd = t_det[pos & detected]
    median_ttd = float(np.median(ttd)) if ttd.size else float("nan")
    return dict(det_rate=float(det_rate), far=float(far), far_hard=float(far_hard),
                median_ttd=median_ttd, n_pos=int(pos.sum()), n_neg=int(neg.sum()))


def tune(fire_fn, grid_params, ds, far_budget, constraint_key="far"):
    """Grid-search params; keep the feasible (constraint_key <= budget) config with
    the best detection rate, tie-broken by earliest median detection. constraint_key
    is "far" (total false alarms) or "far_hard" (false alarms on the genuine-urgent
    hard negatives -- the operationally critical one). Returns (params, metrics) or
    None if nothing meets the budget."""
    acc = running_mean(ds.acoustic)
    keys = list(grid_params)
    best_key, best = None, None
    for combo in itertools.product(*[grid_params[k] for k in keys]):
        params = dict(zip(keys, combo))
        m = evaluate(fire_fn(acc, ds.intent, **params), ds)
        if m[constraint_key] > far_budget:
            continue
        ttd = m["median_ttd"]
        key = (m["det_rate"], -(ttd if ttd == ttd else 1e9))  # ttd==ttd is False for NaN
        if best_key is None or key > best_key:
            best_key, best = key, (params, m)
    return best
