"""Multi-seed robustness check for the hard acoustic regime, under the
hard-negative FAR constraint (the operationally critical one).

Single-seed gaps can be noise. This re-runs the full tune-on-train / eval-on-test
loop across many seeds and reports mean +/- std, so we can tell whether the
Bayesian controller's edge in the ambiguous regime is real or seed luck.
"""
import numpy as np
from voiceguard.simulate import generate_dataset
from voiceguard.evaluate import tune, evaluate, running_mean
from experiments.stage1_mechanism import METHODS, FAR_BUDGET

SEEDS = range(10)


def collect(sep):
    stats = {name: {"det": [], "far_hard": [], "ttd": []} for name, _, _ in METHODS}
    for s in SEEDS:
        train = generate_dataset(n=1500, acoustic_sep=sep, seed=s)
        test = generate_dataset(n=1500, acoustic_sep=sep, seed=s + 999)
        acc = running_mean(test.acoustic)
        for name, fire_fn, grid in METHODS:
            best = tune(fire_fn, grid, train, FAR_BUDGET, constraint_key="far_hard")
            if best is None:
                continue
            params, _ = best
            m = evaluate(fire_fn(acc, test.intent, **params), test)
            stats[name]["det"].append(m["det_rate"])
            stats[name]["far_hard"].append(m["far_hard"])
            stats[name]["ttd"].append(m["median_ttd"])
    return stats


def main():
    n = len(list(SEEDS))
    print(f"Robustness over {n} seeds  (hard-negative FAR <= {FAR_BUDGET:.0%}, positive = clone+scam)\n")
    cols = f"{'sep':>5} | {'method':>9} | {'det_rate':>17} | {'FAR_hard':>15} | {'median_TTD':>15}"
    print(cols)
    print("-" * len(cols))
    for sep in [0.20, 0.12]:
        st = collect(sep)
        for name, _, _ in METHODS:
            d = np.array(st[name]["det"])
            f = np.array(st[name]["far_hard"])
            t = np.array(st[name]["ttd"])
            print(f"{sep:>5.2f} | {name:>9} | {d.mean():>6.3f} +/- {d.std():>5.3f} | "
                  f"{f.mean():>5.3f} +/- {f.std():>5.3f} | {np.nanmean(t):>5.1f} +/- {np.nanstd(t):>4.1f}")
        print("-" * len(cols))


if __name__ == "__main__":
    main()
