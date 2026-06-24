"""Stage 1 mechanism study.

Question: does cross-modal ADAPTIVE fusion (intent modulates the acoustic
threshold) detect clone+scam calls earlier, at equal false-alarm rate, vs a
well-tuned PARALLEL fusion baseline -- and does its edge grow as the acoustic
signal gets more ambiguous (noisy/short phone audio)?

Numbers here are from a simulator of detector outputs, NOT real audio. They tell us
whether the mechanism is worth validating on real models in Stage 2.
"""
import numpy as np
from voiceguard.simulate import generate_dataset
from voiceguard.fusion import parallel_fire, adaptive_fire, bayesian_fire
from voiceguard.evaluate import tune, evaluate, running_mean

FAR_BUDGET = 0.05

PARALLEL_GRID = dict(
    w_a=np.round(np.linspace(0.4, 1.0, 7), 3),
    w_i=np.round(np.linspace(0.0, 0.7, 8), 3),
    tau=np.round(np.linspace(0.45, 0.90, 19), 3),
)
ADAPTIVE_GRID = dict(
    tau_a0=np.round(np.linspace(0.50, 0.85, 8), 3),
    k=np.round(np.linspace(0.0, 0.5, 11), 3),
    gate=np.round(np.linspace(0.10, 0.60, 6), 3),
)
BAYES_GRID = dict(
    gain=np.round(np.linspace(0.1, 1.5, 8), 3),
    beta=np.round(np.linspace(0.0, 4.0, 9), 3),
    tau=np.round(np.linspace(0.20, 0.80, 13), 3),
)

METHODS = [("parallel", parallel_fire, PARALLEL_GRID),
           ("adaptive", adaptive_fire, ADAPTIVE_GRID),
           ("bayesian", bayesian_fire, BAYES_GRID)]


def run_once(acoustic_sep, constraint_key, seed=0):
    train = generate_dataset(n=1500, acoustic_sep=acoustic_sep, seed=seed)
    test = generate_dataset(n=1500, acoustic_sep=acoustic_sep, seed=seed + 999)
    out = {}
    for name, fire_fn, grid in METHODS:
        best = tune(fire_fn, grid, train, FAR_BUDGET, constraint_key=constraint_key)
        if best is None:
            out[name] = None
            continue
        params, _ = best
        acc = running_mean(test.acoustic)
        out[name] = (params, evaluate(fire_fn(acc, test.intent, **params), test))
    return out


def report(constraint_key):
    label = {"far": "total FAR", "far_hard": "hard-negative FAR"}[constraint_key]
    print(f"\n### Tuned to equal {label} <= {FAR_BUDGET:.0%}  (positive = clone+scam)")
    print("lower median_TTD = earlier detection = less money lost\n")
    cols = f"{'acoustic_sep':>12} | {'method':>9} | {'det_rate':>8} | {'FAR':>5} | {'FAR_hard':>8} | {'median_TTD':>10}"
    print(cols)
    print("-" * len(cols))
    for sep in [0.45, 0.30, 0.20, 0.12]:
        res = run_once(sep, constraint_key)
        for name, _, _ in METHODS:
            r = res[name]
            if r is None:
                print(f"{sep:>12.2f} | {name:>9} | (no config meets the budget)")
                continue
            _, m = r
            print(f"{sep:>12.2f} | {name:>9} | {m['det_rate']:>8.3f} | {m['far']:>5.3f} | "
                  f"{m['far_hard']:>8.3f} | {m['median_ttd']:>10.1f}")
        print("-" * len(cols))


def main():
    print(f"Stage 1 mechanism study  (FAR budget = {FAR_BUDGET:.0%})")
    report("far")
    report("far_hard")


if __name__ == "__main__":
    main()
