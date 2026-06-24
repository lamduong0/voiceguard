"""Stage 2 runner: same parallel-vs-bayesian comparison, fed by REAL model signals.

The only new code path vs Stage 1 is realsignals (irregular model outputs -> fixed-step
matrices). Everything downstream (tune, metrics, controllers) is reused.

Run with no arguments for a PIPELINE SMOKE TEST on fabricated events -- it proves the
real-signal path is wired end-to-end. It is NOT a result. Real results require plugging
an AcousticDetector + IntentClassifier over a real paired dataset (see STAGE2.md).
"""
import numpy as np
from voiceguard.simulate import Dataset, generate_dataset
from voiceguard.realsignals import RealCall, build_matrices
from voiceguard.evaluate import tune, evaluate, running_mean
from experiments.stage1_mechanism import METHODS, FAR_BUDGET


def dataset_from_calls(calls, T):
    acoustic, intent, is_clone, is_scam, call_type = build_matrices(calls, T)
    return Dataset(acoustic=acoustic, intent=intent, is_clone=is_clone,
                   is_scam=is_scam, call_type=call_type)


def run_comparison(train, test, constraint_key="far_hard"):
    """train/test are Dataset objects (real or demo). Returns {method: metrics}."""
    out = {}
    acc = running_mean(test.acoustic)
    for name, fire_fn, grid in METHODS:
        best = tune(fire_fn, grid, train, FAR_BUDGET, constraint_key=constraint_key)
        if best is None:
            out[name] = None
            continue
        params, _ = best
        out[name] = evaluate(fire_fn(acc, test.intent, **params), test)
    return out


def _demo_calls(n, T, seed):
    """Fabricate RealCall events from the Stage 1 simulator so the real-signal path can
    be exercised end-to-end. SMOKE TEST ONLY -- not a finding."""
    ds = generate_dataset(n=n, T=T, acoustic_sep=0.20, seed=seed)
    calls = []
    for i in range(n):
        acoustic_events = [(float(t), float(ds.acoustic[i, t])) for t in range(T)]
        intent_events = [(float(t), float(ds.intent[i, t])) for t in range(T)]
        calls.append(RealCall(acoustic_events=acoustic_events, intent_events=intent_events,
                              duration_s=float(T), is_clone=bool(ds.is_clone[i]),
                              is_scam=bool(ds.is_scam[i]), call_type=str(ds.call_type[i])))
    return calls


def main():
    T = 30
    train = dataset_from_calls(_demo_calls(1500, T, seed=0), T)
    test = dataset_from_calls(_demo_calls(1500, T, seed=999), T)
    res = run_comparison(train, test)
    print("Stage 2 PIPELINE SMOKE TEST (fabricated events -- NOT a result)\n")
    cols = f"{'method':>9} | {'det_rate':>8} | {'FAR':>5} | {'FAR_hard':>8} | {'median_TTD':>10}"
    print(cols)
    print("-" * len(cols))
    for name, _, _ in METHODS:
        m = res[name]
        if m is None:
            print(f"{name:>9} | (no config meets the budget)")
            continue
        print(f"{name:>9} | {m['det_rate']:>8.3f} | {m['far']:>5.3f} | "
              f"{m['far_hard']:>8.3f} | {m['median_ttd']:>10.1f}")
    print("\nPipeline OK. Plug real AcousticDetector + IntentClassifier to get real numbers.")


if __name__ == "__main__":
    main()
