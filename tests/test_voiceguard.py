import numpy as np
from voiceguard.simulate import generate_dataset
from voiceguard.fusion import running_mean, adaptive_fire, bayesian_fire, parallel_fire
from voiceguard.evaluate import first_true, evaluate, tune
from voiceguard.realsignals import align_to_timeline, build_matrices, RealCall


def test_running_mean():
    x = np.array([[0.0, 1.0, 2.0]])
    assert np.allclose(running_mean(x), [[0.0, 0.5, 1.0]])


def test_first_true():
    grid = np.array([[False, False, True], [False, False, False]])
    det, idx = first_true(grid)
    assert det.tolist() == [True, False]
    assert idx[0] == 2


def test_dataset_labels_and_shapes():
    ds = generate_dataset(n=300, seed=1)
    assert ds.acoustic.shape == ds.intent.shape == (300, ds.T)
    assert (ds.positive == (ds.is_clone & ds.is_scam)).all()
    assert ds.positive.sum() > 0


def test_adaptive_gate_opens_on_rising_intent():
    # weak-but-present acoustic + intent that crosses the gate mid-stream:
    # adaptive should fire only after the gate opens, not before.
    acc = np.full((1, 10), 0.55)
    intent = np.concatenate([np.zeros(5), np.ones(5)])[None, :]
    fired = adaptive_fire(acc, intent, tau_a0=0.7, k=0.3, gate=0.5)
    assert not fired[0, :5].any()
    assert fired[0, 5:].any()


def test_bayesian_vetoes_genuine_once_evidence_accumulates():
    # genuine voice (acoustic well below 0.5) + maxed intent: once enough negative
    # acoustic evidence accumulates, the posterior is driven down and the high
    # intent prior is VETOED (no fire at steady state). A clone (acoustic above
    # 0.5) + high intent fires. (Early steps may fire before evidence accrues --
    # that's a real property the tuner controls via the FAR budget.)
    T = 30
    genuine = np.full((1, T), 0.30)
    clone = np.full((1, T), 0.70)
    intent = np.ones((1, T))
    assert not bayesian_fire(genuine, intent, gain=0.8, beta=2.0, tau=0.5)[0, -1]
    assert bayesian_fire(clone, intent, gain=0.8, beta=2.0, tau=0.5)[0, -1]


def test_align_to_timeline_forward_fills():
    # 10s call, 5 steps -> step right-edges [2,4,6,8,10]. Events at t=3 and t=7: the
    # first step (edge 2, before any event) gets the fill; later steps hold the most
    # recent event value (forward fill).
    out = align_to_timeline([(3.0, 0.9), (7.0, 0.3)], T=5, duration_s=10.0, fill=0.5)
    assert np.allclose(out, [0.5, 0.9, 0.9, 0.3, 0.3])


def test_build_matrices_shapes():
    calls = [RealCall(acoustic_events=[(0.0, 0.8)], intent_events=[(1.0, 0.7)],
                      duration_s=5.0, is_clone=True, is_scam=True, call_type="clone_scam")]
    acoustic, intent, is_clone, is_scam, call_type = build_matrices(calls, T=8)
    assert acoustic.shape == intent.shape == (1, 8)
    assert is_clone[0] and is_scam[0]


def test_tune_respects_far_budget():
    ds = generate_dataset(n=600, seed=2)
    best = tune(parallel_fire,
                dict(w_a=[0.6, 1.0], w_i=[0.0, 0.4], tau=[0.5, 0.6, 0.7]),
                ds, far_budget=0.05)
    assert best is not None
    _, m = best
    assert m["far"] <= 0.05
