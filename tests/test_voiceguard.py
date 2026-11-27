import numpy as np
from voiceguard.simulate import generate_dataset
from voiceguard.fusion import running_mean, adaptive_fire, bayesian_fire, parallel_fire
from voiceguard.evaluate import first_true, evaluate, tune
from voiceguard.realsignals import align_to_timeline, build_matrices, RealCall
from experiments.stage2_roc import auc, tpr_at_far
import math
import pytest


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


def test_auc_handles_ties():
    # positives [1,2], negatives [1,3]: tie-averaged AUC = 0.375 (sequential-rank bug = 0.25).
    s = np.array([1.0, 2.0, 1.0, 3.0])
    pos = np.array([True, True, False, False])
    assert abs(auc(s, pos, ~pos) - 0.375) < 1e-9


def test_auc_perfect_separation():
    s = np.array([2.0, 3.0, 0.0, 1.0])
    pos = np.array([True, True, False, False])
    assert auc(s, pos, ~pos) == 1.0


def test_tpr_at_far_empty_negatives_is_nan():
    s = np.array([0.9, 0.8])
    pos = np.array([True, True])
    assert math.isnan(tpr_at_far(s, pos, np.array([False, False]), 0.1))


def test_keyword_intent_ignores_contraction_wont():
    # regression: the \bwon\b lottery cue must not fire on the contraction "won't",
    # but must still catch a genuine "won a prize" scam.
    from voiceguard.detect import keyword_intent
    assert keyword_intent([{"text": "Why won't you pick up?"}]) == 0.05
    assert keyword_intent([{"text": "You won a prize, claim your gift!"}]) > 0.5


def test_tune_respects_far_budget():
    ds = generate_dataset(n=600, seed=2)
    best = tune(parallel_fire,
                dict(w_a=[0.6, 1.0], w_i=[0.0, 0.4], tau=[0.5, 0.6, 0.7]),
                ds, far_budget=0.05)
    assert best is not None
    _, m = best
    assert m["far"] <= 0.05


def test_tpr_at_far_splits_ties_at_threshold():
    # Negatives tied at the threshold must be split so the achieved FAR equals the
    # target; positives tied there get the same share. A strict `>` reports 0 here.
    neg_tied = np.full(10, 1.0)
    pos_tied = np.full(5, 1.0)
    s = np.concatenate([pos_tied, neg_tied])
    pos = np.arange(15) < 5
    assert tpr_at_far(s, pos, ~pos, 0.1) == pytest.approx(0.1)
    # 8 negatives at 0 and 2 tied at the 90th-percentile threshold 1.0: flag half of
    # the tied ones. Positives: one above (2.0), two tied (1.0), one below (0.5)
    # -> 1/4 + 0.5 * 2/4 = 0.5.
    s = np.array([2.0, 1.0, 1.0, 0.5] + [0.0] * 8 + [1.0, 1.0])
    pos = np.arange(14) < 4
    assert tpr_at_far(s, pos, ~pos, 0.1) == pytest.approx(0.5)


def test_oof_center_calibrated_on_clone_label():
    # The acoustic center must split clone vs genuine voices. clone_benign calls are
    # negatives (y=0) but clone voices; averaging them into the negative mean drags
    # the center up (0.7 here instead of 0.5).
    from experiments.stage2_roc import oof_scores
    a = np.array([0.1] * 4 + [0.9] * 4 + [0.9] * 4 + [0.5])  # genuine, clone_scam, clone_benign, probe
    i = np.array([0.1, 0.9, 0.1, 0.9] + [0.9] * 4 + [0.1] * 4 + [1.0])
    y = np.array([0] * 4 + [1] * 4 + [0] * 4 + [0])
    clone = np.array([False] * 4 + [True] * 8 + [False])
    sc = oof_scores(a, i, y, [np.array([12])], clone)
    # probe sits exactly on the clone/genuine midpoint with intent 1 -> sigmoid(0) * 1
    assert sc["bayesian"][12] == pytest.approx(0.5)


def test_load_calibration_independent_of_cwd(tmp_path, monkeypatch):
    # Run from another directory: must still read the repo's data, not silently fall
    # back to m1/ASV defaults (c=0.21) for the m2/ITW detector (c=0.4102).
    from voiceguard.detect import load_calibration
    monkeypatch.chdir(tmp_path)
    assert load_calibration(path="data/events_m2_600.json", source="itw")["c"] == pytest.approx(0.4102)


def test_load_calibration_missing_file_raises():
    from voiceguard.detect import load_calibration
    with pytest.raises(FileNotFoundError):
        load_calibration(path="data/does_not_exist.json")


def test_load_audio_decodes_m4a_to_16k_mono(tmp_path):
    # Call recordings arrive as m4a (AAC), which libsndfile cannot decode.
    pytest.importorskip("faster_whisper")
    pytest.importorskip("librosa")
    import av
    from voiceguard.detect import load_audio
    path = str(tmp_path / "call.m4a")
    sr = 44100
    tone = (0.2 * np.sin(2 * np.pi * 440 * np.arange(2 * sr) / sr)).astype(np.float32)
    with av.open(path, "w", format="mp4") as out:
        st = out.add_stream("aac", rate=sr, layout="stereo")
        frame = av.AudioFrame.from_ndarray(np.stack([tone, tone]), format="fltp", layout="stereo")
        frame.rate = sr
        for pkt in list(st.encode(frame)) + list(st.encode(None)):
            out.mux(pkt)
    wav = load_audio(path)
    assert wav.ndim == 1
    assert abs(len(wav) - 2 * 16000) < 0.05 * 16000   # ~2 s at 16 kHz (AAC priming/padding)
    assert 0.1 < np.sqrt(np.mean(wav ** 2)) < 0.2       # the 0.2-amplitude tone (RMS 0.141), not silence


def test_cli_final_verdict_uses_llm_intent(monkeypatch, capsys):
    # With --llm, the final verdict must use the LLM intent the trace used, not the
    # keyword scorer (which rates these benign-looking words ~0.05).
    import json as _json
    import cli
    monkeypatch.setattr(cli, "score_intent", lambda turns: (0.9, "llm"))
    turns = [{"text": "How was your week?"}, {"text": "See you soon."}]
    cli.play(0.9, turns, {"c": 0.5, "g": 10.0}, json_out=True, use_llm=True)
    out = _json.loads(capsys.readouterr().out)
    # p_clone = sigmoid(10 * 0.4) = 0.98201; risk = 0.98201 * 0.9 = 0.884
    assert out["final_risk"] == pytest.approx(0.884, abs=1e-3)
    assert out["final_level"] == "HIGH"
