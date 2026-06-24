"""Bridge from real model outputs to the Stage 1 evaluation harness.

Real detectors emit values at irregular times (acoustic per audio chunk, intent per
transcript turn). The fusion controllers expect fixed-length per-step matrices. This
module aligns the former to the latter, so `voiceguard.fusion` / `voiceguard.evaluate`
are reused verbatim -- Stage 2 changes only the *source* of the signals, not the math.

The AcousticDetector / IntentClassifier protocols are the only model-specific seams;
their concrete implementations depend on the chosen backends (pending decision) and are
deliberately not committed here.
"""
from dataclasses import dataclass
from typing import Protocol, Sequence
import numpy as np


@dataclass
class RealCall:
    acoustic_events: Sequence  # [(time_s, synthetic_prob), ...] from the acoustic model
    intent_events: Sequence    # [(time_s, scam_prob), ...] from the intent model
    duration_s: float
    is_clone: bool
    is_scam: bool
    call_type: str


class AcousticDetector(Protocol):
    def score_chunks(self, audio_path: str) -> Sequence:  # -> [(time_s, synthetic_prob)]
        ...


class IntentClassifier(Protocol):
    def score_turns(self, turns: Sequence) -> Sequence:   # -> [(time_s, scam_prob)]
        ...


def align_to_timeline(events, T, duration_s, fill=0.5, forward_fill=True):
    """Resample irregular (time, value) events onto T equal steps over [0, duration_s].
    Each step takes the most recent event value at/<= that step (forward fill); steps
    before the first event get `fill`. If forward_fill is False, gaps revert to `fill`."""
    out = np.full(T, float(fill))
    if not len(events):
        return out
    times = np.array([e[0] for e in events], dtype=float)
    vals = np.array([e[1] for e in events], dtype=float)
    order = np.argsort(times)
    times, vals = times[order], vals[order]
    step_edges = (np.arange(1, T + 1) / T) * duration_s  # right edge of each step
    idx = np.searchsorted(times, step_edges, side="right") - 1  # latest event index per step
    seen = idx >= 0
    out[seen] = vals[idx[seen]]
    if not forward_fill:
        # only keep a value in the step where its event actually lands
        landed = np.searchsorted(step_edges, times, side="left")
        out[:] = fill
        landed = landed[landed < T]
        out[landed] = vals[: len(landed)]
    return out


def build_matrices(calls: Sequence[RealCall], T: int):
    """Stack aligned per-call signals into the (acoustic[N,T], intent[N,T], labels)
    structure the Stage 1 harness consumes."""
    n = len(calls)
    acoustic = np.empty((n, T))
    intent = np.empty((n, T))
    for i, c in enumerate(calls):
        acoustic[i] = align_to_timeline(c.acoustic_events, T, c.duration_s, fill=0.5)
        intent[i] = align_to_timeline(c.intent_events, T, c.duration_s, fill=0.05)
    is_clone = np.array([c.is_clone for c in calls])
    is_scam = np.array([c.is_scam for c in calls])
    call_type = np.array([c.call_type for c in calls])
    return acoustic, intent, is_clone, is_scam, call_type
