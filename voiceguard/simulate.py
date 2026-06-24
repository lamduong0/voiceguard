"""Streaming-call simulator for the cross-modal fusion study.

Stage 1 is a *mechanism* study: we model the per-step outputs of an acoustic
deepfake detector and a transcript intent classifier, NOT raw audio. The point is
to test whether letting intent modulate the acoustic decision (adaptive fusion)
beats fixed-weight parallel fusion, before spending GPU on real models (Stage 2).
"""
from dataclasses import dataclass
import numpy as np

# Call archetypes. The positive we want to catch and intervene on is clone + scam
# intent. genuine_urgent is the HARD negative: a real family member whose speech is
# emotional/urgent (looks like scam intent) but the voice is genuine.
CLONE_SCAM = "clone_scam"          # positive target
GENUINE_BENIGN = "genuine_benign"  # easy negative
GENUINE_URGENT = "genuine_urgent"  # hard negative
CLONE_BENIGN = "clone_benign"      # rare


@dataclass
class Dataset:
    acoustic: np.ndarray   # [N, T] per-step synthetic-voice probability estimate
    intent: np.ndarray     # [N, T] per-step scam-intent score
    is_clone: np.ndarray   # [N] bool
    is_scam: np.ndarray    # [N] bool (scam intent present)
    call_type: np.ndarray  # [N] str

    @property
    def positive(self):
        return self.is_clone & self.is_scam

    @property
    def T(self):
        return self.acoustic.shape[1]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def generate_dataset(n=1200, T=30, acoustic_sep=0.25, acoustic_noise=0.22,
                     early_noise=4.0, intent_noise=0.08, seed=0,
                     mix=(0.35, 0.40, 0.20, 0.05)):
    """Simulate `n` calls of `T` steps (~1s of speech each).

    acoustic_sep : separation between clone and genuine acoustic means (small =
                   ambiguous phone audio, the hard regime).
    early_noise  : acoustic estimate is noisier early in the call (less audio);
                   noise scales with early_noise/(t+1) and shrinks over time.
    mix          : fraction of (clone_scam, genuine_benign, genuine_urgent, clone_benign).
    """
    rng = np.random.default_rng(seed)
    types = rng.choice([CLONE_SCAM, GENUINE_BENIGN, GENUINE_URGENT, CLONE_BENIGN],
                       size=n, p=list(mix))
    is_clone = np.isin(types, [CLONE_SCAM, CLONE_BENIGN])
    is_scam = np.isin(types, [CLONE_SCAM])
    rising_intent = np.isin(types, [CLONE_SCAM, GENUINE_URGENT])

    t = np.arange(T)
    # Acoustic: mean separated by acoustic_sep around 0.5; per-step noise shrinks as
    # the call proceeds (evidence accumulates).
    mu = np.where(is_clone, 0.5 + acoustic_sep / 2, 0.5 - acoustic_sep / 2)[:, None]
    sd = acoustic_noise * (1 + early_noise / (t + 1))[None, :]
    acoustic = np.clip(mu + rng.normal(size=(n, T)) * sd, 0.0, 1.0)

    # Intent: rising sigmoid for rising_intent calls (the "ask" lands mid-call),
    # flat-low otherwise; onset jittered per call.
    onset = rng.uniform(T * 0.25, T * 0.60, size=n)[:, None]
    width = T * 0.12
    ramp = _sigmoid((t[None, :] - onset) / width)
    base = np.where(rising_intent[:, None], ramp, 0.05)
    intent = np.clip(base + rng.normal(size=(n, T)) * intent_noise, 0.0, 1.0)

    return Dataset(acoustic=acoustic, intent=intent, is_clone=is_clone,
                   is_scam=is_scam, call_type=types)
