"""Fusion controllers: the prior-art baseline (parallel) vs the candidate-novel
adaptive one.

Both expose a fire grid `(acc, intent, **params) -> bool[N, T]`: whether the model
would raise the in-call alert at each step. `acc` is the running mean of the
acoustic signal (evidence accumulation). Detection time is the first True step.
"""
import numpy as np


def running_mean(acoustic):
    csum = np.cumsum(acoustic, axis=1)
    denom = np.arange(1, acoustic.shape[1] + 1)[None, :]
    return csum / denom


def parallel_fire(acc, intent, w_a, w_i, tau):
    """Fixed-weight late fusion (prior-art baseline, e.g. US12284313-style): a static
    weighted sum of the two signals crosses a fixed threshold."""
    return (w_a * acc + w_i * intent) >= tau


def adaptive_fire(acc, intent, tau_a0, k, gate):
    """Cross-modal adaptive fusion (linear, naive): rising intent LOWERS the acoustic
    decision threshold in real time, but only once intent clears `gate` -- so weak
    acoustic evidence can fire early when the conversation turns scam-like, while
    staying silent on benign calls (intent never opens the gate)."""
    eff_thresh = tau_a0 - k * intent
    return (acc >= eff_thresh) & (intent >= gate)


def bayesian_fire(acc, intent, gain, beta, tau):
    """Sequential Bayesian cross-modal fusion (principled candidate).

    Accumulate centered acoustic log-evidence over the call (SPRT-style: the
    cumulative sum of (x - 0.5) is `(t+1)*(acc-0.5)`), let intent shift the clone
    prior dynamically (beta term), and require BOTH a high clone posterior AND scam
    intent via a multiplicative joint. Strong genuine-voice evidence (negative
    accumulated acoustic) drives the posterior down and can VETO a high-intent
    genuine call -- the hard-negative resistance the linear adaptive controller
    lacked. `gain` is tuned like the baseline's weights (no oracle knowledge of the
    true acoustic separation)."""
    steps = np.arange(1, acc.shape[1] + 1)[None, :]
    evidence = gain * steps * (acc - 0.5)          # accumulated centered acoustic
    p_clone = 1.0 / (1.0 + np.exp(-(evidence + beta * (intent - 0.5))))
    risk = p_clone * intent
    return risk >= tau
