"""Render the paper figures from existing data (deterministic, matches the tables).

F1: ROC of the four scorers on ASVspoof (m1).
F2: multiplicative-linear fusion gain vs detector quality (degradation sweep, m2).
"""
import os, json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from voiceguard.intent import build_pools
from experiments.stage2_run import build_dataset
from experiments.stage2_roc import auc, oof_scores
from experiments.stage2_degrade import metrics_one_seed

os.makedirs("figures", exist_ok=True)
pools = build_pools()


def roc_curve(s, posm, negm):
    thr = np.r_[np.inf, np.unique(s)[::-1], -np.inf]
    P, N = s[posm], s[negm]
    fpr = [(N >= t).mean() for t in thr]
    tpr = [(P >= t).mean() for t in thr]
    return fpr, tpr


# ---- F1: ROC on ASVspoof (m1) ----
data = [c for c in json.load(open("data/events_m1_600.json")) if c["source"] == "asv"]
ds = build_dataset(data, pools, seed=0)
a_mean, i_max = ds.acoustic.mean(1), ds.intent.max(1)
y = np.where(ds.positive, 1, np.where(~ds.positive, 0, -1))
folds = np.array_split(np.random.default_rng(0).permutation(len(y)), 2)
sc = oof_scores(a_mean, i_max, y, folds)

plt.figure(figsize=(5, 4.2))
for name, label in [("acoustic_only", "acoustic only"), ("intent_only", "intent only"),
                    ("parallel", "linear fusion"), ("bayesian", "multiplicative")]:
    fpr, tpr = roc_curve(sc[name], y == 1, y == 0)
    plt.plot(fpr, tpr, label=f"{label} (AUC {auc(sc[name], y == 1, y == 0):.3f})")
plt.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
plt.xlabel("false positive rate")
plt.ylabel("true positive rate (clone_scam)")
plt.title("ROC on ASVspoof (m1)")
plt.legend(fontsize=8, loc="lower right")
plt.tight_layout()
plt.savefig("figures/F1_roc.png", dpi=160)
plt.savefig("figures/F1_roc.pdf")

# ---- F2: gain vs detector quality (m2 degradation sweep) ----
deg = json.load(open("data/acoustic_degrade_m2.json"))
levels = ["clean", 20.0, 15.0, 10.0, 5.0, 0.0]
plt.figure(figsize=(5, 4.2))
for src, mk, lab in [("asv", "o", "ASVspoof"), ("itw", "s", "In-the-Wild")]:
    xs, ys = [], []
    for lv in levels:
        clips = [r for r in deg if r["source"] == src and r["snr"] == lv]
        if not clips:
            continue
        A, G = [], []
        for s in range(5):
            a, b, p = metrics_one_seed(clips, pools, s)
            A.append(a); G.append(b - p)
        xs.append(np.mean(A)); ys.append(np.mean(G))
    o = np.argsort(xs)
    plt.plot(np.array(xs)[o], np.array(ys)[o], marker=mk, label=lab)
plt.axhline(0, color="k", lw=0.8, alpha=0.4)
plt.xlabel("detector quality (acoustic-only AUC)")
plt.ylabel("gain: multiplicative − linear\n(TPR@10%FAR vs genuine_scam)")
plt.title("Fusion gain vs detector quality")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("figures/F2_gain_vs_quality.png", dpi=160)
plt.savefig("figures/F2_gain_vs_quality.pdf")
print("WROTE figures/F1_roc.{png,pdf} and figures/F2_gain_vs_quality.{png,pdf}")
