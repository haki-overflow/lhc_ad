"""Metrics, plus the audit: how much of the AE score is just multiplicity?"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, roc_curve


def rejection_at_efficiency(labels: np.ndarray, scores: np.ndarray, eff_s: float = 0.3) -> float:
    """Background rejection 1/eps_B at fixed signal efficiency."""
    fpr, tpr, _ = roc_curve(labels, scores)
    eps_b = np.interp(eff_s, tpr, fpr)
    return float(1.0 / eps_b) if eps_b > 0 else float("inf")


def summarize(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return {
        "auc": float(roc_auc_score(labels, scores)),
        "rej@0.3": rejection_at_efficiency(labels, scores, 0.3),
        "rej@0.5": rejection_at_efficiency(labels, scores, 0.5),
    }


def binned_auc(
    labels: np.ndarray,
    scores: np.ndarray,
    control: np.ndarray,
    n_bins: int = 10,
    min_per_class: int = 20,
) -> dict[str, object]:
    """AUC computed inside quantile bins of a control variable.

    If the AE only counts particles, its AUC collapses towards 0.5 once
    n_constituents is held (approximately) fixed. Bins are quantiles of the
    pooled control variable; bins lacking either class are skipped. The
    weighted mean uses the number of jets per bin.
    """
    edges = np.unique(np.quantile(control, np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.digitize(control, edges[1:-1]), 0, len(edges) - 2)
    per_bin, weights = [], []
    for b in range(len(edges) - 1):
        sel = idx == b
        y = labels[sel]
        if (y == 1).sum() < min_per_class or (y == 0).sum() < min_per_class:
            per_bin.append(np.nan)
            weights.append(0)
            continue
        per_bin.append(float(roc_auc_score(y, scores[sel])))
        weights.append(int(sel.sum()))
    per_bin_arr, w = np.array(per_bin), np.array(weights, dtype=float)
    ok = w > 0
    mean = float(np.average(per_bin_arr[ok], weights=w[ok])) if ok.any() else float("nan")
    return {"edges": edges.tolist(), "per_bin": per_bin_arr.tolist(), "weighted_mean": mean}


def correlations(scores: np.ndarray, hl: dict[str, np.ndarray], labels: np.ndarray) -> dict[str, dict[str, float]]:
    """Spearman correlation of the score with each observable, per class."""
    out: dict[str, dict[str, float]] = {}
    for cls, name in [(0, "qcd"), (1, "top")]:
        sel = labels == cls
        if sel.sum() < 3:
            continue
        out[name] = {k: float(spearmanr(scores[sel], v[sel]).statistic) for k, v in hl.items()}
    return out
