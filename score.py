"""Official aggregate scoring helpers for CAISR20 route outputs."""
from __future__ import annotations

import numpy as np

from evaluate_model import compute_auprc, compute_auroc, compute_auroc_age


def legal_pair_count(labels: np.ndarray, ages: np.ndarray, gap: float = 2) -> int:
    y = np.asarray(labels, dtype=np.int8)
    a = np.asarray(ages, dtype=np.float64)
    positives = np.flatnonzero(y == 1)
    negatives = np.flatnonzero(y == 0)
    return int(sum(abs(float(a[i]) - float(a[j])) <= gap for i in positives for j in negatives))


def score_site(labels: np.ndarray, ages: np.ndarray, scores: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(labels, dtype=np.int8)
    a = np.asarray(ages, dtype=np.float64)
    s = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or a.shape != y.shape or s.shape != y.shape or not np.isfinite(s).all():
        raise ValueError("labels, ages, and scores must be aligned finite vectors")
    return {
        "n_records": int(y.size),
        "n_positive": int(y.sum()),
        "ac_pair_count": legal_pair_count(y, a),
        "ac_auroc": float(compute_auroc_age(y, s, a, gap=2)),
        "auroc": float(compute_auroc(y, s)),
        "auprc": float(compute_auprc(y, s)),
    }


__all__ = ["legal_pair_count", "score_site"]
