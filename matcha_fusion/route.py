"""Public AB_EQUAL_50_50 contract: rank each parent, then average equally."""
from __future__ import annotations

import numpy as np

ROUTE_ID = "AB_EQUAL_50_50"
SOURCE_COMMIT = "2e151f9ec89e596469ba315eebcc312e44d20044"
SITES = ("I0002", "I0006", "S0001")
CORE_WEIGHT = 0.5
MATCHA_WEIGHT = 0.5


class FusionStop(RuntimeError):
    """A parent identity, metric, or arithmetic contract failed."""


def _vector(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise FusionStop(f"{name} must be a finite vector")
    return array


def rank_average(scores: np.ndarray) -> np.ndarray:
    values = _vector(scores, "parent scores")
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    less = np.concatenate(([0], np.cumsum(counts[:-1])))
    return (less[inverse] + 0.5 * counts[inverse]) / values.size


def validate_parent_identity(left_indices: np.ndarray, right_indices: np.ndarray, holdout: str) -> None:
    if holdout not in SITES:
        raise FusionStop(f"unknown holdout: {holdout}")
    left = np.asarray(left_indices); right = np.asarray(right_indices)
    if left.ndim != 1 or right.ndim != 1 or left.dtype.kind not in "iu" or right.dtype.kind not in "iu":
        raise FusionStop("parent indices must be one-dimensional integer vectors")
    if not np.array_equal(left, right) or len(np.unique(left)) != len(left):
        raise FusionStop(f"parent record identity mismatch: {holdout}")


def fuse_ranked_scores(core_scores: np.ndarray, matcha_scores: np.ndarray) -> np.ndarray:
    core = rank_average(core_scores)
    matcha = rank_average(matcha_scores)
    if core.shape != matcha.shape:
        raise FusionStop("parent score lengths differ")
    fused = (CORE_WEIGHT * core + MATCHA_WEIGHT * matcha).astype(np.float64, copy=False)
    if not np.isfinite(fused).all():
        raise FusionStop("fused scores are nonfinite")
    return fused


__all__ = ["ROUTE_ID", "SOURCE_COMMIT", "SITES", "FusionStop", "rank_average", "validate_parent_identity", "fuse_ranked_scores"]
