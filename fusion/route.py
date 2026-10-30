"""Public DEDUP_FUSION contract: direct equal fusion of final parent scores."""
from __future__ import annotations

import numpy as np

ROUTE_ID = "DEDUP_FUSION"
SOURCE_COMMIT = "626ec1e9f178ae5922db09fab79955a6182e0f67"
SITES = ("I0002", "I0006", "S0001")
CORE_WEIGHT = 0.5
CAISR_WEIGHT = 0.5


class FusionStop(RuntimeError):
    """A parent identity, metric, or arithmetic contract failed."""


def _vector(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise FusionStop(f"{name} must be a finite vector")
    return array


def validate_parent_identity(left_indices: np.ndarray, right_indices: np.ndarray, holdout: str) -> None:
    if holdout not in SITES:
        raise FusionStop(f"unknown holdout: {holdout}")
    left = np.asarray(left_indices); right = np.asarray(right_indices)
    if left.ndim != 1 or right.ndim != 1 or left.dtype.kind not in "iu" or right.dtype.kind not in "iu":
        raise FusionStop("parent indices must be one-dimensional integer vectors")
    if not np.array_equal(left, right) or len(np.unique(left)) != len(left):
        raise FusionStop(f"parent record identity mismatch: {holdout}")


def fuse_final_scores(core_scores: np.ndarray, caisr_scores: np.ndarray) -> np.ndarray:
    core = _vector(core_scores, "CORE scores")
    caisr = _vector(caisr_scores, "CAISR20 scores")
    if core.shape != caisr.shape:
        raise FusionStop("parent score lengths differ")
    fused = (CORE_WEIGHT * core + CAISR_WEIGHT * caisr).astype(np.float64, copy=False)
    if not np.isfinite(fused).all():
        raise FusionStop("fused scores are nonfinite")
    return fused


__all__ = ["ROUTE_ID", "SOURCE_COMMIT", "SITES", "FusionStop", "validate_parent_identity", "fuse_final_scores"]
