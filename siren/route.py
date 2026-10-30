"""Public route contracts for the SIREN five-view CORE_NO_CAISR20 route."""
from __future__ import annotations

import numpy as np

from per_epoch_features.extensions.caisr20_core_protocol import CAISR20_FEATURE_IDS

ROUTE_ID = "CORE_NO_CAISR20"
SOURCE_COMMIT = "626ec1e9f178ae5922db09fab79955a6182e0f67"
VIEW_DIMENSIONS = {"V1": 113, "V2": 129, "V3": 349, "V4": 113, "V5": 265}
STRUCTURAL_POSITIONS = sum(VIEW_DIMENSIONS.values())
UNIQUE_FEATURE_IDS = 949


def build_core_no_caisr20_v5(feature_ids, values):
    """Delete the canonical CAISR20 IDs from a 285-column CORE V5 matrix."""
    ids = tuple(str(value) for value in feature_ids)
    matrix = np.asarray(values, dtype=np.float64)
    if len(ids) != 285 or len(set(ids)) != 285:
        raise ValueError("CORE V5 must contain 285 unique feature IDs")
    if matrix.ndim != 2 or matrix.shape[1] != len(ids):
        raise ValueError("CORE V5 values must be N x 285")
    if tuple(ids[:20]) != tuple(CAISR20_FEATURE_IDS):
        raise ValueError("CORE V5 does not start with canonical CAISR20 IDs")
    keep = np.asarray([i for i, feature_id in enumerate(ids) if feature_id not in set(CAISR20_FEATURE_IDS)], dtype=np.int64)
    retained_ids = tuple(ids[i] for i in keep)
    if len(retained_ids) != VIEW_DIMENSIONS["V5"] or set(retained_ids) & set(CAISR20_FEATURE_IDS):
        raise ValueError("CORE_NO_CAISR20 V5 identity contract failed")
    return retained_ids, np.array(matrix[:, keep], dtype=np.float64, copy=True)


def mean_tie_percentile_ranks(scores):
    """Return average-tie percentile ranks and their unweighted column mean."""
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0 or not np.isfinite(values).all():
        raise ValueError("scores must be a non-empty finite matrix")
    n_rows = values.shape[0]
    ranks = np.empty_like(values, dtype=np.float64)
    for column in range(values.shape[1]):
        _, inverse, counts = np.unique(values[:, column], return_inverse=True, return_counts=True)
        less = np.concatenate(([0], np.cumsum(counts[:-1])))
        ranks[:, column] = (less[inverse] + 0.5 * counts[inverse]) / n_rows
    return ranks, ranks.mean(axis=1, dtype=np.float64)
