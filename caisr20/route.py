"""Public CAISR20 historical-semantics route contract."""
from __future__ import annotations

import numpy as np

from per_epoch_features.extensions.caisr20_core_protocol import (
    CAISR20_FEATURE_IDS,
    CAISR20_X_STATIC_INDICES,
    HOLDOUT_TO_TRAIN,
    EXPECTED_SITE_COUNTS,
    EXPECTED_SITE_POSITIVES,
    SEEDS,
    SITES,
    apply_historical_typed_caisr20,
    apply_selected_preprocessor,
    fit_historical_typed_caisr20,
    fit_selected_preprocessor,
)
from per_epoch_features.extensions.caisr20_core_protocol import model_config as _model_config

ROUTE_ID = "CAISR20_HISTORICAL_SEMANTICS_CORE_PROTOCOL"
SOURCE_COMMIT = "626ec1e9f178ae5922db09fab79955a6182e0f67"


def model_config(seed: int) -> dict[str, object]:
    config = dict(_model_config(seed))
    config["decision_function"] = True
    return config


def mean_rank_fusion(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or not np.isfinite(values).all():
        raise ValueError("rank fusion requires a finite non-empty 2-D score matrix")
    ranks = np.empty_like(values, dtype=np.float64)
    for column in range(values.shape[1]):
        _, inverse, counts = np.unique(values[:, column], return_inverse=True, return_counts=True)
        less = np.concatenate(([0], np.cumsum(counts[:-1])))
        ranks[:, column] = (less[inverse] + 0.5 * counts[inverse]) / values.shape[0]
    return ranks.mean(axis=1, dtype=np.float64)


__all__ = [
    "ROUTE_ID", "SOURCE_COMMIT", "CAISR20_FEATURE_IDS", "CAISR20_X_STATIC_INDICES",
    "HOLDOUT_TO_TRAIN", "EXPECTED_SITE_COUNTS", "EXPECTED_SITE_POSITIVES", "SEEDS", "SITES",
    "model_config", "mean_rank_fusion", "apply_historical_typed_caisr20",
    "apply_selected_preprocessor", "fit_historical_typed_caisr20", "fit_selected_preprocessor",
]
