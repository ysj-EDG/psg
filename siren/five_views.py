"""Five-view construction from frozen Rank256 matrices."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .route import VIEW_DIMENSIONS, build_core_no_caisr20_v5


def load_g1b_views(path: Path, role: str) -> dict[str, np.ndarray]:
    """Load one train/test role from the historical G1B ``views.npz`` unit."""
    role = str(role)
    if role not in {"train", "test"}:
        raise ValueError("role must be train or test")
    with np.load(Path(path), allow_pickle=False) as archive:
        out = {view: np.asarray(archive[f"{role}_{view}_values"], dtype=np.float64) for view in ("V1", "V2", "V3", "V4", "V5")}
    expected = {**VIEW_DIMENSIONS, "V5": 123}
    for view, matrix in out.items():
        if matrix.ndim != 2 or matrix.shape[1] != expected[view]:
            raise ValueError(f"G1B {role}/{view} width mismatch")
    return out


def build_five_view_matrices(g1b_views: dict[str, np.ndarray], core_v5_ids, core_v5_values) -> dict[str, np.ndarray]:
    """Return V1--V4 G1B matrices and canonical-ID-filtered CORE V5."""
    required = ("V1", "V2", "V3", "V4")
    if tuple(g1b_views) != required:
        raise ValueError("G1B input must contain only V1--V4")
    for view in required:
        matrix = np.asarray(g1b_views[view], dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != VIEW_DIMENSIONS[view]:
            raise ValueError(f"{view} width mismatch")
    ids, v5 = build_core_no_caisr20_v5(core_v5_ids, core_v5_values)
    if len(ids) != VIEW_DIMENSIONS["V5"]:
        raise ValueError("V5 width mismatch")
    out = {view: np.asarray(g1b_views[view], dtype=np.float64) for view in required}
    out["V5"] = v5
    return out
