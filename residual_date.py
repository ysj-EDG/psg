"""M2 V5 CreationTime residual; functions copied verbatim from upstream team_code.py.

Source: BaoXinQi/psg4ci, f05c3b12b2a2fcfdc7c1518521c0610d11c11021.
The original additive coefficient is retained; this is not the V14/V18 blend.
"""
from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd

AGE_GAP = 2.0


def _clean_identifier(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and np.isfinite(value) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


def _row_value(row: pd.Series | dict, names: list[str], required: bool = True) -> Any:
    for name in names:
        if name in row and not pd.isna(row[name]) and str(row[name]).strip() != "":
            return row[name]
    if required:
        raise KeyError(f"Missing required field; tried {names}")
    return None


def _parse_binary(value: Any) -> int | None:
    if pd.isna(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, np.integer, float, np.floating)) and float(value) in {0.0, 1.0}:
        return int(value)
    text = str(value).strip().lower()
    if text in {"true", "t", "yes", "y", "1", "1.0", "positive"}:
        return 1
    if text in {"false", "f", "no", "n", "0", "0.0", "negative"}:
        return 0
    return None


def _creation_day(value: Any) -> float:
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        return float("nan")
    return float(parsed.value / 86_400_000_000_000.0)


def _legal_date_differences(frame: pd.DataFrame) -> np.ndarray:
    labels = frame["label"].to_numpy(dtype=int)
    ages = frame["age"].to_numpy(dtype=float)
    dates = frame["date_z"].to_numpy(dtype=float)
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    legal = np.abs(ages[positive, None] - ages[negative][None, :]) <= AGE_GAP
    return (dates[positive, None] - dates[negative][None, :])[legal]


def _fit_positive_site_macro_coefficient(groups: list[np.ndarray]) -> float:
    if not groups or any(group.size == 0 for group in groups):
        raise ValueError("Each site must contain legal age-matched positive-negative pairs")
    coefficient = 1.0
    site_weight = 1.0 / len(groups)
    for _ in range(50):
        gradient = 0.0
        hessian = 0.0
        for differences in groups:
            scaled = np.clip(coefficient * differences, -60.0, 60.0)
            probability = 1.0 / (1.0 + np.exp(-scaled))
            gradient += site_weight * float(
                np.mean(-differences * (1.0 - probability))
            )
            hessian += site_weight * float(
                np.mean(differences**2 * probability * (1.0 - probability))
            )
        if hessian <= 1e-12:
            break
        updated = float(np.clip(coefficient - gradient / hessian, 0.0, 10.0))
        if abs(updated - coefficient) <= 1e-10:
            coefficient = updated
            break
        coefficient = updated
    return coefficient


def _fit_date_rule(frame: pd.DataFrame) -> dict[str, Any]:
    rows = []
    for row in frame.to_dict(orient="records"):
        label = _parse_binary(
            _row_value(
                row,
                ["Cognitive_Impairment", "cognitive_impairment", "label"],
                False,
            )
        )
        age_value = pd.to_numeric(
            _row_value(row, ["Age", "age"], False), errors="coerce"
        )
        age = float(age_value) if age_value is not None else float("nan")
        site = _clean_identifier(_row_value(row, ["SiteID", "site_id", "site"], False))
        day = _creation_day(
            _row_value(row, ["CreationTime", "creation_time"], False)
        )
        if label is not None and np.isfinite(age) and site and np.isfinite(day):
            rows.append({"label": label, "age": float(age), "site": site, "day": day})
    working = pd.DataFrame(rows)
    if working.empty:
        raise ValueError("No complete labeled rows are available for the date residual")
    mean_day = float(working["day"].mean())
    std_day = max(float(working["day"].to_numpy().std()), 1e-6)
    working["date_z"] = (working["day"] - mean_day) / std_day
    groups = []
    pair_counts = {}
    for site, site_frame in working.groupby("site", sort=True):
        differences = _legal_date_differences(site_frame.reset_index(drop=True))
        pair_counts[str(site)] = int(differences.size)
        if differences.size:
            groups.append(differences)
    if sum(pair_counts.values()) < 100:
        raise ValueError(f"Too few legal date pairs: {pair_counts}")
    coefficient = _fit_positive_site_macro_coefficient(groups)
    return {
        "version": "creation_time_site_macro_pairwise_v1",
        "source": "official_training_labels",
        "age_gap_years": AGE_GAP,
        "creation_day_mean": mean_day,
        "creation_day_std": std_day,
        "date_coefficient": coefficient,
        "training_pair_counts": pair_counts,
    }


def _date_adjustment(rule: dict[str, Any], row: pd.Series | dict) -> float:
    value = _row_value(row, ["CreationTime", "creation_time"], False)
    day = _creation_day(value)
    if not np.isfinite(day):
        return 0.0
    standardized = (
        day - float(rule["creation_day_mean"])
    ) / float(rule["creation_day_std"])
    return float(rule["date_coefficient"]) * standardized
