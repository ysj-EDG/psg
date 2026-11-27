"""统一 NPZ 模型输入和冻结 P4-C2 checkpoint 推理适配。

只读取 FeatureStore 的 NPZ 与 joblib checkpoint；不读取 EDF/HDF5/TSV，不拟合
scaler，不修改 checkpoint。B1/D7 的 base 模型和 Frozen S2 校正按归档 payload
中的真实 feature_names、scaler 和系数执行。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.special import expit

from .feature_profiles import B1_PROFILE, D7_PROFILE, FeatureProfile
from .feature_store import FeatureStore, StoredFeatureRecord


PROFILE_BY_BASE = {"B1": B1_PROFILE, "D7": D7_PROFILE}


def _profile(profile: str | FeatureProfile) -> FeatureProfile:
    if isinstance(profile, FeatureProfile):
        return profile
    try:
        return PROFILE_BY_BASE[str(profile).upper()]
    except KeyError as exc:
        raise ValueError(f"unknown model profile: {profile}") from exc


def build_npz_model_input(store: FeatureStore, keys: list[tuple[str, str]], profile: str | FeatureProfile) -> dict[str, Any]:
    """按显式 site/record_id 从 NPZ 组装 B1 或 D7 原始模型输入。"""
    selected = _profile(profile)
    result = store.build_model_input(keys, selected)
    if result["X_base_raw"].shape[1] != selected.base_dim or result["X_hrv_raw"].shape[1] != selected.hrv_dim:
        raise ValueError("model input dimensions do not match selected profile")
    return result


def load_frozen_checkpoint(path: Path | str, expected_site: str | None = None) -> dict[str, Any]:
    path = Path(path).resolve()
    payload = joblib.load(path)
    if payload.get("model_version") != "p4c2_pair_frozen_s2_v1":
        raise ValueError(f"unsupported checkpoint version: {payload.get('model_version')}")
    holdout = str(payload.get("holdout_site", ""))
    if expected_site is not None and holdout != str(expected_site):
        raise ValueError(f"checkpoint holdout site mismatch: expected {expected_site}, got {holdout}")
    base_group = str(payload.get("base_group", ""))
    if base_group not in PROFILE_BY_BASE:
        raise ValueError(f"unsupported checkpoint base group: {base_group}")
    base_payload = payload.get("base_pair_model_payload")
    if not isinstance(base_payload, dict):
        raise ValueError("checkpoint missing base_pair_model_payload")
    profile = PROFILE_BY_BASE[base_group]
    expected_base = tuple(profile.base_feature_names)
    if tuple(payload.get("base_feature_names", ())) != expected_base:
        raise ValueError("checkpoint top-level base feature order does not match profile")
    if tuple(base_payload.get("feature_names", ())) != expected_base:
        raise ValueError("checkpoint base feature order does not match profile")
    hrv_names = tuple(payload.get("hrv_feature_names", ()))
    if hrv_names != tuple(profile.hrv_feature_names):
        raise ValueError("checkpoint HRV feature order does not match profile")
    hrv_scaler = payload.get("hrv_preprocessor")
    expected_full = tuple(profile.feature_names)
    if tuple(getattr(hrv_scaler, "feature_names", ())) != expected_full:
        raise ValueError("checkpoint Frozen S2 scaler feature order mismatch")
    coef = np.asarray(payload.get("correction_coef"), dtype=float).reshape(-1)
    if coef.shape != (2,):
        raise ValueError("Frozen S2 correction must have two coefficients")
    return {"path": path, "payload": payload, "profile": profile}


def predict_frozen_s2(checkpoint: dict[str, Any], model_input: dict[str, Any]) -> dict[str, np.ndarray]:
    """用归档 base Pair-Loss + Frozen S2 offset 回放 logits/probability。"""
    payload = checkpoint["payload"]
    profile: FeatureProfile = checkpoint["profile"]
    base_payload = payload["base_pair_model_payload"]
    x_base = np.asarray(model_input["X_base_raw"], dtype=np.float32)
    m_base = np.asarray(model_input["M_base"], dtype=bool)
    x_hrv = np.asarray(model_input["X_hrv_raw"], dtype=np.float64)
    m_hrv = np.asarray(model_input["M_hrv"], dtype=bool)
    if x_base.ndim != 2 or x_base.shape[1] != profile.base_dim or x_hrv.shape != (len(x_base), profile.hrv_dim):
        raise ValueError("model input shape mismatch")
    # Base Pair-Loss scaler receives only the frozen 29/32-dimensional base block.
    base_scaler = base_payload["scaler"]
    base_z = base_scaler.transform(x_base, m_base)
    base_logit = np.asarray(base_z, dtype=float) @ np.asarray(base_payload["coef"], dtype=float) + float(base_payload["intercept"])
    # The archived S2 scaler was fit on base+S01+S02. This is where S02 log1p
    # occurs exactly once; raw NPZ values remain float64 and unmodified.
    full_values = np.concatenate((x_base.astype(np.float64), x_hrv), axis=1)
    full_mask = np.concatenate((m_base, m_hrv), axis=1)
    full_z = payload["hrv_preprocessor"].transform(full_values, full_mask)
    hrv_z = full_z[:, -2:]
    correction_coef = np.asarray(payload["correction_coef"], dtype=float)
    correction_blocks = np.column_stack((
        hrv_z[:, 0] * correction_coef[0],
        hrv_z[:, 1] * correction_coef[1],
        np.full(len(base_logit), float(payload["correction_intercept"])),
    ))
    correction = correction_blocks.sum(axis=1)
    final_logit = base_logit + correction
    return {
        "base_logit": np.asarray(base_logit, dtype=float),
        "hrv_z": np.asarray(hrv_z, dtype=float),
        "correction_blocks": correction_blocks,
        "correction": np.asarray(correction, dtype=float),
        "final_logit": np.asarray(final_logit, dtype=float),
        "probability": np.asarray(expit(final_logit), dtype=float),
    }


def predict_one_npz(store: FeatureStore, record_id: str, site: str, checkpoint: dict[str, Any]) -> dict[str, np.ndarray]:
    profile = checkpoint["profile"]
    model_input = build_npz_model_input(store, [(record_id, site)], profile)
    return predict_frozen_s2(checkpoint, model_input)


def checkpoint_for_site(checkpoint_root: Path | str, site: str) -> dict[str, Any]:
    return load_frozen_checkpoint(Path(checkpoint_root) / str(site) / "model.sav", expected_site=site)


__all__ = ["build_npz_model_input", "load_frozen_checkpoint", "predict_frozen_s2", "predict_one_npz", "checkpoint_for_site"]
