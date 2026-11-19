"""PhysioNet 2026 官方入口的冻结特征适配层。

只读取预先提取的 CAISR20、EEG18 TSV 和 demographics.csv，不读取 EDF、
waveform 或 HDF5。官方 train/run/evaluate 脚本保持原版不变。
"""

from __future__ import annotations

import hashlib
import json
import os
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from feature_scaling import build_feature_scaler
from helper_code import HEADERS
from train_lr import (
    FEATURE_GROUP_COMPONENTS,
    load_feature_group_dataset,
    load_feature_group_index,
)

ROOT = Path(__file__).resolve().parent
MODEL_FILENAME = "model.sav"
MODEL_VERSION = "official_frozen_feature_adapter_v2"
DEFAULT_CAISR_TSV = ROOT / "feature_extraction/features/caisr/caisr.tsv"
DEFAULT_EEG_TSV = ROOT / "feature_extraction/features/eeg/eeg_night.tsv"


def _group_name_from_env() -> str:
    group = os.environ.get("PSG_FEATURE_GROUP", "E0_CAISR20")
    if group not in FEATURE_GROUP_COMPONENTS:
        raise ValueError(f"unsupported PSG_FEATURE_GROUP: {group}")
    return group


def _feature_paths_from_env(group_name: str) -> dict[str, Path]:
    paths = {}
    if "CAISR20" in FEATURE_GROUP_COMPONENTS[group_name]:
        paths["CAISR20"] = Path(
            os.environ.get("CAISR_FEATURE_TSV", str(DEFAULT_CAISR_TSV))
        ).expanduser().resolve()
    if "EEG18" in FEATURE_GROUP_COMPONENTS[group_name]:
        paths["EEG18"] = Path(
            os.environ.get("EEG_FEATURE_TSV", str(DEFAULT_EEG_TSV))
        ).expanduser().resolve()
    return paths


def _feature_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record_id(record: dict[str, Any]) -> str:
    bids_folder = str(record[HEADERS["bids_folder"]])
    session = int(float(record[HEADERS["session_id"]]))
    return f"{bids_folder}_ses-{session}"


def train_model(data_folder, model_folder, verbose):
    """按官方 train_model.py 接口训练一个固定特征组的 Elastic-net LR。"""

    data_folder = Path(data_folder).resolve()
    model_folder = Path(model_folder).resolve()
    model_folder.mkdir(parents=True, exist_ok=True)
    group_name = _group_name_from_env()
    feature_paths = _feature_paths_from_env(group_name)
    dataset = load_feature_group_dataset(
        feature_paths, data_folder / "demographics.csv", group_name=group_name
    )
    labels = dataset.records["label"].to_numpy(dtype=int)
    if np.unique(labels).size < 2:
        raise ValueError("official training view must contain both label classes")
    preprocessor = build_feature_scaler(dataset.feature_names).fit(
        dataset.values, dataset.valid_mask
    )
    transformed = preprocessor.transform(dataset.values, dataset.valid_mask)
    from train_lr import LR_CONFIG
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        estimator = LogisticRegression(**LR_CONFIG).fit(transformed, labels)
    convergence_warnings = [str(item.message) for item in caught if issubclass(item.category, ConvergenceWarning)]
    payload = {
        "model_version": MODEL_VERSION,
        "feature_group": group_name,
        "feature_names": list(dataset.feature_names),
        "feature_versions": dict(dataset.feature_versions),
        "feature_paths": dict(dataset.feature_paths),
        "feature_sha256": {
            key: _feature_sha256(Path(path)) for key, path in dataset.feature_paths.items()
        },
        "scaler_version": preprocessor.version,
        "threshold": 0.5,
        "lr_config": dict(LR_CONFIG),
        "convergence_warnings": convergence_warnings,
        "record_ids": dataset.records["record_id"].tolist(),
        "model": estimator,
        "preprocessor": preprocessor,
    }
    joblib.dump(payload, model_folder / MODEL_FILENAME)
    if verbose:
        print(
            f"Trained {group_name} records={len(dataset.records)} "
            f"features={dataset.values.shape[1]} model={model_folder / MODEL_FILENAME}"
        )


def load_model(model_folder, verbose):
    """按官方 run_model.py 接口加载模型和冻结 TSV 索引。"""

    model_folder = Path(model_folder).resolve()
    model_path = model_folder / MODEL_FILENAME
    payload = joblib.load(model_path)
    if payload.get("model_version") != MODEL_VERSION:
        raise ValueError(f"unsupported model version: {payload.get('model_version')}")
    group_name = str(payload["feature_group"])
    if group_name not in FEATURE_GROUP_COMPONENTS:
        raise ValueError(f"unsupported model feature group: {group_name}")
    feature_paths = {key: Path(value) for key, value in payload["feature_paths"].items()}
    env_paths = _feature_paths_from_env(group_name)
    for key, path in list(feature_paths.items()):
        if not path.is_file():
            feature_paths[key] = env_paths[key]
        if _feature_sha256(feature_paths[key]) != payload["feature_sha256"][key]:
            raise ValueError(f"frozen feature SHA256 mismatch: {feature_paths[key]}")
    feature_index, names, versions = load_feature_group_index(feature_paths, group_name)
    if names != payload["feature_names"] or versions != payload["feature_versions"]:
        raise ValueError("frozen feature schema mismatch")
    model = {
        "estimator": payload["model"],
        "preprocessor": payload["preprocessor"],
        "feature_index": feature_index,
        "feature_names": names,
        "feature_versions": versions,
        "feature_group": group_name,
        "threshold": float(payload["threshold"]),
        "feature_paths": feature_paths,
    }
    if verbose:
        print(f"Loaded {model_path}; group={group_name}; indexed feature records={len(feature_index)}")
    return model


def run_model(model, record, data_folder, verbose):
    """按官方接口运行单条 record；不读取患者标签。"""

    record_id = _record_id(record)
    if record_id not in model["feature_index"]:
        raise KeyError(f"record is absent from frozen feature TSVs: {record_id}")
    values, valid_mask = model["feature_index"][record_id]
    transformed = model["preprocessor"].transform(
        values.reshape(1, -1), valid_mask.reshape(1, -1)
    )
    probability = float(model["estimator"].predict_proba(transformed)[0, 1])
    if not np.isfinite(probability):
        raise ValueError(f"non-finite probability for {record_id}")
    probability = float(np.clip(probability, 0.0, 1.0))
    binary = bool(probability >= model["threshold"])
    if verbose:
        print(f"{record_id}: probability={probability:.17g} binary={int(binary)}")
    return binary, probability
