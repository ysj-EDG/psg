"""PhysioNet 2026 官方入口的 CAISR20 v2 适配层。

本适配器只读取已经冻结的 CAISR20 TSV 和 demographics.csv，不读取 EDF、waveform
或 HDF5。它适合当前研究中的预提取特征/LOSO 验证，不宣称能从隐藏 EDF 在线提取特征。
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from feature_scaling import CAISR20_FEATURE_NAMES, FeatureScaler
from helper_code import DEMOGRAPHICS_FILE, HEADERS
from train_lr import FEATURE_VERSION, LR_CONFIG, load_caisr_dataset


ROOT = Path(__file__).resolve().parent
DEFAULT_FEATURE_TSV = ROOT / "feature_extraction/features/caisr/caisr.tsv"
MODEL_FILENAME = "model.sav"
MODEL_VERSION = "caisr20_official_adapter_v1"


def _feature_path_from_env() -> Path:
    return Path(os.environ.get("CAISR_FEATURE_TSV", str(DEFAULT_FEATURE_TSV))).expanduser().resolve()


def _record_id(record: dict[str, Any]) -> str:
    bids_folder = str(record[HEADERS["bids_folder"]])
    session = int(float(record[HEADERS["session_id"]]))
    return f"{bids_folder}_ses-{session}"


def _load_feature_index(feature_path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    frame = pd.read_csv(feature_path, sep="\t", dtype=str, keep_default_na=False)
    required = {"record_id", "feature_version", "feature_names", "values", "valid_mask"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"frozen feature TSV missing columns: {missing}")
    index: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for row in frame.itertuples(index=False):
        payload = row._asdict()
        if payload["feature_version"] != FEATURE_VERSION:
            raise ValueError(f"unexpected feature version for {payload['record_id']}")
        names = json.loads(payload["feature_names"])
        if names != list(CAISR20_FEATURE_NAMES):
            raise ValueError(f"feature order mismatch for {payload['record_id']}")
        values = json.loads(payload["values"])
        valid = json.loads(payload["valid_mask"])
        if len(values) != 20 or len(valid) != 20:
            raise ValueError(f"feature dimension mismatch for {payload['record_id']}")
        if payload["record_id"] in index:
            raise ValueError(f"duplicate frozen feature record: {payload['record_id']}")
        index[payload["record_id"]] = (
            np.asarray([np.nan if x is None else float(x) for x in values], dtype=float),
            np.asarray(valid, dtype=bool),
        )
    return index


def _feature_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def train_model(data_folder, model_folder, verbose):
    """按官方 train_model.py 接口训练 CAISR20 Elastic-net LR。"""

    data_folder = Path(data_folder).resolve()
    model_folder = Path(model_folder).resolve()
    model_folder.mkdir(parents=True, exist_ok=True)
    demographics_path = data_folder / DEMOGRAPHICS_FILE
    feature_path = _feature_path_from_env()
    if not feature_path.is_file():
        raise FileNotFoundError(feature_path)
    dataset = load_caisr_dataset(feature_path, demographics_path)
    labels = dataset.records["label"].to_numpy(dtype=int)
    if np.unique(labels).size < 2:
        raise ValueError("official training view must contain both label classes")
    preprocessor = FeatureScaler().fit(dataset.values, dataset.valid_mask)
    transformed = preprocessor.transform(dataset.values, dataset.valid_mask)
    estimator = LogisticRegression(**LR_CONFIG).fit(transformed, labels)
    payload = {
        "model_version": MODEL_VERSION,
        "feature_version": dataset.feature_version,
        "feature_names": list(CAISR20_FEATURE_NAMES),
        "feature_tsv": str(feature_path),
        "feature_sha256": _feature_sha256(feature_path),
        "threshold": 0.5,
        "lr_config": dict(LR_CONFIG),
        "record_ids": dataset.records["record_id"].tolist(),
        "model": estimator,
        "preprocessor": preprocessor,
    }
    joblib.dump(payload, model_folder / MODEL_FILENAME)
    if verbose:
        print(
            f"Trained CAISR20 LR records={len(dataset.records)} "
            f"features={dataset.values.shape[1]} model={model_folder / MODEL_FILENAME}"
        )


def load_model(model_folder, verbose):
    """按官方 run_model.py 接口加载模型和只读 CAISR20 特征索引。"""

    model_folder = Path(model_folder).resolve()
    model_path = model_folder / MODEL_FILENAME
    payload = joblib.load(model_path)
    if payload.get("model_version") != MODEL_VERSION:
        raise ValueError(f"unsupported model version: {payload.get('model_version')}")
    if payload.get("feature_version") != FEATURE_VERSION:
        raise ValueError("model feature version mismatch")
    if payload.get("feature_names") != list(CAISR20_FEATURE_NAMES):
        raise ValueError("model feature order mismatch")
    feature_path = Path(payload["feature_tsv"])
    if not feature_path.is_file():
        feature_path = _feature_path_from_env()
    if _feature_sha256(feature_path) != payload["feature_sha256"]:
        raise ValueError(f"frozen feature SHA256 mismatch: {feature_path}")
    feature_index = _load_feature_index(feature_path)
    model = {
        "estimator": payload["model"],
        "preprocessor": payload["preprocessor"],
        "feature_index": feature_index,
        "feature_names": list(CAISR20_FEATURE_NAMES),
        "feature_version": FEATURE_VERSION,
        "threshold": float(payload["threshold"]),
        "feature_tsv": feature_path,
    }
    if verbose:
        print(f"Loaded {model_path}; indexed feature records={len(feature_index)}")
    return model


def run_model(model, record, data_folder, verbose):
    """按官方接口运行单条 record；不读取患者标签。"""

    record_id = _record_id(record)
    if record_id not in model["feature_index"]:
        raise KeyError(f"record is absent from frozen CAISR20 TSV: {record_id}")
    values, valid_mask = model["feature_index"][record_id]
    transformed = model["preprocessor"].transform(values.reshape(1, -1), valid_mask.reshape(1, -1))
    probability = float(model["estimator"].predict_proba(transformed)[0, 1])
    if not np.isfinite(probability):
        raise ValueError(f"non-finite probability for {record_id}")
    probability = float(np.clip(probability, 0.0, 1.0))
    binary = bool(probability >= model["threshold"])
    if verbose:
        print(f"{record_id}: probability={probability:.17g} binary={int(binary)}")
    return binary, probability

