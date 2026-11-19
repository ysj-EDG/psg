"""CAISR20 v2 的三站点 Elastic-net Logistic Regression 训练入口。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from feature_scaling import (
    CAISR20_FEATURE_NAMES,
    EEG18_FEATURE_NAMES,
    FeatureScaler,
    SCALER_VERSION,
    build_feature_scaler,
)
DEFAULT_SITES = ("I0002", "I0006", "S0001")
ROOT = Path(__file__).resolve().parent
OFFICIAL_METRIC_NAMES = (
    "Reward",
    "Age-conditioned AUROC",
    "Age-weighted AUROC",
    "AUROC",
    "AUPRC",
    "Accuracy",
    "F-measure",
)


FEATURE_VERSION = "canonical_v2_sleep_event_range"
LR_CONFIG = {
    "penalty": "elasticnet",
    "solver": "saga",
    "l1_ratio": 0.4,
    "C": 0.03,
    "class_weight": "balanced",
    "fit_intercept": True,
    "max_iter": 100000,
    "tol": 1e-4,
    "random_state": 7,
}


@dataclass
class CaisrDataset:
    records: pd.DataFrame
    values: np.ndarray
    valid_mask: np.ndarray
    feature_names: list[str]
    feature_version: str


@dataclass
class FeatureGroupDataset:
    records: pd.DataFrame
    values: np.ndarray
    valid_mask: np.ndarray
    feature_names: list[str]
    feature_versions: dict[str, str]
    feature_paths: dict[str, str]
    group_name: str


FEATURE_GROUP_COMPONENTS = {
    "E0_CAISR20": ("CAISR20",),
    "E1_EEG18": ("EEG18",),
    "E2_CAISR20_EEG18": ("CAISR20", "EEG18"),
    "A1_NREM5": ("CAISR20", "EEG18"),
    "A2_SIGMA4": ("CAISR20", "EEG18"),
    "A3_REM4": ("CAISR20", "EEG18"),
    "A4_SPATIAL5": ("CAISR20", "EEG18"),
    "B1_SIGMA4_SPATIAL5": ("CAISR20", "EEG18"),
}
FEATURE_GROUP_EEG_FEATURES = {
    "E0_CAISR20": (),
    "E1_EEG18": tuple(EEG18_FEATURE_NAMES),
    "E2_CAISR20_EEG18": tuple(EEG18_FEATURE_NAMES),
    "A1_NREM5": tuple(EEG18_FEATURE_NAMES[:5]),
    "A2_SIGMA4": tuple(EEG18_FEATURE_NAMES[5:9]),
    "A3_REM4": tuple(EEG18_FEATURE_NAMES[9:13]),
    "A4_SPATIAL5": tuple(EEG18_FEATURE_NAMES[13:18]),
    "B1_SIGMA4_SPATIAL5": tuple(EEG18_FEATURE_NAMES[5:9] + EEG18_FEATURE_NAMES[13:18]),
}
FEATURE_COMPONENT_SCHEMA = {
    "CAISR20": (list(CAISR20_FEATURE_NAMES), FEATURE_VERSION),
    "EEG18": (list(EEG18_FEATURE_NAMES), "eeg_night_candidate_v2"),
}


def _read_feature_component(
    path: str | Path, component: str,
) -> tuple[list[str], dict[str, tuple[str, np.ndarray, np.ndarray]], str]:
    expected_names, expected_version = FEATURE_COMPONENT_SCHEMA[component]
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    required = {"record_id", "site", "feature_version", "feature_names", "values", "valid_mask"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{component} feature TSV missing columns: {missing}")
    if frame["record_id"].duplicated().any():
        raise ValueError(f"{component} feature TSV has duplicate record_id")
    records: dict[str, tuple[str, np.ndarray, np.ndarray]] = {}
    for row in frame.itertuples(index=False):
        payload = row._asdict()
        if payload["feature_version"] != expected_version:
            raise ValueError(
                f"{component} unexpected feature version for {payload['record_id']}: "
                f"{payload['feature_version']}"
            )
        names = _load_json_cell(payload["feature_names"])
        if list(names) != expected_names:
            raise ValueError(f"{component} feature order mismatch for {payload['record_id']}")
        values = _load_json_cell(payload["values"])
        valid = _load_json_cell(payload["valid_mask"])
        if len(values) != len(expected_names) or len(valid) != len(expected_names):
            raise ValueError(f"{component} feature dimension mismatch for {payload['record_id']}")
        array = np.asarray([np.nan if x is None else float(x) for x in values], dtype=float)
        mask = np.asarray(valid, dtype=bool)
        records[str(payload["record_id"])] = (str(payload["site"]), array, mask)
    return expected_names, records, expected_version


def _feature_group_index(
    feature_paths: Mapping[str, str | Path], group_name: str,
) -> tuple[dict[str, tuple[str, np.ndarray, np.ndarray]], list[str], dict[str, str]]:
    components = FEATURE_GROUP_COMPONENTS.get(group_name)
    if components is None:
        raise ValueError(f"unknown feature group: {group_name}")
    component_data = {}
    names: list[str] = []
    versions: dict[str, str] = {}
    for component in components:
        if component not in feature_paths:
            raise ValueError(f"missing path for feature component {component}")
        component_names, data, version = _read_feature_component(feature_paths[component], component)
        selected_names = list(component_names)
        if component == "EEG18" and group_name in FEATURE_GROUP_EEG_FEATURES:
            selected_names = list(FEATURE_GROUP_EEG_FEATURES[group_name])
        missing_selected = sorted(set(selected_names) - set(component_names))
        if missing_selected:
            raise ValueError(f"{group_name} requests missing EEG features: {missing_selected}")
        selected_indices = [component_names.index(name) for name in selected_names]
        component_data[component] = (component_names, data, selected_indices)
        names.extend(selected_names)
        versions[component] = version
    anchor = component_data[components[0]][1]
    anchor_ids = list(anchor)
    anchor_set = set(anchor_ids)
    merged: dict[str, tuple[str, np.ndarray, np.ndarray]] = {}
    anchor_indices = component_data[components[0]][2]
    for record_id in anchor_ids:
        site, values, mask = anchor[record_id]
        parts = [values[anchor_indices]]
        masks = [mask[anchor_indices]]
        for component in components[1:]:
            data = component_data[component][1]
            if set(data) != anchor_set:
                raise ValueError(f"feature components are not aligned by the same record_id set")
            other_site, other_values, other_mask = data[record_id]
            if other_site != site:
                raise ValueError(f"site mismatch across feature components for {record_id}")
            selected_indices = component_data[component][2]
            parts.append(other_values[selected_indices]); masks.append(other_mask[selected_indices])
        merged[record_id] = (site, np.concatenate(parts), np.concatenate(masks))
    return merged, names, versions


def load_feature_group_index(
    feature_paths: Mapping[str, str | Path], group_name: str,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], list[str], dict[str, str]]:
    merged, names, versions = _feature_group_index(feature_paths, group_name)
    return {record_id: (values, mask) for record_id, (_, values, mask) in merged.items()}, names, versions


def load_feature_group_dataset(
    feature_paths: Mapping[str, str | Path],
    demographics_path: str | Path,
    *,
    group_name: str,
) -> FeatureGroupDataset:
    """Load approved feature groups and join demographics by exact record identity."""

    merged, names, versions = _feature_group_index(feature_paths, group_name)
    demographics = pd.read_csv(demographics_path)
    required_demo = {"SiteID", "BDSPPatientID", "BidsFolder", "SessionID", "Age", "Cognitive_Impairment"}
    missing_demo = sorted(required_demo - set(demographics.columns))
    if missing_demo:
        raise ValueError(f"demographics missing columns: {missing_demo}")
    demographics = demographics.copy()
    demographics["_session_id"] = pd.to_numeric(demographics["SessionID"], errors="raise").astype(int)
    if demographics.duplicated(["SiteID", "BidsFolder", "_session_id"]).any():
        raise ValueError("demographics has duplicate site/BidsFolder/session identities")
    demo_by_id = {}
    for row in demographics.to_dict(orient="records"):
        record_id = f"{row['BidsFolder']}_ses-{int(row['_session_id'])}"
        if record_id in demo_by_id:
            raise ValueError(f"duplicate demographics record_id: {record_id}")
        demo_by_id[record_id] = row
    missing_feature = sorted(set(demo_by_id) - set(merged))[:5]
    if missing_feature:
        raise ValueError(f"feature/demographics record mismatch: demo_only={missing_feature}")
    record_rows = []
    values = []; masks = []
    for record_id, (site, array, mask) in merged.items():
        if record_id not in demo_by_id:
            continue
        demo = demo_by_id[record_id]
        if str(demo["SiteID"]) != site:
            raise ValueError(f"site mismatch between feature and demographics for {record_id}")
        record_rows.append({
            "record_id": record_id, "site": site, "patient_id": str(demo["BDSPPatientID"]),
            "session_id": str(int(demo["_session_id"])), "SiteID": str(demo["SiteID"]),
            "BDSPPatientID": demo["BDSPPatientID"], "BidsFolder": str(demo["BidsFolder"]),
            "SessionID": int(demo["_session_id"]), "Age": pd.to_numeric(demo["Age"], errors="coerce"),
            "label": _coerce_label(demo["Cognitive_Impairment"]),
        })
        values.append(array); masks.append(mask)
    records = pd.DataFrame(record_rows)
    validate_identity(records)
    return FeatureGroupDataset(
        records=records, values=np.asarray(values, dtype=float), valid_mask=np.asarray(masks, dtype=bool),
        feature_names=names, feature_versions=versions,
        feature_paths={key: str(Path(value).resolve()) for key, value in feature_paths.items() if key in FEATURE_GROUP_COMPONENTS[group_name]},
        group_name=group_name,
    )


def _load_json_cell(value: Any) -> Any:
    if not isinstance(value, str):
        raise ValueError(f"expected JSON text, got {type(value).__name__}")
    return json.loads(value)


def _coerce_label(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, str):
        value = value.strip().lower()
        if value in {"true", "1"}:
            return 1
        if value in {"false", "0"}:
            return 0
    if value in (0, 1):
        return int(value)
    raise ValueError(f"Cognitive_Impairment is not binary: {value!r}")


def _parse_record_id(record_id: str) -> tuple[str, int]:
    if "_ses-" not in record_id:
        raise ValueError(f"record_id lacks _ses- suffix: {record_id}")
    bids_folder, session = record_id.rsplit("_ses-", 1)
    try:
        return bids_folder, int(session)
    except ValueError as exc:
        raise ValueError(f"record_id session is not an integer: {record_id}") from exc


def load_caisr_dataset(feature_path: str | Path, demographics_path: str | Path) -> CaisrDataset:
    """读取冻结 CAISR TSV 并与 demographics 做精确 site/BidsFolder/session join。"""

    feature_frame = pd.read_csv(feature_path, sep="\t", dtype=str, keep_default_na=False)
    required = {
        "record_id",
        "site",
        "annotation_source",
        "feature_version",
        "feature_names",
        "values",
        "valid_mask",
    }
    missing = sorted(required - set(feature_frame.columns))
    if missing:
        raise ValueError(f"feature TSV missing columns: {missing}")
    demographics = pd.read_csv(demographics_path)
    demo_identity_required = {"BidsFolder", "SessionID"}
    missing_demo_identity = sorted(demo_identity_required - set(demographics.columns))
    if missing_demo_identity:
        raise ValueError(f"demographics missing identity columns: {missing_demo_identity}")
    demo_record_ids = (
        demographics["BidsFolder"].astype(str)
        + "_ses-"
        + pd.to_numeric(demographics["SessionID"], errors="raise").astype(int).astype(str)
    )
    feature_ids = set(feature_frame["record_id"].astype(str))
    missing_feature_ids = sorted(set(demo_record_ids) - feature_ids)
    if missing_feature_ids:
        raise ValueError(f"feature TSV missing demographics records: {missing_feature_ids[:5]}")
    feature_frame = feature_frame[feature_frame["record_id"].isin(set(demo_record_ids))].copy()
    names: list[str] | None = None
    values: list[list[float]] = []
    valid_masks: list[list[bool]] = []
    for row in feature_frame.itertuples(index=False):
        row_dict = row._asdict()
        row_names = _load_json_cell(row_dict["feature_names"])
        if names is None:
            names = list(row_names)
        if list(row_names) != list(CAISR20_FEATURE_NAMES):
            raise ValueError(f"feature name/order mismatch for {row_dict['record_id']}")
        row_values = _load_json_cell(row_dict["values"])
        row_valid = _load_json_cell(row_dict["valid_mask"])
        if len(row_values) != len(CAISR20_FEATURE_NAMES) or len(row_valid) != len(CAISR20_FEATURE_NAMES):
            raise ValueError(f"feature dimension mismatch for {row_dict['record_id']}")
        values.append([np.nan if x is None else float(x) for x in row_values])
        valid_masks.append([bool(x) for x in row_valid])
    if names is None or names != list(CAISR20_FEATURE_NAMES):
        raise ValueError("feature TSV has no valid CAISR20 rows")
    feature_versions = set(feature_frame["feature_version"])
    if feature_versions != {FEATURE_VERSION}:
        raise ValueError(f"unexpected feature versions: {sorted(feature_versions)}")

    records = feature_frame[["record_id", "site"]].copy()
    parsed = records["record_id"].map(_parse_record_id)
    records["_bids_folder"] = parsed.map(lambda item: item[0])
    records["_session_id"] = parsed.map(lambda item: item[1])

    demo_required = {
        "SiteID",
        "BDSPPatientID",
        "BidsFolder",
        "SessionID",
        "Age",
        "Cognitive_Impairment",
    }
    missing_demo = sorted(demo_required - set(demographics.columns))
    if missing_demo:
        raise ValueError(f"demographics missing columns: {missing_demo}")
    demographics = demographics.copy()
    demographics["_session_id"] = pd.to_numeric(demographics["SessionID"], errors="raise").astype(int)
    if demographics.duplicated(["SiteID", "BidsFolder", "_session_id"]).any():
        raise ValueError("demographics has duplicate site/BidsFolder/session identities")
    joined = records.merge(
        demographics,
        left_on=["site", "_bids_folder", "_session_id"],
        right_on=["SiteID", "BidsFolder", "_session_id"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not bool((joined["_merge"] == "both").all()):
        missing_records = joined.loc[joined["_merge"] != "both", "record_id"].tolist()[:5]
        raise ValueError(f"demographics join missing records, examples: {missing_records}")
    joined["label"] = joined["Cognitive_Impairment"].map(_coerce_label)
    joined["patient_id"] = joined["BDSPPatientID"].astype(str)
    joined["session_id"] = joined["_session_id"].astype(str)
    joined["Age"] = pd.to_numeric(joined["Age"], errors="coerce")
    output_records = joined[
        [
            "record_id",
            "site",
            "patient_id",
            "session_id",
            "SiteID",
            "BDSPPatientID",
            "BidsFolder",
            "SessionID",
            "Age",
            "label",
        ]
    ].copy()
    if len(output_records) != len(feature_frame):
        raise ValueError("demographics join changed record count")
    return CaisrDataset(
        records=output_records,
        values=np.asarray(values, dtype=float),
        valid_mask=np.asarray(valid_masks, dtype=bool),
        feature_names=list(CAISR20_FEATURE_NAMES),
        feature_version=FEATURE_VERSION,
    )


@dataclass(frozen=True)
class LOSOFold:
    holdout_site: str
    train_indices: np.ndarray
    test_indices: np.ndarray


def validate_identity(rows: pd.DataFrame) -> pd.DataFrame:
    required = {"record_id", "site", "patient_id", "session_id"}
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"identity columns missing: {missing}")
    if rows["record_id"].duplicated().any():
        raise ValueError("duplicate record_id")
    sites_per_patient = rows.groupby("patient_id", dropna=False)["site"].nunique()
    if bool((sites_per_patient > 1).any()):
        raise ValueError("patient appears in multiple sites")
    if rows[["record_id", "site", "patient_id"]].isna().any().any():
        raise ValueError("record/site/patient identity fields cannot be null")
    return rows


def make_loso_folds(rows: pd.DataFrame, sites: tuple[str, ...] = DEFAULT_SITES) -> list[LOSOFold]:
    frame = validate_identity(rows)
    present = set(frame["site"].astype(str))
    missing = sorted(set(sites) - present)
    if missing:
        raise ValueError(f"requested LOSO sites absent from data: {missing}")
    site_values = frame["site"].astype(str).to_numpy()
    folds = []
    for holdout_site in sites:
        test_indices = np.flatnonzero(site_values == holdout_site).astype(int)
        train_indices = np.flatnonzero(site_values != holdout_site).astype(int)
        train_patients = set(frame.iloc[train_indices]["patient_id"].astype(str))
        test_patients = set(frame.iloc[test_indices]["patient_id"].astype(str))
        if train_patients & test_patients:
            raise ValueError(f"patient leakage for holdout {holdout_site}")
        folds.append(LOSOFold(holdout_site, train_indices, test_indices))
    return folds


def _official_view(records: pd.DataFrame) -> pd.DataFrame:
    return records[
        ["SiteID", "BDSPPatientID", "BidsFolder", "SessionID", "Age", "label"]
    ].rename(columns={"label": "Cognitive_Impairment"})


def _run_official(script: str, args: list[str], env: dict[str, str]) -> None:
    subprocess.run([sys.executable, str(ROOT / script), *args], cwd=ROOT, env=env, check=True)


def _parse_score(path: Path) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        name, value = line.split(":", 1)
        metrics[name.strip()] = float(value.strip())
    missing = [name for name in OFFICIAL_METRIC_NAMES if name not in metrics]
    if missing:
        raise ValueError(f"official score missing metrics: {missing}")
    return metrics


def _load_reference_summary(reference_output: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    root = Path(reference_output).resolve()
    summary_path = root / "summary.tsv"
    info_path = root / "run_info.json"
    if not summary_path.is_file() or not info_path.is_file():
        raise FileNotFoundError(f"reference experiment requires {summary_path} and {info_path}")
    frames = [pd.read_csv(summary_path, sep="\t")]
    # B1 reports use both the historical E0/E2 summary and the prior A1-A4 summary.
    sibling_paths = [root.parent / "caisr_eeg", root.parent / "caisr_eeg_ablation"]
    for sibling in sibling_paths:
        sibling_summary = sibling / "summary.tsv"
        if sibling_summary.is_file() and sibling_summary.resolve() != summary_path.resolve():
            frames.append(pd.read_csv(sibling_summary, sep="\t"))
    frame = pd.concat(frames, ignore_index=True).drop_duplicates(["experiment", "scope", "site"], keep="first")
    required_groups = {"E0_CAISR20", "E2_CAISR20_EEG18"}
    if not required_groups.issubset(set(frame["experiment"].astype(str))):
        raise ValueError("reference experiment must contain E0_CAISR20 and E2_CAISR20_EEG18")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    versions = info.get("feature_groups", {})
    if versions.get("E0_CAISR20", {}).get("dimensions") != 20:
        raise ValueError("reference E0 configuration is not 20-dimensional CAISR20")
    if versions.get("E2_CAISR20_EEG18", {}).get("dimensions") != 38:
        raise ValueError("reference E2 configuration is not 38-dimensional CAISR20+EEG18")
    return frame, {"root": str(root), "summary_path": str(summary_path), "info_path": str(info_path), "info": info}


def _reference_metric(frame: pd.DataFrame, group: str, scope: str, site: str, metric: str) -> float:
    rows = frame[(frame["experiment"] == group) & (frame["scope"] == scope) & (frame["site"] == site)]
    if len(rows) != 1:
        raise ValueError(f"reference summary missing unique row for {group}/{scope}/{site}")
    return float(rows.iloc[0][metric])


def _current_metric(result: dict[str, Any], scope: str, site: str, metric: str) -> float:
    if scope == "holdout":
        rows = [row for row in result["folds"] if row["holdout_site"] == site]
        if len(rows) != 1:
            raise ValueError(f"current result missing unique row for {scope}/{site}")
        return float(rows[0][metric])
    return float(result["macro" if scope == "macro" else "worst_site"][metric])


def run_loso_experiment(
    feature_path: str | Path = "feature_extraction/features/caisr/caisr.tsv",
    demographics_path: str | Path = "/database2/physionet2026_kaggle/data2/demographics.csv",
    output_dir: str | Path = "output/caisr",
) -> dict[str, Any]:
    dataset = load_caisr_dataset(feature_path, demographics_path)
    folds = make_loso_folds(dataset.records, sites=DEFAULT_SITES)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    if any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_root}")
    feature_hash = hashlib.sha256(Path(feature_path).read_bytes()).hexdigest()
    fold_metrics: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="caisr_official_loso_") as temp_name:
        temp_root = Path(temp_name)
        env = os.environ.copy()
        env.pop("PYTHONNOUSERSITE", None)
        env["CAISR_FEATURE_TSV"] = str(Path(feature_path).resolve())
        for fold in folds:
            fold_dir = output_root / fold.holdout_site
            fold_dir.mkdir(parents=True, exist_ok=True)
            train_data = temp_root / f"{fold.holdout_site}_train"
            test_data = temp_root / f"{fold.holdout_site}_test"
            model_dir = temp_root / f"{fold.holdout_site}_model"
            train_data.mkdir()
            test_data.mkdir()
            model_dir.mkdir()
            train_view = _official_view(dataset.records.iloc[fold.train_indices])
            test_view = _official_view(dataset.records.iloc[fold.test_indices])
            train_view.to_csv(train_data / "demographics.csv", index=False)
            test_view.to_csv(test_data / "demographics.csv", index=False)
            _run_official("train_model.py", ["-d", str(train_data), "-m", str(model_dir)], env)
            prediction_dir = fold_dir / "predictions"
            _run_official(
                "run_model.py",
                ["-d", str(test_data), "-m", str(model_dir), "-o", str(prediction_dir)],
                env,
            )
            shutil.copy2(model_dir / "model.sav", fold_dir / "model.sav")
            labels_path = temp_root / f"{fold.holdout_site}_labels.csv"
            prevalence_path = temp_root / f"{fold.holdout_site}_prevalence.csv"
            train_view.drop(columns=["BidsFolder", "SessionID"]).to_csv(prevalence_path, index=False)
            test_view.drop(columns=["BidsFolder", "SessionID"]).to_csv(labels_path, index=False)
            _run_official(
                "evaluate_model.py",
                [
                    "-d", str(labels_path),
                    "-o", str(prediction_dir / "demographics.csv"),
                    "-p", str(prevalence_path),
                    "-s", str(fold_dir / "score.txt"),
                    "-t", str(fold_dir / "table.tsv"),
                ],
                env,
            )
            metrics = _parse_score(fold_dir / "score.txt")
            test_labels = test_view["Cognitive_Impairment"].astype(int).to_numpy()
            train_labels = train_view["Cognitive_Impairment"].astype(int).to_numpy()
            fold_metrics.append(
                {
                    "holdout_site": fold.holdout_site,
                    "train_records": int(len(train_view)),
                    "test_records": int(len(test_view)),
                    "train_positive": int(train_labels.sum()),
                    "train_negative": int((train_labels == 0).sum()),
                    "test_positive": int(test_labels.sum()),
                    "test_negative": int((test_labels == 0).sum()),
                    **metrics,
                }
            )

    metric_names = list(OFFICIAL_METRIC_NAMES)
    macro = {name: float(np.mean([row[name] for row in fold_metrics])) for name in metric_names}
    worst = {name: float(np.min([row[name] for row in fold_metrics])) for name in metric_names}
    with (output_root / "summary.tsv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["scope", "site"] + metric_names
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in fold_metrics:
            writer.writerow({"scope": "holdout", "site": row["holdout_site"], **{name: row[name] for name in metric_names}})
        writer.writerow({"scope": "macro", "site": "ALL", **macro})
        writer.writerow({"scope": "worst_site", "site": "MIN", **worst})
    run_info = {
        "feature_path": str(Path(feature_path).resolve()),
        "feature_sha256": feature_hash,
        "feature_version": dataset.feature_version,
        "feature_names": list(CAISR20_FEATURE_NAMES),
        "scaler_version": SCALER_VERSION,
        "lr_config": LR_CONFIG,
        "random_seed": 7,
        "fold_count": len(folds),
        "records": len(dataset.records),
        "official_script_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in ("train_model.py", "run_model.py", "evaluate_model.py", "helper_code.py")
        },
    }
    (output_root / "run_info.json").write_text(json.dumps(run_info, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "fold_count": len(folds),
        "record_count": len(dataset.records),
        "feature_count": int(dataset.values.shape[1]),
        "macro": macro,
        "worst_site": worst,
        "output_dir": str(output_root),
    }




def _validate_prediction_output(prediction_path: Path, test_view: pd.DataFrame) -> dict[str, Any]:
    """Check complete, unique and bounded official predictions before scoring."""

    predictions = pd.read_csv(prediction_path)
    required = {
        "SiteID", "BDSPPatientID", "BidsFolder", "SessionID",
        "Cognitive_Impairment", "Cognitive_Impairment_Probability",
    }
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"official predictions missing columns: {missing}")
    key_columns = ["SiteID", "BDSPPatientID", "BidsFolder", "SessionID"]
    if predictions.duplicated(key_columns).any():
        raise ValueError("official predictions contain duplicate patient/session identities")
    expected = set(map(tuple, test_view[key_columns].astype(str).to_numpy()))
    actual = set(map(tuple, predictions[key_columns].astype(str).to_numpy()))
    if expected != actual:
        raise ValueError(
            f"prediction coverage mismatch: expected_only={list(expected - actual)[:3]}, "
            f"unexpected={list(actual - expected)[:3]}"
        )
    probabilities = pd.to_numeric(
        predictions["Cognitive_Impairment_Probability"], errors="coerce"
    ).to_numpy(dtype=float)
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("official predictions contain non-finite or out-of-range probabilities")
    return {
        "records": int(len(predictions)),
        "unique_identities": int(len(actual)),
        "probability_min": float(np.min(probabilities)),
        "probability_max": float(np.max(probabilities)),
    }


def _age_conditioned_pairs(test_view: pd.DataFrame, prediction_path: Path) -> int:
    predictions = pd.read_csv(prediction_path)
    merged = test_view[["SiteID", "BDSPPatientID", "BidsFolder", "SessionID", "Age", "Cognitive_Impairment"]].merge(
        predictions[["SiteID", "BDSPPatientID", "BidsFolder", "SessionID", "Cognitive_Impairment_Probability"]],
        on=["SiteID", "BDSPPatientID", "BidsFolder", "SessionID"],
        how="inner",
        validate="one_to_one",
    )
    ages = pd.to_numeric(merged["Age"], errors="coerce").to_numpy(dtype=float)
    labels = merged["Cognitive_Impairment"].to_numpy(dtype=int)
    pairs = 0
    for i in range(len(merged)):
        if labels[i] != 1 or not np.isfinite(ages[i]):
            continue
        for j in range(len(merged)):
            if labels[j] == 0 and np.isfinite(ages[j]) and abs(ages[i] - ages[j]) <= 2:
                pairs += 1
    return int(pairs)


def _compare_e0_predictions(new_root: Path, old_root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"available": old_root.is_dir(), "passed": True, "sites": {}}
    if not result["available"]:
        result["passed"] = False
        result["reason"] = "old E0 output directory is absent"
        return result
    for site in DEFAULT_SITES:
        new_path = new_root / site / "predictions" / "demographics.csv"
        old_path = old_root / site / "predictions" / "demographics.csv"
        if not new_path.is_file() or not old_path.is_file():
            result["passed"] = False
            result["sites"][site] = {"passed": False, "reason": "prediction file missing"}
            continue
        key = ["SiteID", "BDSPPatientID", "BidsFolder", "SessionID"]
        new = pd.read_csv(new_path)
        old = pd.read_csv(old_path)
        new = new[key + ["Cognitive_Impairment_Probability"]].copy()
        old = old[key + ["Cognitive_Impairment_Probability"]].copy()
        new = new.sort_values(key).reset_index(drop=True)
        old = old.sort_values(key).reset_index(drop=True)
        if not new[key].astype(str).equals(old[key].astype(str)):
            result["passed"] = False
            result["sites"][site] = {"passed": False, "reason": "identity mismatch"}
            continue
        new_prob = pd.to_numeric(new["Cognitive_Impairment_Probability"], errors="coerce").to_numpy(float)
        old_prob = pd.to_numeric(old["Cognitive_Impairment_Probability"], errors="coerce").to_numpy(float)
        diff = np.abs(new_prob - old_prob)
        max_diff = float(np.nanmax(diff)) if len(diff) else 0.0
        site_ok = bool(np.allclose(new_prob, old_prob, rtol=0.0, atol=1e-12, equal_nan=False))
        result["passed"] = result["passed"] and site_ok
        result["sites"][site] = {
            "passed": site_ok,
            "records": int(len(new)),
            "max_abs_probability_difference": max_diff,
        }
    return result


def _run_feature_group_loso(
    dataset: FeatureGroupDataset,
    group_name: str,
    output_root: Path,
    temp_root: Path,
    base_env: dict[str, str],
) -> dict[str, Any]:
    folds = make_loso_folds(dataset.records, sites=DEFAULT_SITES)
    fold_metrics: list[dict[str, Any]] = []
    output_root.mkdir(parents=True, exist_ok=False)
    for fold in folds:
        fold_dir = output_root / fold.holdout_site
        fold_dir.mkdir(parents=True, exist_ok=False)
        train_data = temp_root / f"{group_name}_{fold.holdout_site}_train"
        test_data = temp_root / f"{group_name}_{fold.holdout_site}_test"
        model_dir = temp_root / f"{group_name}_{fold.holdout_site}_model"
        train_data.mkdir(); test_data.mkdir(); model_dir.mkdir()
        train_view = _official_view(dataset.records.iloc[fold.train_indices])
        test_view = _official_view(dataset.records.iloc[fold.test_indices])
        train_view.to_csv(train_data / "demographics.csv", index=False)
        test_view.to_csv(test_data / "demographics.csv", index=False)
        env = base_env.copy()
        env["PSG_FEATURE_GROUP"] = group_name
        env["CAISR_FEATURE_TSV"] = dataset.feature_paths.get("CAISR20", "")
        env["EEG_FEATURE_TSV"] = dataset.feature_paths.get("EEG18", "")
        _run_official("train_model.py", ["-d", str(train_data), "-m", str(model_dir)], env)
        prediction_dir = fold_dir / "predictions"
        _run_official(
            "run_model.py",
            ["-d", str(test_data), "-m", str(model_dir), "-o", str(prediction_dir)],
            env,
        )
        prediction_path = prediction_dir / "demographics.csv"
        prediction_check = _validate_prediction_output(prediction_path, test_view)
        import joblib
        trained_payload = joblib.load(model_dir / "model.sav")
        shutil.copy2(model_dir / "model.sav", fold_dir / "model.sav")
        labels_path = temp_root / f"{group_name}_{fold.holdout_site}_labels.csv"
        prevalence_path = temp_root / f"{group_name}_{fold.holdout_site}_prevalence.csv"
        test_view.to_csv(labels_path, index=False)
        train_view.drop(columns=["BidsFolder", "SessionID"]).to_csv(prevalence_path, index=False)
        _run_official(
            "evaluate_model.py",
            ["-d", str(labels_path), "-o", str(prediction_path), "-p", str(prevalence_path),
             "-s", str(fold_dir / "score.txt"), "-t", str(fold_dir / "table.tsv")],
            env,
        )
        metrics = _parse_score(fold_dir / "score.txt")
        train_labels = train_view["Cognitive_Impairment"].astype(int).to_numpy()
        test_labels = test_view["Cognitive_Impairment"].astype(int).to_numpy()
        fold_metrics.append({
            "holdout_site": fold.holdout_site,
            "train_records": int(len(train_view)), "test_records": int(len(test_view)),
            "train_positive": int(train_labels.sum()), "train_negative": int((train_labels == 0).sum()),
            "test_positive": int(test_labels.sum()), "test_negative": int((test_labels == 0).sum()),
            "age_conditioned_pairs": _age_conditioned_pairs(test_view, prediction_path),
            "convergence_warnings": list(trained_payload.get("convergence_warnings", [])),
            "prediction_check": prediction_check,
            **metrics,
        })
    macro = {name: float(np.mean([r[name] for r in fold_metrics])) for name in OFFICIAL_METRIC_NAMES}
    worst = {name: float(np.min([r[name] for r in fold_metrics])) for name in OFFICIAL_METRIC_NAMES}
    return {"folds": fold_metrics, "macro": macro, "worst_site": worst, "feature_count": len(dataset.feature_names)}


def _write_feature_group_summary(output_root: Path, results: dict[str, dict[str, Any]]) -> None:
    fields = ["experiment", "scope", "site", "train_records", "test_records", "age_conditioned_pairs", "convergence_warning_count"] + list(OFFICIAL_METRIC_NAMES)
    with (output_root / "summary.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for group, result in results.items():
            for fold in result["folds"]:
                writer.writerow({"experiment": group, "scope": "holdout", "site": fold["holdout_site"],
                                 "train_records": fold["train_records"], "test_records": fold["test_records"],
                                 "age_conditioned_pairs": fold["age_conditioned_pairs"],
                                 "convergence_warning_count": len(fold["convergence_warnings"]),
                                 **{name: fold[name] for name in OFFICIAL_METRIC_NAMES}})
            for scope, site, metrics in (("macro", "ALL", result["macro"]), ("worst_site", "MIN", result["worst_site"])):
                writer.writerow({"experiment": group, "scope": scope, "site": site,
                                 "train_records": "", "test_records": "", "age_conditioned_pairs": "",
                                 "convergence_warning_count": "", **metrics})


def _write_feature_group_report(
    output_root: Path, results: dict[str, dict[str, Any]], regression: dict[str, Any],
    datasets: dict[str, FeatureGroupDataset], reference_summary_path: str | Path | None = None,
) -> None:
    ac_name = "Age-conditioned AUROC"
    lines = ["CAISR20 + EEG Night18 FEATURE ABLATION THREE-SITE LOSO EXPERIMENT", ""]
    lines.append("All scores below come from the official evaluate_model.py; no EDF/HDF5/waveform was read.")
    lines.append("")
    for group, result in results.items():
        lines.append(f"[{group}] dimensions={result['feature_count']}")
        for fold in result["folds"]:
            lines.append(
                f"  {fold['holdout_site']}: AC-AUROC={fold[ac_name]:.6f}; AUROC={fold['AUROC']:.6f}; "
                f"AUPRC={fold['AUPRC']:.6f}; age_pairs={fold['age_conditioned_pairs']}; "
                f"convergence_warnings={len(fold['convergence_warnings'])}"
            )
        lines.append(f"  Macro AC-AUROC={result['macro'][ac_name]:.6f}; Worst-site AC-AUROC={result['worst_site'][ac_name]:.6f}")
        lines.append("")

    lines.append("E0/E2 historical reference")
    lines.append(json.dumps(regression, indent=2, sort_keys=True))
    if reference_summary_path is not None:
        reference_frame, _ = _load_reference_summary(Path(reference_summary_path).parent)
        lines.append("")
        lines.append("Complete AC-AUROC comparison")
        lines.append("group\tI0002\tI0006\tS0001\tMacro\tWorst")
        comparison_groups = ["E0_CAISR20", "E2_CAISR20_EEG18", "A1_NREM5", "A2_SIGMA4", "A3_REM4", "A4_SPATIAL5"]
        for group in comparison_groups:
            if group in results:
                values = [_current_metric(results[group], "holdout", site, ac_name) for site in DEFAULT_SITES]
                macro = _current_metric(results[group], "macro", "ALL", ac_name)
                worst = _current_metric(results[group], "worst_site", "MIN", ac_name)
            else:
                values = [_reference_metric(reference_frame, group, "holdout", site, ac_name) for site in DEFAULT_SITES]
                macro = _reference_metric(reference_frame, group, "macro", "ALL", ac_name)
                worst = _reference_metric(reference_frame, group, "worst_site", "MIN", ac_name)
            lines.append("\t".join([group] + [f"{value:.6f}" for value in (*values, macro, worst)]))
        lines.append("")
        lines.append("DELTA AC-AUROC VS E0")
        lines.append("group\tI0002\tI0006\tS0001\tMacro\tWorst")
        e0_values = [_reference_metric(reference_frame, "E0_CAISR20", "holdout", site, ac_name) for site in DEFAULT_SITES]
        e0_values += [_reference_metric(reference_frame, "E0_CAISR20", "macro", "ALL", ac_name), _reference_metric(reference_frame, "E0_CAISR20", "worst_site", "MIN", ac_name)]
        lines.extend(_format_delta_row(group, values, e0_values, results, reference_frame, ac_name) for group in ("A1_NREM5", "A2_SIGMA4", "A3_REM4", "A4_SPATIAL5"))
        lines.append("")
        lines.append("DELTA AC-AUROC VS E2")
        lines.append("group\tI0002\tI0006\tS0001\tMacro\tWorst")
        e2_values = [_reference_metric(reference_frame, "E2_CAISR20_EEG18", "holdout", site, ac_name) for site in DEFAULT_SITES]
        e2_values += [_reference_metric(reference_frame, "E2_CAISR20_EEG18", "macro", "ALL", ac_name), _reference_metric(reference_frame, "E2_CAISR20_EEG18", "worst_site", "MIN", ac_name)]
        lines.extend(_format_delta_row(group, values, e2_values, results, reference_frame, ac_name) for group in ("A1_NREM5", "A2_SIGMA4", "A3_REM4", "A4_SPATIAL5"))
        lines.append("")
        lines.append("Macro secondary metrics")
        lines.append("group\tAUROC\tAUPRC\tReward")
        for group in comparison_groups:
            if group in results:
                values = [_current_metric(results[group], "macro", "ALL", metric) for metric in ("AUROC", "AUPRC", "Reward")]
            else:
                values = [_reference_metric(reference_frame, group, "macro", "ALL", metric) for metric in ("AUROC", "AUPRC", "Reward")]
            lines.append("\t".join([group] + [f"{value:.6f}" for value in values]))

    if all(group in results for group in ("E0_CAISR20", "E1_EEG18", "E2_CAISR20_EEG18")):
        eeg = datasets["E1_EEG18"]
        for site in DEFAULT_SITES:
            idx = np.flatnonzero(eeg.records["site"].astype(str).to_numpy() == site)
            mask = eeg.valid_mask[idx]
            lines.append(f"EEG missingness {site}: records={len(idx)}; valid_cell_rate={float(mask.mean()):.6f}; all_EEG_invalid_records={int(np.sum(~mask.any(axis=1)))}")
        warning_total = sum(len(f["convergence_warnings"]) for g in results.values() for f in g["folds"])
        lines.append(f"Total convergence warnings across all folds/groups: {warning_total}")
    (output_root / "ablation_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_delta_row(
    group: str, baseline: list[float], reference: list[float],
    results: dict[str, dict[str, Any]], reference_frame: pd.DataFrame, metric: str,
) -> str:
    if group in results:
        current = [_current_metric(results[group], "holdout", site, metric) for site in DEFAULT_SITES]
        current += [_current_metric(results[group], "macro", "ALL", metric), _current_metric(results[group], "worst_site", "MIN", metric)]
    else:
        current = [_reference_metric(reference_frame, group, "holdout", site, metric) for site in DEFAULT_SITES]
        current += [_reference_metric(reference_frame, group, "macro", "ALL", metric), _reference_metric(reference_frame, group, "worst_site", "MIN", metric)]
    # The second argument is retained as the explicit baseline vector for readability.
    deltas = [value - base for value, base in zip(current, reference)]
    return "\t".join([group] + [f"{value:+.6f}" for value in deltas])


def run_feature_group_experiments(
    caisr_feature_path: str | Path = "feature_extraction/features/caisr/caisr.tsv",
    eeg_feature_path: str | Path = "feature_extraction/features/eeg/eeg_night.tsv",
    demographics_path: str | Path = "/database2/physionet2026_kaggle/data2/demographics.csv",
    output_dir: str | Path = "output/caisr_eeg",
    old_e0_output: str | Path = "output/caisr",
    groups: tuple[str, ...] = ("E0_CAISR20", "E1_EEG18", "E2_CAISR20_EEG18"),
    reference_output: str | Path = "output/caisr_eeg",
) -> dict[str, Any]:
    """Run the fixed three-group official LOSO experiment without touching old output."""

    output_root = Path(output_dir)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    paths = {"CAISR20": Path(caisr_feature_path).resolve(), "EEG18": Path(eeg_feature_path).resolve()}
    datasets = {
        group: load_feature_group_dataset(paths, demographics_path, group_name=group)
        for group in groups
    }
    anchor = datasets[groups[0]].records[["record_id", "site", "patient_id", "label"]]
    for group, dataset in datasets.items():
        other = dataset.records[["record_id", "site", "patient_id", "label"]]
        if not anchor.reset_index(drop=True).equals(other.reset_index(drop=True)):
            raise ValueError(f"record/site/patient/label alignment mismatch for {group}")
    base_env = os.environ.copy()
    # helper_code.py is an official file whose edfio dependency is installed in
    # the environment's user site; match the existing official runner behavior.
    base_env.pop("PYTHONNOUSERSITE", None)
    results = {}
    reference_summary_path = None
    if "E0_CAISR20" in groups:
        reference_summary = None
    else:
        reference_frame, reference_meta = _load_reference_summary(reference_output)
        reference_summary_path = reference_meta["summary_path"]
        reference_summary = {
            "available": True, "passed": True, "mode": "historical_read_only",
            "summary_path": reference_meta["summary_path"], "groups": ["E0_CAISR20", "E2_CAISR20_EEG18"],
        }
    with tempfile.TemporaryDirectory(prefix="caisr_eeg_official_loso_") as temp_name:
        temp_root = Path(temp_name)
        if "E0_CAISR20" in groups:
            results["E0_CAISR20"] = _run_feature_group_loso(datasets["E0_CAISR20"], "E0_CAISR20", output_root / "E0_CAISR20", temp_root, base_env)
            regression = _compare_e0_predictions(output_root / "E0_CAISR20", Path(old_e0_output))
            if not regression.get("passed", False):
                raise RuntimeError(f"E0 regression failed; stopping before other groups: {regression}")
        else:
            regression = reference_summary
        for group in groups:
            if group == "E0_CAISR20":
                continue
            results[group] = _run_feature_group_loso(datasets[group], group, output_root / group, temp_root, base_env)
    _write_feature_group_summary(output_root, results)
    _write_feature_group_report(output_root, results, regression, datasets, reference_summary_path)
    run_info = {
        "feature_groups": {group: {"dimensions": len(datasets[group].feature_names), "feature_names": datasets[group].feature_names,
                                    "feature_versions": datasets[group].feature_versions,
                                    "feature_sha256": {key: hashlib.sha256(Path(path).read_bytes()).hexdigest() for key, path in paths.items() if key in FEATURE_GROUP_COMPONENTS[group]}}
                         for group in groups},
        "feature_paths": {key: str(path) for key, path in paths.items()},
        "demographics_path": str(Path(demographics_path).resolve()),
        "sites": list(DEFAULT_SITES), "groups": list(groups), "random_seed": 7,
        "lr_config": LR_CONFIG, "official_script_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in ("train_model.py", "run_model.py", "evaluate_model.py", "helper_code.py")},
        "e0_regression": regression,
        "reference_output": str(Path(reference_output).resolve()),
        "reference_summary_sha256": hashlib.sha256((Path(reference_output) / "summary.tsv").read_bytes()).hexdigest() if (Path(reference_output) / "summary.tsv").is_file() else None,
        "software": {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__},
    }
    (output_root / "run_info.json").write_text(json.dumps(run_info, indent=2, sort_keys=True), encoding="utf-8")
    return {"output_dir": str(output_root), "groups": results, "e0_regression": regression}

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run CAISR20 v2 Elastic-net LR LOSO")
    parser.add_argument("--features", default="feature_extraction/features/caisr/caisr.tsv")
    parser.add_argument(
        "--demographics",
        default="/database2/physionet2026_kaggle/data2/demographics.csv",
    )
    parser.add_argument("--output", default="output/caisr")
    parser.add_argument("--eeg-features", default="feature_extraction/features/eeg/eeg_night.tsv")
    parser.add_argument("--old-e0-output", default="output/caisr")
    parser.add_argument("--reference-output", default="output/caisr_eeg")
    parser.add_argument("--experiment-groups", nargs="+", choices=tuple(FEATURE_GROUP_COMPONENTS))
    args = parser.parse_args(argv)
    if args.experiment_groups:
        result = run_feature_group_experiments(
            args.features, args.eeg_features, args.demographics, args.output, args.old_e0_output,
            tuple(args.experiment_groups), args.reference_output,
        )
    else:
        result = run_loso_experiment(args.features, args.demographics, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

