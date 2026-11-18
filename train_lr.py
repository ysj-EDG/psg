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
from typing import Any

import numpy as np
import pandas as pd

from feature_scaling import CAISR20_FEATURE_NAMES, FeatureScaler, SCALER_VERSION
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run CAISR20 v2 Elastic-net LR LOSO")
    parser.add_argument("--features", default="feature_extraction/features/caisr/caisr.tsv")
    parser.add_argument(
        "--demographics",
        default="/database2/physionet2026_kaggle/data2/demographics.csv",
    )
    parser.add_argument("--output", default="output/caisr")
    args = parser.parse_args(argv)
    result = run_loso_experiment(args.features, args.demographics, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

