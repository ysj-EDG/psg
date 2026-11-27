"""Manifest 驱动的三模态 canonical 特征批处理 runner。

本模块只编排 ``extract_combined_one``：不实现 CAISR、EEG 或 ECG 公式，也不
重新实现 NPZ 序列化。每条任务在 worker 内独立打开并关闭 HDF5；结果通过
FeatureStore 的原子写入和 reopen 校验发布。默认只做小规模试运行，正式 6600
条任务必须由调用者显式指定输出目录。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
import traceback
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .feature_extractor import extract_combined_one
from .feature_profiles import EEG12_FEATURE_NAMES, SCHEMA_REVISION, schema_hash
from .feature_store import FeatureStoreError, StoredFeatureRecord, validate_record
from .sample_loader import Manifest


STATUSES = ("PENDING", "RUNNING", "SUCCESS", "SKIPPED_VERIFIED", "FAILED", "REVIEW_REQUIRED", "CANCELLED")
AUDIT_STATUSES = ("NOT_RUN", "PASS", "REVIEW_REQUIRED", "FAIL")
RECIPE_KEY = "extraction_recipe_fingerprint"
REASON_RENAME = {"no_b_exact_stage_window": "no_ecg_exact_stage_window"}
ECG_FEATURES = ("S01_N2_HF_relative", "S02_N3_HF_absolute_ms2")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _git_info() -> dict[str, Any]:
    def run(*args: str) -> str:
        try:
            return subprocess.check_output(("git", *args), text=True, stderr=subprocess.STDOUT).strip()
        except Exception as exc:
            return f"ERROR:{type(exc).__name__}:{exc}"
    status = run("status", "--short")
    return {"head": run("rev-parse", "HEAD"), "status_short": status, "dirty": bool(status)}


def _dependency_versions() -> dict[str, str]:
    result = {"python": sys.version.split()[0], "platform": platform.platform(), "hostname": socket.gethostname()}
    for name in ("numpy", "h5py"):
        try:
            module = __import__(name)
            result[name] = str(getattr(module, "__version__", "unknown"))
        except Exception:
            result[name] = "unavailable"
    return result


@dataclass(frozen=True)
class Task:
    record_id: str
    site: str
    output_path: Path


@dataclass(frozen=True)
class WorkerResult:
    record_id: str
    site: str
    output_path: str
    status: str
    duration_sec: float
    error_type: str = ""
    error_message: str = ""
    traceback_text: str = ""
    quality: Mapping[str, Any] | None = None


def _worker(manifest_path: str, task: Task) -> WorkerResult:
    started = time.monotonic()
    try:
        record = extract_combined_one(manifest_path, task.record_id, task.output_path)
        quality = quality_summary(record)
        return WorkerResult(task.record_id, task.site, str(task.output_path), "SUCCESS", time.monotonic() - started, quality=quality)
    except Exception as exc:
        return WorkerResult(
            task.record_id, task.site, str(task.output_path), "FAILED", time.monotonic() - started,
            type(exc).__name__, str(exc), traceback.format_exc(), None,
        )


def quality_summary(record: StoredFeatureRecord) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for modality in ("caisr", "eeg", "ecg"):
        block = record.modalities[modality]
        reasons: dict[str, int] = {}
        for reason in block.invalid_reason[~block.valid_mask].tolist():
            reasons[str(reason)] = reasons.get(str(reason), 0) + 1
        summary[modality] = {
            "valid_count": int(block.valid_mask.sum()),
            "total_count": int(block.valid_mask.size),
            "invalid_reason_counts": reasons,
        }
    summary["metadata_bytes"] = len(json.dumps(record.metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    ecg_meta = record.metadata.get("ecg", {})
    summary["ecg_windows"] = {key: ecg_meta.get(key) for key in (
        "ecg_exact_n2_windows", "ecg_exact_n3_windows", "ecg_exact_n2_qc_ge120_windows",
        "ecg_exact_n3_qc_ge120_windows", "ecg_exact_n2_hrv_valid_windows", "ecg_exact_n3_hrv_valid_windows",
    )}
    eeg_meta = record.metadata.get("eeg", {})
    summary["eeg_support"] = {
        key: eeg_meta.get(key) for key in ("stage_valid_seconds", "stage_qc_valid_seconds", "feature_support_seconds")
    }
    return summary


def _read_id_file(path: Path) -> list[str]:
    ids: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            ids.append(text.split("\t", 1)[0].strip())
    if len(ids) != len(set(ids)):
        raise ValueError("record-ids-file contains duplicate record_id")
    return ids


def select_tasks(manifest: Manifest, output_root: Path, record_ids_file: Path | None = None,
                 sites: Iterable[str] = (), limit: int | None = None) -> list[Task]:
    ids = _read_id_file(record_ids_file) if record_ids_file else list(manifest.records)
    selected_sites = {str(site) for site in sites if str(site)}
    tasks: list[Task] = []
    for record_id in ids:
        record = manifest.get(record_id)
        if selected_sites and record.site not in selected_sites:
            continue
        tasks.append(Task(record_id, record.site, output_root / record.site / f"{record_id}.npz"))
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        tasks = tasks[:limit]
    if len({task.record_id for task in tasks}) != len(tasks):
        raise ValueError("duplicate task identity")
    return tasks


def _lock_path(output_root: Path) -> Path:
    return output_root / ".feature_run.lock"


def acquire_lock(output_root: Path, run_id: str) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    path = _lock_path(output_root)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError as exc:
        raise RuntimeError(f"another feature runner is active or lock is stale: {path}") from exc
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump({"run_id": run_id, "pid": os.getpid(), "started": utc_now()}, handle, indent=2)
    return path


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _status_header() -> list[str]:
    return ["record_id", "site", "status", "audit_status", "start_time", "end_time", "duration_sec", "attempts", "output_path", "output_sha256", "error_type", "error_message", "error_traceback"]


def _write_status(path: Path, rows: Mapping[str, Mapping[str, Any]]) -> None:
    columns = _status_header()
    lines = ["\t".join(columns)]
    for record_id in sorted(rows):
        row = rows[record_id]
        lines.append("\t".join(str(row.get(col, "")) for col in columns))
    _atomic_write_text(path, "\n".join(lines) + "\n")


def _read_status(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["record_id"]: row for row in csv.DictReader(handle, delimiter="\t") if row.get("record_id")}


def extraction_recipe_fingerprint() -> str:
    recipe = {
        "extractor": "canonical_hdf5_to_combined_npz",
        "schema_revision": SCHEMA_REVISION,
        "caisr_feature_version": "canonical_v2_sleep_event_range",
        "eeg_feature_version": "eeg_night_candidate_v2",
        "ecg_feature_version": "stage_hrv_s2_v1",
        "eeg_selected_feature_names": list(EEG12_FEATURE_NAMES),
    }
    recipe_json = json.dumps(recipe, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(recipe_json.encode("utf-8")).hexdigest()


def _compatible_existing(path: Path, record_id: str, site: str) -> tuple[bool, str, StoredFeatureRecord | None]:
    try:
        record = validate_record(path)
    except Exception as exc:
        return False, f"invalid_existing_npz:{type(exc).__name__}:{exc}", None
    if record.record_id != record_id or record.site != site:
        return False, "identity_mismatch", record
    recipe_fingerprint = str(record.metadata.get(RECIPE_KEY, ""))
    if not recipe_fingerprint:
        return False, "missing_recipe_fingerprint", record
    if recipe_fingerprint != extraction_recipe_fingerprint():
        return False, "recipe_fingerprint_mismatch", record
    return True, "verified", record


def existing_output_state(path: Path, record_id: str, site: str) -> tuple[str, StoredFeatureRecord | None]:
    """返回 MISSING、VERIFIED 或 REVIEW_REQUIRED；绝不静默覆盖已有文件。"""
    if not path.exists():
        return "MISSING", None
    ok, detail, record = _compatible_existing(path, record_id, site)
    return ("VERIFIED", record) if ok else (f"REVIEW_REQUIRED:{detail}", record)


def compare_reasons(golden: str, new: str) -> tuple[bool, str]:
    if golden == new:
        return True, "raw_exact"
    if REASON_RENAME.get(golden) == new:
        return True, "approved_ecg_exact_rename"
    return False, "unexplained_reason_difference"


def _json_cell(value: str) -> Any:
    try:
        return json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return value


def _load_feature_tsv(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["record_id"]: row for row in csv.DictReader(handle, delimiter="\t") if row.get("record_id")}


def _close_float(a: Any, b: Any) -> bool:
    try:
        # JSON/TSV golden files commonly encode missing numeric values as
        # ``null``/``None`` while NPZ stores them as NaN. Treat these
        # representations as the same missing value for audit purposes.
        if a is None or (isinstance(a, str) and a.strip().lower() in ("", "none", "null")):
            af = float("nan")
        else:
            af = float(a)
        if b is None or (isinstance(b, str) and b.strip().lower() in ("", "none", "null")):
            bf = float("nan")
        else:
            bf = float(b)
        if np.isnan(af) and np.isnan(bf):
            return True
        return bool(np.isclose(af, bf, rtol=2e-6, atol=2e-6, equal_nan=True))
    except (TypeError, ValueError):
        return str(a) == str(b)


def _tsv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return list(reader.fieldnames or ()), list(reader)


def _require_columns(fieldnames: list[str], required: set[str], label: str) -> None:
    missing = sorted(required - set(fieldnames))
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def _optional_float(value: Any) -> float:
    text = str(value).strip()
    if text in ("", "NA", "NaN", "nan", "None"):
        return float("nan")
    return float(text)


def _optional_bool(value: Any) -> bool:
    text = str(value).strip().lower()
    if text in ("true", "1"):
        return True
    if text in ("false", "0"):
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


def _load_ecg_golden(
    path: Path | None,
    raw_path: Path | None = None,
    validity_path: Path | None = None,
    selected_path: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """加载旧 reason-diff 或正式 S2 raw/validity/selected 黄金输入。"""
    formal_paths = (raw_path, validity_path, selected_path)
    if any(item is not None for item in formal_paths):
        if not all(item is not None for item in formal_paths):
            raise ValueError("formal ECG golden requires raw, validity and selected paths together")
        raw_fields, raw_rows = _tsv_rows(raw_path)
        _require_columns(raw_fields, {"record_id", "site", "s01_n2_hf_relative", "s02_n3_hf_absolute_ms2", "s01_valid", "s02_valid", "s01_invalid_reason", "s02_invalid_reason"}, "stage_hrv_s2_raw.tsv")
        validity_fields, validity_rows = _tsv_rows(validity_path)
        _require_columns(validity_fields, {"record_id", "feature_name", "value", "valid", "invalid_reason"}, "stage_hrv_s2_validity.tsv")
        selected_fields, selected_rows = _tsv_rows(selected_path)
        _require_columns(selected_fields, {"record_id", "feature", "selected", "stage_run_id", "value", "window_start_sec", "window_end_sec"}, "stage_hrv_s2_selected_windows.tsv")
        result: dict[str, dict[str, Any]] = {}
        for row in raw_rows:
            rid = str(row.get("record_id", "")).strip()
            if not rid or rid in result:
                raise ValueError(f"duplicate or missing ECG raw record_id: {rid!r}")
            result[rid] = {
                "golden_source_kind": "formal_s2",
                "gold_s01_value": _optional_float(row["s01_n2_hf_relative"]),
                "gold_s02_value": _optional_float(row["s02_n3_hf_absolute_ms2"]),
                "gold_s01_valid": _optional_bool(row["s01_valid"]),
                "gold_s02_valid": _optional_bool(row["s02_valid"]),
                "gold_s01_reason": str(row.get("s01_invalid_reason", "")),
                "gold_s02_reason": str(row.get("s02_invalid_reason", "")),
                "selected_windows": [],
            }
        validity_by_record: dict[str, dict[str, dict[str, Any]]] = {}
        for row in validity_rows:
            rid, feature = str(row.get("record_id", "")).strip(), str(row.get("feature_name", "")).strip()
            if feature not in ECG_FEATURES:
                continue
            if rid not in result:
                raise ValueError(f"ECG validity record missing from raw: {rid}")
            validity_by_record.setdefault(rid, {})[feature] = {"value": _optional_float(row["value"]), "valid": _optional_bool(row["valid"]), "reason": str(row.get("invalid_reason", ""))}
        for rid, row in result.items():
            for feature, prefix in ((ECG_FEATURES[0], "s01"), (ECG_FEATURES[1], "s02")):
                item = validity_by_record.get(rid, {}).get(feature)
                if item is None:
                    raise ValueError(f"ECG validity missing {feature} for {rid}")
                row[f"gold_{prefix}_value"] = item["value"]
                row[f"gold_{prefix}_valid"] = item["valid"]
                row[f"gold_{prefix}_reason"] = item["reason"]
        for row in selected_rows:
            if str(row.get("selected", "")).strip().lower() not in ("true", "1"):
                continue
            rid = str(row.get("record_id", "")).strip()
            if rid not in result:
                continue
            result[rid]["selected_windows"].append({
                "feature": str(row.get("feature", "")), "stage_run_id": int(row["stage_run_id"]),
                "value": _optional_float(row["value"]), "window_start_sec": float(row["window_start_sec"]), "window_end_sec": float(row["window_end_sec"]),
            })
        return result
    return _load_feature_tsv(path)


def _compare_ecg_reasons(record_id: str, feature_name: str, golden_reason: str, actual_reason: str, registry: Mapping[tuple[str, str], Mapping[str, str]]) -> tuple[bool, str]:
    """严格比较 ECG reason，只接受逐条登记且来源指纹有效的历史差异。"""
    if golden_reason == actual_reason:
        return True, "raw_exact"
    ok, kind = compare_reasons(golden_reason, actual_reason)
    if ok and kind == "approved_ecg_exact_rename":
        return True, kind
    entry = registry.get((record_id, feature_name))
    if entry is not None and entry.get("approval_status") == "VERIFIED_LEGACY_REASON_DIFFERENCE":
        if entry.get("golden_raw_reason", "") == golden_reason and entry.get("new_raw_reason", "") == actual_reason:
            source_ok = entry.get("_source_fingerprint_valid", "")
            # 旧 registry 没有来源字段；它仍可用于历史回归，但正式修订 registry
            # 必须提供并验证 Golden 来源 SHA256。
            if source_ok in ("", "True"):
                return True, "verified_legacy_reason_difference"
            return False, "registry_source_fingerprint_mismatch"
    return False, kind


def _compare_selected_windows(record: StoredFeatureRecord, expected: list[Mapping[str, Any]]) -> bool:
    actual_by_feature = {"S01_N2_HF_relative": "selection_n2_windows", "S02_N3_HF_absolute_ms2": "selection_n3_windows"}
    actual: list[dict[str, Any]] = []
    for feature, field in actual_by_feature.items():
        for item in record.metadata.get("ecg", {}).get(field, []) or []:
            actual.append({
                "feature": feature,
                "stage_run_id": int(item.get("stage_run_id", -1)),
                "value": _optional_float(item.get("value", "nan")),
                "window_start_sec": float(item.get("start_sec", float("nan"))),
                "window_end_sec": float(item.get("end_sec", float("nan"))),
            })
    def key(item: Mapping[str, Any]) -> tuple[Any, ...]:
        return (str(item["feature"]), int(item["stage_run_id"]), float(item["window_start_sec"]), float(item["window_end_sec"]))
    actual_sorted, expected_sorted = sorted(actual, key=key), sorted((dict(item) for item in expected), key=key)
    if len(actual_sorted) != len(expected_sorted):
        return False
    for left, right in zip(actual_sorted, expected_sorted):
        if key(left) != key(right) or not _close_float(left["value"], right["value"]):
            return False
    return True


def audit_npz_against_golden(record: StoredFeatureRecord, golden: Mapping[str, Mapping[str, Mapping[str, Any]]], registry: Mapping[tuple[str, str], Mapping[str, str]]) -> tuple[str, str]:
    """比较 NPZ 与 CAISR/EEG/S2 黄金值；黄金输入绝不参与 NPZ 生成。"""
    row_caisr = golden.get("caisr", {}).get(record.record_id)
    row_eeg = golden.get("eeg", {}).get(record.record_id)
    row_ecg = golden.get("ecg", {}).get(record.record_id)
    if golden.get("caisr") and row_caisr is None:
        return "REVIEW_REQUIRED", "caisr_golden_record_missing"
    if golden.get("eeg") and row_eeg is None:
        return "REVIEW_REQUIRED", "eeg_golden_record_missing"
    if golden.get("ecg") and row_ecg is None:
        return "REVIEW_REQUIRED", "ecg_golden_record_missing"
    if row_caisr:
        names = list(_json_cell(row_caisr.get("feature_names", "[]")))
        values = list(_json_cell(row_caisr.get("values", "[]")))
        masks = list(_json_cell(row_caisr.get("valid_mask", "[]")))
        reasons = list(_json_cell(row_caisr.get("invalid_reason", "[]")))
        actual = record.modalities["caisr"]
        if not names or tuple(names) != actual.names or len(values) != len(actual.values):
            return "REVIEW_REQUIRED", "caisr_schema_mismatch"
        if not all(_close_float(a, b) for a, b in zip(actual.values, values)) or not np.array_equal(actual.valid_mask, np.asarray(masks, dtype=bool)) or list(str(x) for x in reasons) != actual.invalid_reason.tolist():
            return "REVIEW_REQUIRED", "caisr_value_mask_reason_mismatch"
    if row_eeg:
        names = list(_json_cell(row_eeg.get("feature_names", "[]")))
        values = list(_json_cell(row_eeg.get("values", "[]")))
        masks = list(_json_cell(row_eeg.get("valid_mask", "[]")))
        reasons = list(_json_cell(row_eeg.get("invalid_reason", "[]")))
        if not names:
            return "REVIEW_REQUIRED", "eeg_schema_mismatch"
        by_name = {str(n): i for i, n in enumerate(names)}
        actual = record.modalities["eeg"]
        for i, name in enumerate(actual.names):
            if name not in by_name or not _close_float(actual.values[i], values[by_name[name]]) or bool(actual.valid_mask[i]) != bool(masks[by_name[name]]) or str(actual.invalid_reason[i]) != str(reasons[by_name[name]]):
                return "REVIEW_REQUIRED", f"eeg_mismatch:{name}"
    if row_ecg:
        actual = record.modalities["ecg"]
        formal = row_ecg.get("golden_source_kind") == "formal_s2"
        approved_legacy = False
        if formal:
            for feature_name, prefix in ((ECG_FEATURES[0], "s01"), (ECG_FEATURES[1], "s02")):
                index = actual.names.index(feature_name)
                if not _close_float(actual.values[index], row_ecg[f"gold_{prefix}_value"]) or bool(actual.valid_mask[index]) != bool(row_ecg[f"gold_{prefix}_valid"]):
                    return "REVIEW_REQUIRED", f"{feature_name}:value_mask_mismatch"
                ok, kind = _compare_ecg_reasons(record.record_id, feature_name, str(row_ecg[f"gold_{prefix}_reason"]), str(actual.invalid_reason[index]), registry)
                if not ok:
                    return "REVIEW_REQUIRED", f"{feature_name}:{kind}"
                entry = registry.get((record.record_id, feature_name), {})
                approved_legacy = approved_legacy or (kind == "verified_legacy_reason_difference" and entry.get("audit_result", "PASS_WITH_APPROVED_LEGACY_REASON_DIFFERENCE") == "PASS_WITH_APPROVED_LEGACY_REASON_DIFFERENCE")
            if not _compare_selected_windows(record, row_ecg.get("selected_windows", [])):
                return "REVIEW_REQUIRED", "ecg_selected_window_mismatch"
            if approved_legacy:
                return "PASS_WITH_APPROVED_LEGACY_REASON_DIFFERENCE", "golden_value_mask_reason_selected_window_pass_with_approved_legacy_reason_difference"
            return "PASS", "golden_value_mask_reason_selected_window_pass"
        for feature_name, reason_field in ((ECG_FEATURES[0], "gold_s01_reason"), (ECG_FEATURES[1], "gold_s02_reason")):
            actual_reason = dict(zip(actual.names, actual.invalid_reason))[feature_name]
            golden_reason = row_ecg.get(reason_field, "")
            ok, kind = _compare_ecg_reasons(record.record_id, feature_name, str(golden_reason), actual_reason, registry)
            if not ok:
                return "REVIEW_REQUIRED", f"{feature_name}:{kind}"
            entry = registry.get((record.record_id, feature_name), {})
            approved_legacy = approved_legacy or (kind == "verified_legacy_reason_difference" and entry.get("audit_result", "PASS_WITH_APPROVED_LEGACY_REASON_DIFFERENCE") == "PASS_WITH_APPROVED_LEGACY_REASON_DIFFERENCE")
        return ("PASS_WITH_APPROVED_LEGACY_REASON_DIFFERENCE", "golden_value_mask_reason_pass_with_approved_legacy_reason_difference") if approved_legacy else ("PASS", "golden_value_mask_reason_pass")
    return "PASS", "golden_value_mask_reason_pass"

def _load_registry(path: Path | None) -> dict[tuple[str, str], dict[str, str]]:
    if path is None or not path.is_file():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    result: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        source_path = str(row.get("golden_source_path", "")).strip()
        expected_sha = str(row.get("golden_source_sha256", "")).strip()
        if source_path or expected_sha:
            candidate = Path(source_path)
            if not candidate.is_absolute():
                candidate = Path.cwd() / candidate
            row["_source_fingerprint_valid"] = str(candidate.is_file() and bool(expected_sha) and sha256_file(candidate) == expected_sha)
        else:
            row["_source_fingerprint_valid"] = ""
        key = (str(row.get("record_id", "")), str(row.get("feature_name", "")))
        if not all(key):
            raise ValueError("reason registry requires record_id and feature_name")
        if key in result:
            raise ValueError(f"duplicate reason registry key: {key}")
        result[key] = row
    return result


def _quality_rows(rows: Mapping[str, Mapping[str, Any]]) -> str:
    cols = ["record_id", "site", "caisr_valid", "caisr_total", "eeg_valid", "eeg_total", "ecg_valid", "ecg_total", "caisr_invalid_reason_counts", "eeg_invalid_reason_counts", "ecg_invalid_reason_counts", "ecg_windows", "eeg_support", "metadata_bytes"]
    lines = ["\t".join(cols)]
    for record_id in sorted(rows):
        row = rows[record_id]
        vals = [record_id, str(row["site"])]
        for modality in ("caisr", "eeg", "ecg"):
            vals.extend((str(row[modality]["valid_count"]), str(row[modality]["total_count"])))
        vals.extend(json.dumps(row[modality]["invalid_reason_counts"], ensure_ascii=False, sort_keys=True) for modality in ("caisr", "eeg", "ecg"))
        vals.extend((json.dumps(row.get("ecg_windows", {}), ensure_ascii=False), json.dumps(row.get("eeg_support", {}), ensure_ascii=False), str(row.get("metadata_bytes", ""))))
        lines.append("\t".join(vals))
    return "\n".join(lines) + "\n"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--record-ids-file", type=Path)
    parser.add_argument("--sites", nargs="*", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-retries", type=int, default=0)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--golden-caisr", type=Path)
    parser.add_argument("--golden-eeg", type=Path)
    parser.add_argument("--golden-ecg", type=Path, help="legacy ECG reason-diff TSV")
    parser.add_argument("--golden-ecg-raw", type=Path)
    parser.add_argument("--golden-ecg-validity", type=Path)
    parser.add_argument("--golden-ecg-selected", type=Path)
    parser.add_argument("--legacy-reason-registry", type=Path)
    parser.add_argument("--run-id", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.workers < 1 or args.workers > 12:
        raise SystemExit("--workers must be between 1 and 12 for this runner")
    if args.max_retries < 0:
        raise SystemExit("--max-retries must be non-negative")
    manifest = Manifest.from_tsv(args.manifest)
    tasks = select_tasks(manifest, args.output_root, args.record_ids_file, args.sites, args.limit)
    args.audit_root.mkdir(parents=True, exist_ok=True)
    status_path = args.audit_root / "record_status.tsv"
    quality_path = args.audit_root / "FEATURE_QUALITY_SUMMARY.tsv"
    run_id = args.run_id or datetime.now(timezone.utc).strftime("p5_p5_%Y%m%dT%H%M%SZ")
    if status_path.exists() and not args.resume:
        raise SystemExit(f"audit already exists; use --resume: {status_path}")
    lock = None
    started_at = utc_now()
    status_rows = _read_status(status_path) if args.resume else {}
    quality_rows: dict[str, Any] = {}
    if args.dry_run:
        print(json.dumps({"run_id": run_id, "tasks": len(tasks), "sites": sorted({task.site for task in tasks}), "output_root": str(args.output_root.resolve()), "dry_run": True}, ensure_ascii=False, indent=2))
        return 0
    try:
        lock = acquire_lock(args.output_root, run_id)
        golden = {
            "caisr": _load_feature_tsv(args.golden_caisr),
            "eeg": _load_feature_tsv(args.golden_eeg),
            "ecg": _load_ecg_golden(args.golden_ecg, args.golden_ecg_raw, args.golden_ecg_validity, args.golden_ecg_selected),
        }
        registry = _load_registry(args.legacy_reason_registry)
        for task in tasks:
            existing = status_rows.get(task.record_id, {})
            status = existing.get("status", "")
            if status == "REVIEW_REQUIRED" and args.resume and any(golden.values()):
                state, record = existing_output_state(task.output_path, task.record_id, task.site)
                if state == "VERIFIED" and record is not None:
                    audit_status, detail = audit_npz_against_golden(record, golden, registry)
                    updated = {**existing, "audit_status": audit_status, "output_sha256": sha256_file(task.output_path), "error_message": detail}
                    updated["status"] = "SUCCESS" if audit_status in ("PASS", "PASS_WITH_APPROVED_LEGACY_REASON_DIFFERENCE") else "REVIEW_REQUIRED"
                    status_rows[task.record_id] = updated
                    quality_rows[task.record_id] = {"site": task.site, **quality_summary(record)}
                    continue
            if status in ("SUCCESS", "SKIPPED_VERIFIED") and args.resume:
                state, record = existing_output_state(task.output_path, task.record_id, task.site)
                if state == "VERIFIED":
                    updated = {**existing, "status": "SKIPPED_VERIFIED", "audit_status": existing.get("audit_status", "NOT_RUN"), "output_sha256": sha256_file(task.output_path)}
                    if record:
                        quality_rows[task.record_id] = {"site": task.site, **quality_summary(record)}
                        if any(golden.values()):
                            audit_status, detail = audit_npz_against_golden(record, golden, registry)
                            updated["audit_status"] = audit_status
                            updated["error_message"] = detail
                            if audit_status not in ("PASS", "PASS_WITH_APPROVED_LEGACY_REASON_DIFFERENCE"):
                                updated["status"] = "REVIEW_REQUIRED"
                                updated["error_type"] = "golden_audit"
                    status_rows[task.record_id] = updated
                    continue
                status_rows[task.record_id] = {"record_id": task.record_id, "site": task.site, "status": "REVIEW_REQUIRED", "audit_status": "REVIEW_REQUIRED", "output_path": str(task.output_path), "error_type": "existing_output_conflict", "error_message": state}
                continue
            state, record = existing_output_state(task.output_path, task.record_id, task.site)
            if state == "VERIFIED":
                updated = {"record_id": task.record_id, "site": task.site, "status": "SKIPPED_VERIFIED", "audit_status": "NOT_RUN", "output_path": str(task.output_path), "attempts": "0", "output_sha256": sha256_file(task.output_path)}
                if record:
                    quality_rows[task.record_id] = {"site": task.site, **quality_summary(record)}
                    if any(golden.values()):
                        audit_status, detail = audit_npz_against_golden(record, golden, registry)
                        updated["audit_status"] = audit_status
                        updated["error_message"] = detail
                        if audit_status not in ("PASS", "PASS_WITH_APPROVED_LEGACY_REASON_DIFFERENCE"):
                            updated["status"] = "REVIEW_REQUIRED"
                            updated["error_type"] = "golden_audit"
                status_rows[task.record_id] = updated
                continue
            if state.startswith("REVIEW_REQUIRED"):
                status_rows[task.record_id] = {"record_id": task.record_id, "site": task.site, "status": "REVIEW_REQUIRED", "audit_status": "REVIEW_REQUIRED", "output_path": str(task.output_path), "attempts": "0", "error_type": "existing_output_conflict", "error_message": state}
                continue
            status_rows[task.record_id] = {"record_id": task.record_id, "site": task.site, "status": "PENDING", "audit_status": "NOT_RUN", "output_path": str(task.output_path), "attempts": "0"}
        _write_status(status_path, status_rows)
        pending = [task for task in tasks if status_rows[task.record_id].get("status") == "PENDING"]
        if pending:
            remaining = list(pending)
            for attempt_index in range(args.max_retries + 1):
                if not remaining:
                    break
                with ProcessPoolExecutor(max_workers=args.workers) as executor:
                    futures = {}
                    for task in remaining:
                        row = status_rows[task.record_id]
                        row.update(status="RUNNING", start_time=utc_now(), attempts=str(int(row.get("attempts", "0")) + 1))
                        futures[executor.submit(_worker, str(args.manifest.resolve()), task)] = task
                    _write_status(status_path, status_rows)
                    next_remaining: list[Task] = []
                    for future in as_completed(futures):
                        task = futures[future]
                        result = future.result()
                        if result.status == "FAILED" and attempt_index < args.max_retries:
                            next_remaining.append(task)
                            continue
                        row = status_rows[task.record_id]
                        row.update(status=result.status, end_time=utc_now(), duration_sec=f"{result.duration_sec:.6f}", output_path=result.output_path, error_type=result.error_type, error_message=result.error_message, error_traceback=result.traceback_text)
                        if result.status == "SUCCESS":
                            row["output_sha256"] = sha256_file(task.output_path)
                            quality_rows[task.record_id] = {"site": task.site, **(result.quality or {})}
                            if any(golden.values()):
                                stored = validate_record(task.output_path)
                                audit_status, detail = audit_npz_against_golden(stored, golden, registry)
                                row["audit_status"] = audit_status
                                if audit_status != "PASS":
                                    row["status"] = "REVIEW_REQUIRED"
                                    row["error_type"] = "golden_audit"
                                    row["error_message"] = detail
                                else:
                                    row["audit_status"] = "PASS"
                        _write_status(status_path, status_rows)
                        if result.status == "FAILED" and args.fail_fast:
                            for other in futures:
                                other.cancel()
                            next_remaining = []
                            break
                    remaining = next_remaining
        # Any never-finished task is explicitly cancelled, never silently successful.
        for task in tasks:
            row = status_rows[task.record_id]
            if row.get("status") in ("PENDING", "RUNNING"):
                row["status"] = "CANCELLED"
                row["end_time"] = utc_now()
                row["error_type"] = "runner_incomplete"
                row["error_message"] = "task did not complete"
        _write_status(status_path, status_rows)
        _atomic_write_text(quality_path, _quality_rows(quality_rows))
        counts: dict[str, int] = {}
        for row in status_rows.values():
            counts[row.get("status", "")] = counts.get(row.get("status", ""), 0) + 1
        run_info = {
            "run_id": run_id, "git": _git_info(), "schema_revision": SCHEMA_REVISION, "schema_hash": schema_hash(),
            "manifest": str(args.manifest.resolve()), "manifest_sha256": sha256_file(args.manifest),
            "output_root": str(args.output_root.resolve()), "audit_root": str(args.audit_root.resolve()),
            "workers": args.workers, "max_retries": args.max_retries, "start_time": started_at, "end_time": utc_now(),
            "completion_status": "COMPLETE" if all(row.get("status") in ("SUCCESS", "SKIPPED_VERIFIED", "REVIEW_REQUIRED") for row in status_rows.values()) else "INCOMPLETE",
            "counts": counts, "dependencies": _dependency_versions(), "task_count": len(tasks),
        }
        _atomic_write_text(args.audit_root / "run_info.json", json.dumps(_jsonable(run_info), ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        summary = ["# P5-P5 trial batch summary", "", f"- tasks: {len(tasks)}", f"- statuses: {json.dumps(counts, ensure_ascii=False, sort_keys=True)}", f"- output_root: `{args.output_root.resolve()}`", f"- release_ready: {counts.get('FAILED', 0) == 0 and counts.get('CANCELLED', 0) == 0 and counts.get('REVIEW_REQUIRED', 0) == 0}", ""]
        _atomic_write_text(args.audit_root / "batch_summary.md", "\n".join(summary))
        print(json.dumps({"run_id": run_id, "task_count": len(tasks), "counts": counts, "audit_root": str(args.audit_root.resolve())}, ensure_ascii=False, indent=2))
        return 0 if counts.get("FAILED", 0) == 0 and counts.get("CANCELLED", 0) == 0 else 2
    finally:
        if lock is not None:
            try:
                lock.unlink()
            except FileNotFoundError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
