"""Canonical v1 HDF5 保守质量审计器与正式 Manifest 发布器。

本脚本只读取 canonical HDF5、source manifest 和 demographics.csv。它不打开 EDF，
不修改 HDF5，也不计算特征、修复波形或重算统计阈值。默认执行结构、metadata、
quality mask、CAISR/SpO2 contract 检查；``--deep-signal`` 才按 epoch chunk 读取
waveform，检查 Inf 和 structural NaN padding。
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import h5py
import numpy as np


SCHEMA_PROFILE = "canonical_v1"
RULESET_VERSION = "canonical_v1_conservative_1.0"
FORMAL_COLUMNS = (
    "record_id", "site", "subject_id", "session_id", "cache_h5_path",
    "demographics_csv_path", "psg_edf_path", "caisr_edf_path",
)
SOURCE_COLUMNS = ("index", "site", "record_id", "psg_path", "caisr_path", "psg_size_bytes", "caisr_size_bytes", "status")
RECORD_RE = re.compile(r"^sub-(?P<subject>[^_]+)_ses-(?P<session>[^_]+)$")

MODALITY_SPECS: dict[str, tuple[int, int]] = {
    "eeg": (6, 3840),
    "eog": (2, 3840),
    "ecg": (1, 6000),
    "emg": (3, 6000),
    "resp": (7, 750),
    "spo2": (1, 30),
}
CANONICAL_CHANNELS: dict[str, tuple[str, ...]] = {
    "eeg": ("EEG_F3", "EEG_F4", "EEG_C3", "EEG_C4", "EEG_O1", "EEG_O2"),
    "eog": ("EOG_E1", "EOG_E2"),
    "ecg": ("ECG",),
    "emg": ("CHIN_EMG", "LEG_EMG_LEFT", "LEG_EMG_RIGHT"),
    "resp": ("NASAL_PRESSURE", "THERMAL_AIRFLOW", "AIRFLOW_UNSPECIFIED", "PAP_FLOW", "PAP_PRESSURE", "THORACIC_EFFORT", "ABDOMINAL_EFFORT"),
    "spo2": ("SPO2",),
}
TARGET_FS = {"eeg": 128.0, "eog": 128.0, "ecg": 200.0, "emg": 200.0, "resp": 25.0, "spo2": 1.0}
CAISR_CHANNELS = ("arousal_caisr", "caisr_prob_no-ar", "caisr_prob_arous", "limb_caisr", "resp_caisr", "stage_caisr", "caisr_prob_n3", "caisr_prob_n2", "caisr_prob_n1", "caisr_prob_r", "caisr_prob_w")
MANUAL_ANNOTATION_CHANNELS = ("stage_expert", "arousal_expert", "resp_expert", "limb_expert")
CAISR_DATASET_SPECS = {
    "stage": (("T",), np.dtype("int8")),
    "stage_probability": (("T", 5), np.dtype("float32")),
    "stage_probability_valid": (("T",), np.dtype("bool")),
    "arousal": (("T", 60), np.dtype("int8")),
    "arousal_probability": (("T", 60, 2), np.dtype("float32")),
    "arousal_probability_valid": (("T", 60), np.dtype("bool")),
    "respiratory": (("T", 30), np.dtype("int8")),
    "limb": (("T", 30), np.dtype("int8")),
}
REQUIRED_GROUPS = ("signals", "quality", "metadata", "annotations", "timebase")
QUALITY_NAMES = ("channel_invalid_5s", "channel_flat_5s", "channel_extreme_5s")


@dataclass(frozen=True)
class RuleSpec:
    rule_id: str
    rule_version: str
    severity: str
    applies_to: str
    check_function: str
    description: str


@dataclass(frozen=True)
class AuditEvent:
    severity: str
    rule_id: str
    record_id: str
    site: str
    reason: str
    evidence: str


@dataclass
class RecordAudit:
    record_id: str
    site: str
    cache_path: Path
    errors: list[AuditEvent] = field(default_factory=list)
    warnings: list[AuditEvent] = field(default_factory=list)
    infos: list[AuditEvent] = field(default_factory=list)
    checks: list[str] = field(default_factory=list)
    channel_present: dict[str, np.ndarray] = field(default_factory=dict)
    caisr_available: bool | None = None
    annotation_source: str | None = None
    spo2_valid_seconds: int | None = None

    def add(self, event: AuditEvent) -> None:
        if event.severity == "ERROR":
            self.errors.append(event)
        elif event.severity == "WARN":
            self.warnings.append(event)
        else:
            self.infos.append(event)


@dataclass(frozen=True)
class SourceRecord:
    record_id: str
    site: str
    subject_id: str
    session_id: str
    psg_edf_path: str
    caisr_edf_path: str

    @property
    def h5_record_path(self) -> str:
        return self.record_id


ANNOTATION_STATUSES = (
    "AUTO_CAISR_AVAILABLE",
    "MANUAL_ANNOTATION_AVAILABLE",
    "ANNOTATION_UNAVAILABLE",
    "ANNOTATION_AMBIGUOUS",
    "ANNOTATION_UNREADABLE",
)


@dataclass(frozen=True)
class AnnotationInspection:
    """单个 annotation EDF 的 header/低速标注检查结果。"""

    path: Path
    record_id: str
    status: str
    channel_names: tuple[str, ...] = ()
    sampling_rates_hz: tuple[float, ...] = ()
    sample_counts: tuple[int, ...] = ()
    physical_dimensions: tuple[str, ...] = ()
    start_datetime: str = ""
    duration_sec: float | None = None
    values_by_channel: Mapping[str, np.ndarray] = field(default_factory=dict)
    error: str = ""


@dataclass(frozen=True)
class AnnotationSourceResolution:
    """自动 CAISR / 人工 annotation / unavailable 的明确来源状态。"""

    record_id: str
    site: str
    status: str
    path: Path | None
    candidates: tuple[Path, ...] = ()
    inspection: AnnotationInspection | None = None
    reason: str = ""


RULES = (
    RuleSpec("H5_OPEN_FAILED", "1", "ERROR", "file", "audit_record", "HDF5 cannot be opened"),
    RuleSpec("H5_SCHEMA_MISMATCH", "1", "ERROR", "hdf5", "audit_record", "canonical v1 group/dataset/shape/dtype mismatch"),
    RuleSpec("RECORD_ID_MISMATCH", "1", "ERROR", "identity", "audit_record", "path, site or HDF5 identity mismatch"),
    RuleSpec("TIMEBASE_MISMATCH", "1", "ERROR", "timebase", "audit_record", "epoch index or 30-second timebase mismatch"),
    RuleSpec("CHANNEL_METADATA_CONFLICT", "1", "ERROR", "metadata", "audit_record", "canonical channel metadata mismatch"),
    RuleSpec("CAISR_CODE_INVALID", "1", "ERROR", "caisr", "audit_record", "categorical code or probability validity mismatch"),
    RuleSpec("DEMOGRAPHICS_JOIN_INVALID", "1", "ERROR", "demographics", "validate_demographics", "BidsFolder plus SessionID is not unique"),
    RuleSpec("SPO2_NO_VALID_SECONDS", "1", "WARN", "spo2", "audit_record", "present SpO2 has no valid seconds"),
    RuleSpec("CHANNEL_UNAVAILABLE", "1", "INFO", "availability", "audit_record", "canonical channel is structurally unavailable"),
    RuleSpec("SHORT_CHANNEL_SUPPORT", "1", "INFO", "availability", "audit_record", "native support is shorter than PSG timebase"),
    RuleSpec("CAISR_UNAVAILABLE", "1", "INFO", "caisr", "audit_record", "CAISR file is unavailable"),
    RuleSpec("MANUAL_ANNOTATION_FALLBACK", "1", "INFO", "annotation", "resolve_annotation_source", "manual annotation EDF selected after automatic CAISR is unavailable"),
    RuleSpec("SOURCE_MANIFEST_INVALID", "1", "ERROR", "source", "parse_source_manifest", "source manifest cannot be parsed or requested scope is absent"),
    RuleSpec("SOURCE_MANIFEST_DUPLICATE", "1", "ERROR", "source", "parse_source_manifest", "source manifest contains duplicate record_id"),
    RuleSpec("CACHE_SET_MISMATCH", "1", "ERROR", "cache-root", "main", "expected source record set and HDF5 identity set differ"),
)


def _manual_annotation_candidates(record_id: str, site: str, manual_root: Path) -> tuple[Path, ...]:
    """只接受完整 record_id 前缀的人工 EDF，避免 subject-only 模糊匹配。"""

    site_root = manual_root / site
    if not site_root.is_dir():
        return ()
    prefix = f"{record_id}_"
    return tuple(
        sorted(
            path.resolve()
            for path in site_root.rglob("*.edf")
            if path.is_file() and path.name.startswith(prefix)
        )
    )


def inspect_annotation_edf(path: Path | str, record_id: str) -> AnnotationInspection:
    """读取 annotation EDF 的 header 和 annotation channels，不触碰 PSG waveform。"""

    annotation_path = Path(path).resolve()
    try:
        import pyedflib

        reader = pyedflib.EdfReader(str(annotation_path))
    except Exception as exc:
        return AnnotationInspection(
            path=annotation_path,
            record_id=record_id,
            status="ANNOTATION_UNREADABLE",
            error=f"{type(exc).__name__}: {exc}",
        )
    try:
        labels = tuple(str(value).strip() for value in reader.getSignalLabels())
        if len(set(labels)) != len(labels):
            return AnnotationInspection(
                path=annotation_path,
                record_id=record_id,
                status="ANNOTATION_UNREADABLE",
                error="duplicate annotation channel labels",
            )
        rates = tuple(float(value) for value in reader.getSampleFrequencies())
        counts = tuple(int(value) for value in reader.getNSamples())
        dimensions = tuple(
            str(reader.getPhysicalDimension(index)).strip()
            for index in range(len(labels))
        )
        values = {
            label: np.asarray(reader.readSignal(index), dtype=np.float64)
            for index, label in enumerate(labels)
        }
        return AnnotationInspection(
            path=annotation_path,
            record_id=record_id,
            status="AUTO_CAISR_AVAILABLE" if "_caisr_" in annotation_path.name else "MANUAL_ANNOTATION_AVAILABLE",
            channel_names=labels,
            sampling_rates_hz=rates,
            sample_counts=counts,
            physical_dimensions=dimensions,
            start_datetime=str(reader.getStartdatetime()),
            duration_sec=float(reader.file_duration),
            values_by_channel=values,
        )
    except Exception as exc:
        return AnnotationInspection(
            path=annotation_path,
            record_id=record_id,
            status="ANNOTATION_UNREADABLE",
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        reader.close()


def resolve_annotation_source(
    record_id: str,
    site: str,
    auto_caisr_path: Path | str | None,
    *,
    manual_root: Path | str = Path("/database2/physionet2026_kaggle/data2/human_annotations"),
    validate: bool = False,
) -> AnnotationSourceResolution:
    """按自动 CAISR → 唯一人工 EDF → unavailable 解析 annotation 来源。

    validate=True 时会打开候选 annotation EDF 进行解析；自动文件无法解析时才
    回退到人工文件。解析失败、候选歧义和缺失都保留明确状态，绝不猜测。
    """

    manual_root = Path(manual_root).resolve()
    auto_path = Path(auto_caisr_path).resolve() if auto_caisr_path else None
    auto_inspection: AnnotationInspection | None = None
    if auto_path is not None and auto_path.is_file():
        if not validate:
            return AnnotationSourceResolution(record_id, site, "AUTO_CAISR_AVAILABLE", auto_path, (auto_path,))
        auto_inspection = inspect_annotation_edf(auto_path, record_id)
        if auto_inspection.status != "ANNOTATION_UNREADABLE":
            return AnnotationSourceResolution(record_id, site, "AUTO_CAISR_AVAILABLE", auto_path, (auto_path,), auto_inspection)

    candidates = _manual_annotation_candidates(record_id, site, manual_root)
    if len(candidates) > 1:
        return AnnotationSourceResolution(
            record_id, site, "ANNOTATION_AMBIGUOUS", None, candidates,
            reason="multiple exact record_id manual annotation EDF candidates",
        )
    if not candidates:
        reason = "automatic CAISR missing"
        if auto_inspection is not None:
            reason = f"automatic CAISR unreadable: {auto_inspection.error}"
        return AnnotationSourceResolution(record_id, site, "ANNOTATION_UNAVAILABLE", None, reason=reason)
    manual_path = candidates[0]
    inspection = inspect_annotation_edf(manual_path, record_id) if validate else None
    if inspection is not None and inspection.status == "ANNOTATION_UNREADABLE":
        return AnnotationSourceResolution(record_id, site, "ANNOTATION_UNREADABLE", manual_path, candidates, inspection, inspection.error)
    return AnnotationSourceResolution(record_id, site, "MANUAL_ANNOTATION_AVAILABLE", manual_path, candidates, inspection)


def _text(value: Any) -> str:
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value[()]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _scalar(handle: h5py.File, path: str) -> Any:
    value = handle[path][()]
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return value[()]
    if isinstance(value, np.ndarray) and value.size == 1:
        return value.reshape(-1)[0]
    return value


def _event(severity: str, rule_id: str, record_id: str, site: str, reason: str, evidence: str) -> AuditEvent:
    return AuditEvent(severity, rule_id, record_id, site, reason, evidence)


def _subject_session(record_id: str) -> tuple[str, str]:
    match = RECORD_RE.fullmatch(record_id)
    if match is None:
        raise ValueError(f"invalid record_id: {record_id}")
    # subject 保留 BIDS 的 sub- 前缀；SessionID 与 demographics.csv 的项目键一致，
    # 使用不带 ses- 的原始编号。feature_extraction/sample_loader.py 同时接受两种输入。
    return f"sub-{match.group('subject')}", match.group("session")


@lru_cache(maxsize=16)
def _demographics_index(path_text: str) -> dict[tuple[str, str], list[dict[str, str]]]:
    path = Path(path_text)
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        required = {"BidsFolder", "SessionID"}
        if not required.issubset(fields):
            raise ValueError(f"demographics missing columns: {sorted(required - fields)}")
        result: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
        for row in reader:
            key = ((row.get("BidsFolder") or "").strip(), (row.get("SessionID") or "").strip())
            if key[0] and key[1]:
                result[key].append({key_name: (value or "").strip() for key_name, value in row.items()})
        return dict(result)


def validate_demographics(path: Path, record_id: str, site: str) -> tuple[bool, str]:
    try:
        index = _demographics_index(str(path.resolve()))
        subject, session = _subject_session(record_id)
    except Exception as exc:
        return False, str(exc)
    session_options = (session, session.removeprefix("ses-"))
    subject_options = (subject, subject.removeprefix("sub-"))
    matches = []
    matched_keys: set[tuple[str, str]] = set()
    for subject_option in subject_options:
        for session_option in session_options:
            key = (subject_option, session_option)
            if key not in matched_keys:
                matches.extend(index.get(key, []))
                matched_keys.add(key)
    if len(matches) != 1:
        return False, f"key=({subject},{session}) matches={len(matches)} path={path}"
    site_values = {row.get("SiteID", "") for row in matches if row.get("SiteID", "")}
    if site_values and site not in site_values:
        return False, f"SiteID={sorted(site_values)} expected={site}"
    return True, f"key=({subject},{session}) unique"


def _add_demographics_error(result: RecordAudit, path: Path) -> None:
    ok, evidence = validate_demographics(path, result.record_id, result.site)
    result.checks.append("demographics_join")
    if not ok:
        result.add(_event("ERROR", "DEMOGRAPHICS_JOIN_INVALID", result.record_id, result.site, "demographics join failed", evidence))


def _require_dataset(handle: h5py.File, result: RecordAudit, path: str) -> h5py.Dataset | None:
    if path not in handle or not isinstance(handle[path], h5py.Dataset):
        result.add(_event("ERROR", "H5_SCHEMA_MISMATCH", result.record_id, result.site, "required dataset missing", path))
        return None
    return handle[path]


def _check_scalar_metadata(handle: h5py.File, result: RecordAudit) -> None:
    for path in ("metadata/psg/start_datetime", "metadata/psg/duration_sec", "metadata/caisr/start_datetime", "metadata/caisr/duration_sec", "metadata/caisr/file_available", "metadata/caisr/datetime_offset_used", "metadata/caisr/alignment_method", "metadata/caisr/source_path"):
        if _require_dataset(handle, result, path) is None:
            continue
        if handle[path].shape != () and path not in {"metadata/caisr/alignment_method", "metadata/caisr/source_path"}:
            result.add(_event("ERROR", "H5_SCHEMA_MISMATCH", result.record_id, result.site, "scalar metadata has non-scalar shape", f"{path} shape={handle[path].shape}"))
    if "metadata/psg/duration_sec" in handle:
        duration = float(_scalar(handle, "metadata/psg/duration_sec"))
        if not np.isfinite(duration) or duration < 0:
            result.add(_event("ERROR", "H5_SCHEMA_MISMATCH", result.record_id, result.site, "PSG duration is invalid", f"duration_sec={duration}"))
    if "metadata/caisr/alignment_method" in handle and _text(_scalar(handle, "metadata/caisr/alignment_method")) != "psg_relative_sample0":
        result.add(_event("ERROR", "H5_SCHEMA_MISMATCH", result.record_id, result.site, "CAISR alignment contract mismatch", f"alignment_method={_text(_scalar(handle, 'metadata/caisr/alignment_method'))}"))
    if "metadata/caisr/datetime_offset_used" in handle and bool(_scalar(handle, "metadata/caisr/datetime_offset_used")):
        result.add(_event("ERROR", "H5_SCHEMA_MISMATCH", result.record_id, result.site, "CAISR datetime offset must not be used", "datetime_offset_used=True"))


def _check_metadata_vectors(handle: h5py.File, result: RecordAudit, t: int) -> None:
    for modality, (channels, _) in MODALITY_SPECS.items():
        names_path = f"metadata/{modality}/channel_names"
        names_ds = _require_dataset(handle, result, names_path)
        if names_ds is not None:
            names = tuple(_text(value) for value in np.asarray(names_ds))
            if names != CANONICAL_CHANNELS[modality]:
                result.add(_event("ERROR", "CHANNEL_METADATA_CONFLICT", result.record_id, result.site, "canonical channel order mismatch", f"{modality} names={names!r}"))
        for field in ("candidate_source_labels", "selected_source_labels", "channel_source", "derivation_method", "reference_system", "physical_dimension", "native_sampling_rate_hz", "native_sample_count"):
            ds = _require_dataset(handle, result, f"metadata/{modality}/{field}")
            if ds is not None and ds.shape != (channels,):
                result.add(_event("ERROR", "CHANNEL_METADATA_CONFLICT", result.record_id, result.site, "metadata vector length mismatch", f"{modality}/{field} shape={ds.shape} expected={(channels,)}"))
        present = _require_dataset(handle, result, f"quality/channel_present/{modality}")
        if present is not None:
            if present.shape != (channels,) or present.dtype != np.dtype("bool"):
                result.add(_event("ERROR", "H5_SCHEMA_MISMATCH", result.record_id, result.site, "channel_present shape/dtype mismatch", f"{modality} shape={present.shape} dtype={present.dtype}"))
            else:
                values = np.asarray(present, dtype=bool)
                result.channel_present[modality] = values
                for index, available in enumerate(values):
                    if not available:
                        result.add(_event("INFO", "CHANNEL_UNAVAILABLE", result.record_id, result.site, "canonical channel is structurally unavailable", f"{modality}/{CANONICAL_CHANNELS[modality][index]}"))
        rates = np.asarray(handle[f"metadata/{modality}/native_sampling_rate_hz"]) if f"metadata/{modality}/native_sampling_rate_hz" in handle else np.empty(0)
        counts = np.asarray(handle[f"metadata/{modality}/native_sample_count"]) if f"metadata/{modality}/native_sample_count" in handle else np.empty(0)
        if rates.shape == (channels,) and counts.shape == (channels,) and modality in result.channel_present:
            for index, (available, rate, count) in enumerate(zip(result.channel_present[modality], rates, counts)):
                if available and (not np.isfinite(rate) or float(rate) <= 0 or int(count) < 0):
                    result.add(_event("ERROR", "CHANNEL_METADATA_CONFLICT", result.record_id, result.site, "present channel has invalid native support metadata", f"{modality}/{CANONICAL_CHANNELS[modality][index]} fs={rate} samples={count}"))
                if available and np.isfinite(rate) and int(count) >= 0 and int(count) < int(round(t * float(rate))):
                    result.add(_event("INFO", "SHORT_CHANNEL_SUPPORT", result.record_id, result.site, "native support shorter than PSG timebase", f"{modality}/{CANONICAL_CHANNELS[modality][index]} samples={int(count)} expected_at_least={int(round(t * float(rate)))}"))
        if modality == "spo2":
            for path in ("metadata/spo2/canonical_unit", "metadata/spo2/scale_factor"):
                _require_dataset(handle, result, path)
            if "metadata/spo2/canonical_unit" in handle and _text(_scalar(handle, "metadata/spo2/canonical_unit")) != "%":
                result.add(_event("ERROR", "CHANNEL_METADATA_CONFLICT", result.record_id, result.site, "canonical SpO2 unit mismatch", f"unit={_text(_scalar(handle, 'metadata/spo2/canonical_unit'))}"))
            if "metadata/spo2/scale_factor" in handle and int(_scalar(handle, "metadata/spo2/scale_factor")) not in (1, 100):
                result.add(_event("ERROR", "CHANNEL_METADATA_CONFLICT", result.record_id, result.site, "canonical SpO2 scale factor mismatch", f"scale_factor={_scalar(handle, 'metadata/spo2/scale_factor')}"))


def _check_waveform_structure(handle: h5py.File, result: RecordAudit, t: int) -> None:
    for modality, (channels, samples) in MODALITY_SPECS.items():
        ds = _require_dataset(handle, result, f"signals/{modality}")
        if ds is not None and (ds.shape != (t, channels, samples) or ds.dtype != np.dtype("float32")):
            result.add(_event("ERROR", "H5_SCHEMA_MISMATCH", result.record_id, result.site, "signal shape/dtype mismatch", f"signals/{modality} shape={ds.shape} dtype={ds.dtype} expected={(t, channels, samples)} float32"))
        for qc_name in QUALITY_NAMES:
            q = _require_dataset(handle, result, f"quality/{qc_name}/{modality}")
            if q is not None and (q.shape != (t, channels, 6) or q.dtype != np.dtype("bool")):
                result.add(_event("ERROR", "H5_SCHEMA_MISMATCH", result.record_id, result.site, "quality mask shape/dtype mismatch", f"quality/{qc_name}/{modality} shape={q.shape} dtype={q.dtype}"))
        vf = _require_dataset(handle, result, f"quality/channel_valid_fraction_30s/{modality}")
        if vf is not None:
            if vf.shape != (t, channels) or vf.dtype != np.dtype("float32"):
                result.add(_event("ERROR", "H5_SCHEMA_MISMATCH", result.record_id, result.site, "valid fraction shape/dtype mismatch", f"{modality} shape={vf.shape} dtype={vf.dtype}"))
            else:
                values = np.asarray(vf)
                if np.isfinite(values).any() and (float(np.nanmin(values)) < 0 or float(np.nanmax(values)) > 1):
                    result.add(_event("ERROR", "H5_SCHEMA_MISMATCH", result.record_id, result.site, "valid fraction is outside [0,1]", f"{modality} min={float(np.nanmin(values))} max={float(np.nanmax(values))}"))
    second = _require_dataset(handle, result, "quality/spo2_second_valid")
    if second is not None and (second.shape != (t, 30) or second.dtype != np.dtype("bool")):
        result.add(_event("ERROR", "H5_SCHEMA_MISMATCH", result.record_id, result.site, "SpO2 second-valid shape/dtype mismatch", f"shape={second.shape} dtype={second.dtype}"))
    if second is not None and second.shape == (t, 30):
        result.spo2_valid_seconds = int(np.asarray(second, dtype=bool).sum())
        available = bool(result.channel_present.get("spo2", np.asarray([False]))[0])
        if available and result.spo2_valid_seconds == 0:
            result.add(_event("WARN", "SPO2_NO_VALID_SECONDS", result.record_id, result.site, "SpO2 channel is present but has no valid seconds", f"quality/spo2_second_valid true_count=0 shape={second.shape}"))


def _check_timebase(handle: h5py.File, result: RecordAudit, t: int) -> None:
    arrays = []
    for name in ("epoch_index", "epoch_start_sec", "epoch_end_sec"):
        ds = _require_dataset(handle, result, f"timebase/{name}")
        if ds is None:
            continue
        arrays.append((name, np.asarray(ds)))
        if ds.shape != (t,):
            result.add(_event("ERROR", "TIMEBASE_MISMATCH", result.record_id, result.site, "timebase length mismatch", f"{name} shape={ds.shape} expected={(t,)}"))
    if len(arrays) != 3 or any(values.shape != (t,) for _, values in arrays):
        return
    index = dict(arrays)["epoch_index"]
    start = dict(arrays)["epoch_start_sec"]
    end = dict(arrays)["epoch_end_sec"]
    if not np.array_equal(index, np.arange(t, dtype=index.dtype)):
        result.add(_event("ERROR", "TIMEBASE_MISMATCH", result.record_id, result.site, "epoch_index is not contiguous", f"first={index[:3].tolist()} last={index[-3:].tolist()}"))
    expected_start = np.arange(t, dtype=np.float64) * 30.0
    if not np.allclose(start, expected_start, atol=1e-5, rtol=0) or not np.allclose(end, expected_start + 30.0, atol=1e-5, rtol=0):
        result.add(_event("ERROR", "TIMEBASE_MISMATCH", result.record_id, result.site, "epoch boundaries are not PSG-relative 30-second intervals", f"start_head={start[:3].tolist()} end_head={end[:3].tolist()}"))


def _check_caisr(handle: h5py.File, result: RecordAudit, t: int) -> None:
    metadata_paths = ("channel_names", "native_sampling_rate_hz", "native_sample_count", "physical_dimension")
    names: tuple[str, ...] = ()
    available = bool(_scalar(handle, "metadata/caisr/file_available")) if "metadata/caisr/file_available" in handle else False
    if "metadata/caisr/annotation_source" in handle:
        annotation_source = _text(_scalar(handle, "metadata/caisr/annotation_source"))
    else:
        annotation_source = "caisr" if available else "unavailable"
    if annotation_source not in {"caisr", "manual", "unavailable"}:
        result.add(_event("ERROR", "CAISR_CODE_INVALID", result.record_id, result.site, "unknown annotation source", f"annotation_source={annotation_source!r}"))
        annotation_source = "unavailable"
    result.annotation_source = annotation_source
    result.caisr_available = annotation_source == "caisr"
    if annotation_source == "manual":
        result.add(_event("INFO", "MANUAL_ANNOTATION_FALLBACK", result.record_id, result.site, "manual annotation EDF is the selected source", "metadata/caisr/annotation_source=manual"))
    elif annotation_source == "unavailable":
        result.add(_event("INFO", "CAISR_UNAVAILABLE", result.record_id, result.site, "CAISR and manual annotation are unavailable; categorical -1/NaN representation is required", "metadata/caisr/annotation_source=unavailable"))
    if available != (annotation_source != "unavailable"):
        result.add(_event("ERROR", "H5_SCHEMA_MISMATCH", result.record_id, result.site, "annotation source and file_available disagree", f"annotation_source={annotation_source} file_available={available}"))
    allowed_channel_names = set(MANUAL_ANNOTATION_CHANNELS if annotation_source == "manual" else CAISR_CHANNELS)
    if "metadata/caisr/channel_names" in handle:
        names = tuple(_text(value) for value in np.asarray(handle["metadata/caisr/channel_names"]))
        if len(set(names)) != len(names) or any(name not in allowed_channel_names for name in names):
            result.add(_event("ERROR", "CAISR_CODE_INVALID", result.record_id, result.site, "CAISR metadata contains duplicate or unknown channel", f"names={names!r}"))
    for field in metadata_paths:
        ds = _require_dataset(handle, result, f"metadata/caisr/{field}")
        if ds is not None and ds.shape != (len(names),):
            result.add(_event("ERROR", "H5_SCHEMA_MISMATCH", result.record_id, result.site, "CAISR metadata length mismatch", f"{field} shape={ds.shape}"))
    for name, (shape_spec, dtype) in CAISR_DATASET_SPECS.items():
        ds = _require_dataset(handle, result, f"annotations/caisr/{name}")
        if ds is None:
            continue
        shape = tuple(t if part == "T" else part for part in shape_spec)
        if ds.shape != shape or ds.dtype != dtype:
            result.add(_event("ERROR", "H5_SCHEMA_MISMATCH", result.record_id, result.site, "CAISR dataset shape/dtype mismatch", f"{name} shape={ds.shape} dtype={ds.dtype} expected={shape} {dtype}"))
    if not all(f"annotations/caisr/{name}" in handle for name in CAISR_DATASET_SPECS):
        return
    stage = np.asarray(handle["annotations/caisr/stage"])
    arousal = np.asarray(handle["annotations/caisr/arousal"])
    resp = np.asarray(handle["annotations/caisr/respiratory"])
    limb = np.asarray(handle["annotations/caisr/limb"])
    allowed = (("stage", stage, {-1, 1, 2, 3, 4, 5}), ("arousal", arousal, {-1, 0, 1}), ("respiratory", resp, {-1, 0, 1, 2, 3, 4, 5}), ("limb", limb, {-1, 0, 1, 2}))
    if annotation_source != "unavailable":
        for name, values, permitted in allowed:
            unexpected = sorted(set(int(value) for value in np.unique(values)) - permitted)
            if unexpected:
                result.add(_event("ERROR", "CAISR_CODE_INVALID", result.record_id, result.site, f"{name} contains unsupported categorical code", f"values={unexpected}"))
    else:
        for name, values, _ in allowed:
            if np.any(values != -1):
                result.add(_event("ERROR", "CAISR_CODE_INVALID", result.record_id, result.site, f"unavailable {name} must be all -1", f"values={sorted(set(int(value) for value in np.unique(values)))}"))
    probabilities = (("stage_probability", "stage_probability_valid"), ("arousal_probability", "arousal_probability_valid"))
    for probability_name, valid_name in probabilities:
        values = np.asarray(handle[f"annotations/caisr/{probability_name}"])
        valid = np.asarray(handle[f"annotations/caisr/{valid_name}"], dtype=bool)
        finite = np.isfinite(values)
        if annotation_source != "unavailable":
            if finite.any() and (float(np.nanmin(values)) < 0 or float(np.nanmax(values)) > 1):
                result.add(_event("ERROR", "CAISR_CODE_INVALID", result.record_id, result.site, "CAISR probability outside [0,1]", f"{probability_name} min={float(np.nanmin(values))} max={float(np.nanmax(values))}"))
            if probability_name == "stage_probability":
                valid_broadcast = valid[:, None]
            else:
                valid_broadcast = valid[:, :, None]
            if np.any(valid_broadcast & ~finite) or np.any((~valid_broadcast) & finite):
                result.add(_event("ERROR", "CAISR_CODE_INVALID", result.record_id, result.site, "probability validity mask disagrees with NaN/finite storage", f"{probability_name} valid={int(valid.sum())} finite={int(finite.sum())}"))


def _deep_signal_check(handle: h5py.File, result: RecordAudit, t: int) -> None:
    result.checks.append("deep_signal_chunk_check")
    for modality, (channels, _) in MODALITY_SPECS.items():
        ds = handle[f"signals/{modality}"]
        present = result.channel_present.get(modality, np.ones(channels, dtype=bool))
        for start in range(0, t, 8):
            values = np.asarray(ds[start:min(t, start + 8)])
            if np.isinf(values).any():
                result.add(_event("ERROR", "H5_SCHEMA_MISMATCH", result.record_id, result.site, "waveform contains Inf; canonical missing values must be NaN", f"signals/{modality} chunk_start={start}"))
            for index, available in enumerate(present):
                if not available and np.isfinite(values[:, index, :]).any():
                    result.add(_event("ERROR", "H5_SCHEMA_MISMATCH", result.record_id, result.site, "structurally missing channel contains finite waveform values", f"signals/{modality}/{CANONICAL_CHANNELS[modality][index]} chunk_start={start}"))


def audit_record(path: Path | str, record_id: str, site: str, demographics_path: Path | str, *, schema_profile: str = SCHEMA_PROFILE, deep_signal: bool = False) -> RecordAudit:
    result = RecordAudit(record_id=record_id, site=site, cache_path=Path(path).resolve())
    if schema_profile != SCHEMA_PROFILE:
        result.add(_event("ERROR", "H5_SCHEMA_MISMATCH", record_id, site, "unsupported schema profile", schema_profile))
        return result
    _add_demographics_error(result, Path(demographics_path))
    if not result.cache_path.is_file():
        result.add(_event("ERROR", "H5_OPEN_FAILED", record_id, site, "HDF5 file missing", str(result.cache_path)))
        return result
    try:
        handle = h5py.File(result.cache_path, "r")
    except Exception as exc:
        result.add(_event("ERROR", "H5_OPEN_FAILED", record_id, site, "HDF5 open failed", f"{result.cache_path}: {type(exc).__name__}: {exc}"))
        return result
    with handle:
        result.checks.append("h5_open")
        parent_site = result.cache_path.parent.name
        if result.cache_path.stem != record_id or parent_site != site:
            result.add(_event("ERROR", "RECORD_ID_MISMATCH", record_id, site, "HDF5 path identity mismatch", f"parent={parent_site} stem={result.cache_path.stem}"))
        for attr, expected in (("record_id", record_id), ("site", site), ("cache_schema_version", "1.0")):
            if attr not in handle.attrs:
                result.add(_event("ERROR", "RECORD_ID_MISMATCH" if attr in {"record_id", "site"} else "H5_SCHEMA_MISMATCH", record_id, site, "required root attribute missing", attr))
            elif _text(handle.attrs[attr]) != expected:
                result.add(_event("ERROR", "RECORD_ID_MISMATCH" if attr in {"record_id", "site"} else "H5_SCHEMA_MISMATCH", record_id, site, "root identity/schema attribute mismatch", f"{attr}={_text(handle.attrs[attr])!r} expected={expected!r}"))
        missing_groups = [name for name in REQUIRED_GROUPS if name not in handle]
        if missing_groups:
            result.add(_event("ERROR", "H5_SCHEMA_MISMATCH", record_id, site, "required group missing", repr(missing_groups)))
            return result
        if "complete_epoch_count" not in handle.attrs:
            result.add(_event("ERROR", "H5_SCHEMA_MISMATCH", record_id, site, "complete_epoch_count missing", "root attrs"))
            return result
        t = int(handle.attrs["complete_epoch_count"])
        if t < 0:
            result.add(_event("ERROR", "H5_SCHEMA_MISMATCH", record_id, site, "complete_epoch_count is negative", str(t)))
            return result
        result.checks.extend(("root_identity", "waveform_shapes_dtypes", "quality_shapes_dtypes", "metadata_vectors", "timebase", "caisr", "spo2"))
        _check_metadata_vectors(handle, result, t)
        _check_waveform_structure(handle, result, t)
        _check_scalar_metadata(handle, result)
        _check_timebase(handle, result, t)
        _check_caisr(handle, result, t)
        if deep_signal and not any(event.rule_id == "H5_SCHEMA_MISMATCH" and "signal shape" in event.reason for event in result.errors):
            _deep_signal_check(handle, result, t)
    return result


def parse_source_manifest(path: Path | str) -> tuple[list[SourceRecord], list[AuditEvent]]:
    manifest_path = Path(path).resolve()
    events: list[AuditEvent] = []
    records: list[SourceRecord] = []
    if not manifest_path.is_file():
        return [], [_event("ERROR", "SOURCE_MANIFEST_INVALID", "<source-manifest>", "<all>", "source manifest missing", str(manifest_path))]
    try:
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            columns = tuple(reader.fieldnames or ())
            if columns == SOURCE_COLUMNS:
                old_format = True
            elif columns == FORMAL_COLUMNS:
                old_format = False
            else:
                return [], [_event("ERROR", "SOURCE_MANIFEST_INVALID", "<source-manifest>", "<all>", "source manifest columns are not recognized", repr(columns))]
            seen: set[str] = set()
            for line, row in enumerate(reader, start=2):
                record_id = (row.get("record_id") or "").strip()
                site = (row.get("site") or "").strip()
                try:
                    subject, session = _subject_session(record_id)
                except ValueError as exc:
                    events.append(_event("ERROR", "SOURCE_MANIFEST_INVALID", record_id or f"line-{line}", site or "<unknown>", str(exc), f"line={line}"))
                    continue
                if record_id in seen:
                    events.append(_event("ERROR", "SOURCE_MANIFEST_DUPLICATE", record_id, site, "duplicate record_id", f"line={line}"))
                    continue
                seen.add(record_id)
                if old_format:
                    psg = (row.get("psg_path") or "").strip()
                    caisr = (row.get("caisr_path") or "").strip()
                else:
                    subject = (row.get("subject_id") or subject).strip()
                    session = (row.get("session_id") or session).strip()
                    psg = (row.get("psg_edf_path") or "").strip()
                    caisr = (row.get("caisr_edf_path") or "").strip()
                if not record_id or not site:
                    events.append(_event("ERROR", "SOURCE_MANIFEST_INVALID", record_id or f"line-{line}", site or "<unknown>", "record_id/site is empty", f"line={line}"))
                    continue
                records.append(SourceRecord(record_id, site, subject, session, psg, caisr))
    except Exception as exc:
        events.append(_event("ERROR", "SOURCE_MANIFEST_INVALID", "<source-manifest>", "<all>", "source manifest cannot be read", f"{type(exc).__name__}: {exc}"))
    return records, events


def _csv_has_site(path: Path, site: str) -> bool:
    try:
        index = _demographics_index(str(path.resolve()))
    except Exception:
        return False
    return any(rows and any(row.get("SiteID", "") == site for row in rows) for rows in index.values())


def resolve_demographics(site: str, override: Path | None) -> Path:
    candidates: list[Path] = []
    if override is not None:
        if override.is_dir():
            candidates.append(override / "demographics.csv")
        else:
            candidates.append(override)
            candidates.extend((override.parent / name / "demographics.csv" for name in ("data", "data2", "data3")))
    candidates.extend((Path("/database2/physionet2026_kaggle") / name / "demographics.csv" for name in ("data", "data2", "data3")))
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen or not candidate.is_file():
            continue
        seen.add(candidate)
        if _csv_has_site(candidate, site):
            return candidate
    return (override.resolve() if override is not None else Path("/database2/physionet2026_kaggle/data/demographics.csv").resolve())


def _formal_row(source: SourceRecord, cache_root: Path, demo_path: Path) -> dict[str, str]:
    return {
        "record_id": source.record_id,
        "site": source.site,
        "subject_id": source.subject_id,
        "session_id": source.session_id,
        "cache_h5_path": str((cache_root / source.site / f"{source.record_id}.h5").resolve()),
        "demographics_csv_path": str(demo_path.resolve()),
        "psg_edf_path": source.psg_edf_path,
        "caisr_edf_path": source.caisr_edf_path,
    }


def resolve_manifest_annotations(
    records: Sequence[SourceRecord],
    *,
    manual_root: Path | str = Path("/database2/physionet2026_kaggle/data2/human_annotations"),
) -> tuple[list[SourceRecord], list[AuditEvent]]:
    """把缺失的 automatic CAISR 路径解析为唯一 manual 路径或空值。"""

    resolved: list[SourceRecord] = []
    events: list[AuditEvent] = []
    for source in records:
        resolution = resolve_annotation_source(
            source.record_id,
            source.site,
            source.caisr_edf_path or None,
            manual_root=manual_root,
            validate=False,
        )
        if resolution.status == "MANUAL_ANNOTATION_AVAILABLE":
            events.append(_event("INFO", "MANUAL_ANNOTATION_FALLBACK", source.record_id, source.site, "selected manual annotation EDF for final manifest", str(resolution.path)))
            annotation_path = str(resolution.path)
        elif resolution.status == "ANNOTATION_UNAVAILABLE":
            annotation_path = ""
        elif resolution.status == "AUTO_CAISR_AVAILABLE":
            annotation_path = str(resolution.path or source.caisr_edf_path)
        else:
            events.append(_event("ERROR", "SOURCE_MANIFEST_INVALID", source.record_id, source.site, "annotation source resolution failed", f"status={resolution.status} reason={resolution.reason}"))
            annotation_path = source.caisr_edf_path
        resolved.append(SourceRecord(source.record_id, source.site, source.subject_id, source.session_id, source.psg_edf_path, annotation_path))
    return resolved, events


def _audit_job(job: tuple[dict[str, str], SourceRecord, Path, str, bool]) -> RecordAudit:
    row, source, demo_path, schema_profile, deep_signal = job
    return audit_record(row["cache_h5_path"], source.record_id, source.site, demo_path, schema_profile=schema_profile, deep_signal=deep_signal)


def _validate_manifest_rows(rows: Sequence[Mapping[str, str]]) -> None:
    seen: set[str] = set()
    for row in rows:
        if tuple(row) != FORMAL_COLUMNS:
            raise ValueError("manifest row columns do not match formal eight-column schema")
        record_id = row["record_id"]
        if record_id in seen:
            raise ValueError(f"duplicate manifest record_id: {record_id}")
        seen.add(record_id)
        if not Path(row["cache_h5_path"]).is_file():
            raise ValueError(f"cache_h5_path does not exist: {row['cache_h5_path']}")
        if not Path(row["demographics_csv_path"]).is_file():
            raise ValueError(f"demographics_csv_path does not exist: {row['demographics_csv_path']}")


def publish_manifest(rows: Sequence[Mapping[str, str]], errors: Sequence[AuditEvent], warnings: Sequence[AuditEvent], output: Path | str, *, replace: bool, dry_run: bool) -> bool:
    output_path = Path(output).resolve()
    if errors or dry_run or (output_path.exists() and not replace):
        return False
    try:
        _validate_manifest_rows(rows)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=output_path.parent, prefix=f".{output_path.name}.", suffix=".tmp", delete=False) as handle:
                temp_path = Path(handle.name)
                writer = csv.DictWriter(handle, fieldnames=FORMAL_COLUMNS, delimiter="\t", lineterminator="\n", extrasaction="raise")
                writer.writeheader()
                writer.writerows(rows)
                handle.flush()
                os.fsync(handle.fileno())
            with temp_path.open("r", encoding="utf-8", newline="") as check:
                reader = csv.DictReader(check, delimiter="\t")
                if tuple(reader.fieldnames or ()) != FORMAL_COLUMNS or len(list(reader)) != len(rows):
                    raise ValueError("temporary manifest validation failed")
            os.replace(temp_path, output_path)
            temp_path = None
            return True
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
    except Exception:
        return False


def _iter_cache_paths(cache_root: Path) -> Iterable[Path]:
    if not cache_root.is_dir():
        return ()
    return (path for site_dir in sorted(cache_root.iterdir()) if site_dir.is_dir() for path in sorted(site_dir.glob("*.h5")))


def _global_event_lines(events: Iterable[AuditEvent]) -> list[str]:
    return [f"[{event.severity}][{event.rule_id}] record_id={event.record_id} site={event.site}\nreason={event.reason}\nevidence={event.evidence}" for event in events]


def _partition_events(events: Iterable[AuditEvent]) -> tuple[list[AuditEvent], list[AuditEvent], list[AuditEvent]]:
    """按严重级别拆分全局事件，避免 INFO 被当作发布错误。"""

    errors: list[AuditEvent] = []
    warnings: list[AuditEvent] = []
    infos: list[AuditEvent] = []
    for event in events:
        if event.severity == "ERROR":
            errors.append(event)
        elif event.severity == "WARN":
            warnings.append(event)
        else:
            infos.append(event)
    return errors, warnings, infos


def render_report(*, cache_root: Path, source_manifest: Path, ruleset_version: str, scope: str, started: str, completed: str, publish_attempted: bool, publish_result: str, audits: Sequence[RecordAudit], global_events: Sequence[AuditEvent], manifest_rows: int, deep_signal: bool) -> str:
    global_errors, global_warnings, global_infos = _partition_events(global_events)
    errors = global_errors + [event for audit in audits for event in audit.errors]
    warnings = global_warnings + [event for audit in audits for event in audit.warnings]
    infos = global_infos + [event for audit in audits for event in audit.infos]
    all_records = {audit.record_id for audit in audits}
    error_records = {event.record_id for event in errors if not event.record_id.startswith("<")}
    warning_records = {event.record_id for event in warnings}
    per_site = Counter(audit.site for audit in audits)
    coverage: Counter[tuple[str, str, str]] = Counter()
    missing: Counter[tuple[str, str]] = Counter()
    for audit in audits:
        for modality, values in audit.channel_present.items():
            for index, available in enumerate(values):
                key = (audit.site, modality, CANONICAL_CHANNELS[modality][index])
                if available:
                    coverage[key] += 1
                else:
                    missing[(audit.site, modality)] += 1
    annotation_sources = Counter((audit.annotation_source or "unknown") for audit in audits)
    caisr_unavailable = annotation_sources.get("unavailable", 0)
    no_spo2 = sum(event.rule_id == "SPO2_NO_VALID_SECONDS" for event in warnings)
    lines = [
        "CACHE RECORD AUDIT",
        f"schema_profile: {SCHEMA_PROFILE}",
        f"cache_root: {cache_root}",
        f"source_manifest: {source_manifest}",
        f"ruleset_version: {ruleset_version}",
        f"started_at: {started}",
        f"completed_at: {completed}",
        f"audit_scope: {scope}",
        f"publish_attempted: {str(publish_attempted).lower()}",
        f"publish_result: {publish_result}",
        f"deep_signal_checked: {str(deep_signal).lower()}",
        "",
        "ERRORS",
    ]
    lines.extend(_global_event_lines(errors) or ["none"])
    lines.extend(["", "WARNINGS"])
    lines.extend(_global_event_lines(warnings) or ["none"])
    lines.extend(["", "SUMMARY", f"records_checked: {len(audits)}", f"unique_records: {len(all_records)}", f"error_records: {len(error_records)}", f"warning_records: {len(warning_records)}", f"error_events: {len(errors)}", f"warning_events: {len(warnings)}", f"per_site_counts: {dict(sorted(per_site.items()))}", f"annotation_availability: source_counts={dict(sorted(annotation_sources.items()))} caisr_unavailable={caisr_unavailable}", f"spo2_no_valid_seconds: {no_spo2}", f"manifest_rows: {manifest_rows}", f"info_events_not_printed: {len(infos)}", "channel_coverage:"])
    for key, count in sorted(coverage.items()):
        lines.append(f"  {key[0]} {key[1]}/{key[2]} present={count}")
    lines.append("structural_missing_by_site_modality:")
    for key, count in sorted(missing.items()):
        lines.append(f"  {key[0]} {key[1]} missing_slots={count}")
    return "\n".join(lines) + "\n"


def write_report(path: Path | str, text: str) -> None:
    report_path = Path(path).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(text, encoding="utf-8")


def _print_rules() -> None:
    for rule in RULES:
        print(f"{rule.rule_id}\t{rule.severity}\t{rule.applies_to}\t{rule.description}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=Path("/database2/physionet2026_cache/waveform_cache_v1"))
    parser.add_argument("--source-manifest", type=Path, default=Path("waveform_cache_manifest.tsv"))
    parser.add_argument("--demographics-csv", type=Path)
    parser.add_argument("--manifest-out", type=Path, default=Path("cache_manifest.tsv"))
    parser.add_argument("--manual-root", type=Path, default=Path("/database2/physionet2026_kaggle/data2/human_annotations"))
    parser.add_argument("--report-out", type=Path, default=Path("local_resume_output/cache_record.txt"))
    parser.add_argument("--schema-profile", default=SCHEMA_PROFILE)
    parser.add_argument("--record-id")
    parser.add_argument("--site")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--replace-manifest", action="store_true")
    parser.add_argument("--deep-signal", action="store_true")
    parser.add_argument("--workers", type=int, default=2, help="低并发 HDF5 metadata 审计 worker 数；默认 2")
    parser.add_argument("--list-rules", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.list_rules:
        _print_rules()
        return 0
    started = datetime.now(timezone.utc).isoformat()
    source_records, global_events = parse_source_manifest(args.source_manifest)
    source_records, annotation_events = resolve_manifest_annotations(source_records, manual_root=args.manual_root)
    global_events.extend(annotation_events)
    full_scope = not args.record_id and not args.site
    scope = "full" if full_scope else f"record_id={args.record_id or '*'} site={args.site or '*'}"
    selected = [record for record in source_records if (not args.record_id or record.record_id == args.record_id) and (not args.site or record.site == args.site)]
    if args.record_id and not selected:
        global_events.append(_event("ERROR", "SOURCE_MANIFEST_INVALID", args.record_id, args.site or "<unknown>", "requested record_id not found", "source manifest"))
    if args.site and not any(record.site == args.site for record in source_records):
        global_events.append(_event("ERROR", "SOURCE_MANIFEST_INVALID", "<site>", args.site, "requested site not found", "source manifest"))
    audits: list[RecordAudit] = []
    rows: list[dict[str, str]] = []
    demo_by_site: dict[str, Path] = {}
    jobs: list[tuple[dict[str, str], SourceRecord, Path, str, bool]] = []
    for source in selected:
        demo_by_site.setdefault(source.site, resolve_demographics(source.site, args.demographics_csv))
        demo_path = demo_by_site[source.site]
        row = _formal_row(source, args.cache_root.resolve(), demo_path)
        rows.append(row)
        jobs.append((row, source, demo_path, args.schema_profile, args.deep_signal))
    worker_count = max(1, min(int(args.workers), 4))
    if jobs:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            audits = list(executor.map(_audit_job, jobs))
    if full_scope:
        expected_ids = {record.record_id for record in source_records}
        actual_ids = {path.stem for path in _iter_cache_paths(args.cache_root.resolve())}
        missing = sorted(expected_ids - actual_ids)
        extra = sorted(actual_ids - expected_ids)
        if missing:
            global_events.append(_event("ERROR", "CACHE_SET_MISMATCH", "<cache-root>", "<all>", "expected source records have no HDF5", f"count={len(missing)} examples={missing[:5]}"))
        if extra:
            global_events.append(_event("ERROR", "CACHE_SET_MISMATCH", "<cache-root>", "<all>", "HDF5 records are not in source manifest", f"count={len(extra)} examples={extra[:5]}"))
    global_errors, global_warnings, _ = _partition_events(global_events)
    errors = global_errors + [event for audit in audits for event in audit.errors]
    warnings = global_warnings + [event for audit in audits for event in audit.warnings]
    publish_attempted = full_scope and not args.dry_run
    published = False
    if publish_attempted and not errors:
        published = publish_manifest(rows, errors, warnings, args.manifest_out, replace=args.replace_manifest, dry_run=args.dry_run)
    if not full_scope or args.dry_run:
        publish_result = "not_attempted_scope_or_dry_run"
    elif errors:
        publish_result = "blocked_by_error"
    elif Path(args.manifest_out).resolve().exists() and not args.replace_manifest and not published:
        publish_result = "existing_manifest_protected"
    elif published:
        publish_result = "published"
    else:
        publish_result = "not_published"
    completed = datetime.now(timezone.utc).isoformat()
    report = render_report(cache_root=args.cache_root.resolve(), source_manifest=args.source_manifest.resolve(), ruleset_version=RULESET_VERSION, scope=scope, started=started, completed=completed, publish_attempted=publish_attempted, publish_result=publish_result, audits=audits, global_events=global_events, manifest_rows=len(rows), deep_signal=args.deep_signal)
    write_report(args.report_out, report)
    print(f"report: {Path(args.report_out).resolve()}")
    print(f"records_checked: {len(audits)} errors: {len(errors)} warnings: {len(warnings)}")
    print(f"publish_result: {publish_result}")
    return 2 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
