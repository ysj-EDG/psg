"""PSG 多模态 canonical waveform cache 的 schema 与 metadata 预处理器。

本模块负责读取 PSG 和 CAISR EDF header，建立以 PSG 相对 t=0 为基准的 30 秒
timebase，组织各 modality 的 canonical channel metadata，并写入 source provenance
与 channel-present quality index。它还提供单 record 的 header metadata 打印、source-plan
检查、schema smoke 入口，以及全量 PSG record 的 20-slot audit，作为后续 waveform
cache 预处理的统一边界。

CAISR annotation 采用固定 index contract：``stage_caisr`` 和 stage probabilities
为 1/30 Hz（每个 30 秒 PSG epoch 一个值），arousal 为 2 Hz（每个 epoch 60 个样本），
respiratory 与 limb annotation 为 1 Hz（每个 epoch 30 个样本）。CAISR header datetime
作为 metadata 字段保存，所有 alignment 以 PSG-relative t=0 表达。后续 annotation cache
需要同时保存原始 code 和 validity mask，并保留 ``limb_caisr == 2`` 与 stage code 9 的
原始语义。

waveform dataset 的正式存储契约为 float32、缺失值 NaN，以及 HDF5 chunk/compression；
本模块中的 target sampling rates 和 schema 定义为后续各 modality writer 提供固定接口。
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, TextIO

import h5py
import numpy as np

from channel_sources import (
    CANONICAL_BY_MODALITY,
    SourcePlan,
    build_source_plans,
    find_resp_source_conflicts,
)
from waveform_preprocessing import (
    CaisrAnnotationResult,
    missing_caisr_annotations,
    preprocess_caisr_arrays,
    read_caisr_annotations,
)

CACHE_SCHEMA_VERSION = "1.0"
PREPROCESSING_VERSION = "1.0"
EPOCH_DURATION_SEC = 30.0
DEFAULT_PSG_ROOT = Path("/database2/physionet2026_kaggle/data2/physiological_data")
DEFAULT_CHANNEL_INSUFFICIENT_PATH = Path(__file__).resolve().with_name("channel_insufficient.txt")
TOTAL_CANONICAL_SLOTS = sum(len(names) for names in CANONICAL_BY_MODALITY.values())

# 这些采样率属于本缓存契约，特意独立于任何旧 feature/cache 实现所使用的采样率。
TARGET_SAMPLING_RATES = {
    "eeg": 128,
    "eog": 128,
    "ecg": 200,
    "emg": 200,
    "resp": 25,
    "spo2": 1,
}

# canonical slot 的既有语义由独立 resolver 维护；本模块只把最终 public slot 名称写入 metadata。
MODALITY_ORDER = tuple(CANONICAL_BY_MODALITY)
CANONICAL_CHANNEL_NAMES = {
    modality: tuple(channel_names)
    for modality, channel_names in CANONICAL_BY_MODALITY.items()
}

# 为未来 signal writer 预先声明统一的数据类型和缺失值约定。
SIGNAL_STORAGE_DTYPE = np.dtype("float32")
SIGNAL_STORAGE_MISSING_VALUE = np.nan


@dataclass(frozen=True)
class EdfHeader:
    """PSG/CAISR EDF header metadata 在 cache schema 中使用的字段集合。"""

    start_datetime: str
    duration_sec: float
    channel_names: tuple[str, ...]
    native_sampling_rates_hz: tuple[float, ...]
    native_sample_counts: tuple[int, ...]
    physical_dimensions: tuple[str, ...]

    def __post_init__(self) -> None:
        lengths = {
            len(self.channel_names),
            len(self.native_sampling_rates_hz),
            len(self.native_sample_counts),
            len(self.physical_dimensions),
        }
        if len(lengths) != 1:
            raise ValueError("EDF header channel metadata lengths do not match")
        if not np.isfinite(float(self.duration_sec)) or self.duration_sec < 0:
            raise ValueError(f"Invalid EDF duration: {self.duration_sec!r}")


@dataclass(frozen=True)
class SourceSlotAuditRecord:
    """单个 PSG record 的 canonical slot audit 结果。"""

    site: str
    record_id: str
    psg_path: Path
    present_count: int
    missing_slots: tuple[str, ...]
    error: str = ""
    resp_source_conflicts: tuple[tuple[int, str, str], ...] = ()

    @property
    def warning(self) -> bool:
        return bool(self.missing_slots) or bool(self.error)


@dataclass(frozen=True)
class SourceSlotAuditSummary:
    """一批 PSG record 的 canonical slot audit 汇总。"""

    records: tuple[SourceSlotAuditRecord, ...]
    expected_records: int | None

    @property
    def total_records(self) -> int:
        return len(self.records)

    @property
    def warning_records(self) -> int:
        return sum(record.warning for record in self.records)


def write_channel_insufficient_report(
    summary: SourceSlotAuditSummary,
    report_path: Path | str,
) -> Path:
    """将 20-slot audit 的 presence 和 RESP 一致性摘要写入文本报告。"""

    output_path = Path(report_path)
    denominator = summary.expected_records if summary.expected_records is not None else summary.total_records
    present_distribution = Counter(record.present_count for record in summary.records)
    missing_distribution = Counter(
        TOTAL_CANONICAL_SLOTS - record.present_count
        for record in summary.records
    )
    complete_records = [
        record
        for record in summary.records
        if record.present_count == TOTAL_CANONICAL_SLOTS and not record.error
    ]
    missing_frequency = Counter(
        slot
        for record in summary.records
        for slot in record.missing_slots
    )
    affected_records = [
        record
        for record in summary.records
        if record.resp_source_conflicts
    ]
    affected_pairs = Counter(
        conflict
        for record in summary.records
        for conflict in record.resp_source_conflicts
    )

    lines = [
        "结果：",
        f"records_scanned: {summary.total_records}",
        f"record_count: {summary.total_records}/{denominator}",
        "",
        "present_slot_count:",
    ]
    lines.extend(
        f"{present}/{TOTAL_CANONICAL_SLOTS}   {count}"
        for present, count in sorted(present_distribution.items())
    )
    lines.append("missing_slot_count:")
    lines.extend(
        f"{missing}/{TOTAL_CANONICAL_SLOTS}   {count}"
        for missing, count in sorted(missing_distribution.items())
    )
    lines.extend((
        f"all_20_slots_present: {len(complete_records)}/{denominator}",
        "",
        f"实际发现 {len(complete_records)} 条完整 20/20：",
    ))
    lines.extend(record.record_id for record in complete_records)
    if not complete_records:
        lines.append("none")

    lines.extend(("", "最常缺失的 canonical slot："))
    lines.extend(
        f"{slot:<23}{count}"
        for slot, count in missing_frequency.most_common(5)
    )
    lines.extend(("", "RESP selected raw-channel consistency:"))
    lines.append(f"affected_record_count: {len(affected_records)}")
    lines.append("affected_slot_pairs:")
    if affected_pairs:
        for (raw_index, left, right), count in sorted(affected_pairs.items()):
            lines.append(
                f"  {left} <-> {right} "
                f"(raw_channel_index={raw_index})   {count}"
            )
    else:
        lines.append("  none")
    lines.append("example_record_ids:")
    example_ids = []
    for record in affected_records:
        if record.record_id not in example_ids:
            example_ids.append(record.record_id)
        if len(example_ids) >= 10:
            break
    lines.extend(f"  {record_id}" for record_id in example_ids or ["none"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def read_edf_header(path: Path | str) -> EdfHeader:
    """读取 EDF 的起始时间、duration 和逐通道采样 metadata，构成 EdfHeader。"""

    import pyedflib

    reader = pyedflib.EdfReader(str(path))
    try:
        labels = tuple(str(value).strip() for value in reader.getSignalLabels())
        rates = tuple(float(value) for value in reader.getSampleFrequencies())
        counts = tuple(int(value) for value in reader.getNSamples())
        dimensions = tuple(
            str(reader.getPhysicalDimension(index)).strip()
            for index in range(len(labels))
        )
        return EdfHeader(
            start_datetime=str(reader.getStartdatetime()),
            duration_sec=float(reader.file_duration),
            channel_names=labels,
            native_sampling_rates_hz=rates,
            native_sample_counts=counts,
            physical_dimensions=dimensions,
        )
    finally:
        reader.close()


def iter_psg_records(psg_root: Path | str) -> tuple[tuple[str, str, Path], ...]:
    """按 site 和文件名稳定枚举 PSG record。"""

    root = Path(psg_root)
    if not root.is_dir():
        raise FileNotFoundError(f"PSG data root not found: {root}")
    records = tuple(
        (path.parent.name, path.stem, path)
        for path in sorted(root.glob("*/*.edf"), key=lambda item: (item.parent.name, item.name))
    )
    return records


def audit_source_slots(
    psg_root: Path | str,
    *,
    expected_records: int | None = 6600,
    header_reader: Callable[[Path | str], EdfHeader] = read_edf_header,
    stream: TextIO | None = None,
    report_path: Path | str | None = None,
) -> SourceSlotAuditSummary:
    """扫描 PSG record 的 20 个 canonical slots 并实时输出完整性日志。"""

    import sys

    output = sys.stdout if stream is None else stream
    records = iter_psg_records(psg_root)
    if expected_records is not None and len(records) != expected_records:
        print(
            f"WARNING record_count: discovered={len(records)} "
            f"expected={expected_records}",
            file=output,
            flush=True,
        )

    audit_records: list[SourceSlotAuditRecord] = []
    canonical_names = tuple(
        canonical
        for modality in MODALITY_ORDER
        for canonical in CANONICAL_CHANNEL_NAMES[modality]
    )
    total = len(records)
    for position, (site, record_id, psg_path) in enumerate(records, start=1):
        error = ""
        resp_source_conflicts: tuple[tuple[int, str, str], ...] = ()
        try:
            header = header_reader(psg_path)
            plans = build_source_plans(
                header.channel_names,
                header.native_sampling_rates_hz,
                header.native_sample_counts,
                header.physical_dimensions,
            )
            missing_slots = tuple(
                canonical for canonical in canonical_names if not plans[canonical].channel_present
            )
            present_count = TOTAL_CANONICAL_SLOTS - len(missing_slots)
            resp_source_conflicts = find_resp_source_conflicts(plans)
        except Exception as exc:  # record-level audit warning; continue scanning
            missing_slots = canonical_names
            present_count = 0
            error = f"{type(exc).__name__}: {exc}"

        audit_record = SourceSlotAuditRecord(
            site=site,
            record_id=record_id,
            psg_path=psg_path,
            present_count=present_count,
            missing_slots=missing_slots,
            error=error,
            resp_source_conflicts=resp_source_conflicts,
        )
        audit_records.append(audit_record)
        status = f"[{position}/{total}] {record_id} ({present_count}/{TOTAL_CANONICAL_SLOTS})"
        if audit_record.warning:
            status += " WARNING"
            if missing_slots:
                status += f" missing={','.join(missing_slots)}"
            if error:
                status += f" error={error}"
        print(status, file=output, flush=True)

    summary = SourceSlotAuditSummary(tuple(audit_records), expected_records)
    denominator = expected_records if expected_records is not None else summary.total_records
    print("source-slot audit summary:", file=output, flush=True)
    print(f"records_scanned: {summary.total_records}", file=output, flush=True)
    if expected_records is not None:
        print(
            f"record_count: {summary.total_records}/{expected_records}",
            file=output,
            flush=True,
        )
    present_distribution = Counter(record.present_count for record in summary.records)
    missing_distribution = Counter(
        TOTAL_CANONICAL_SLOTS - record.present_count
        for record in summary.records
    )
    complete_records = [
        record
        for record in summary.records
        if record.present_count == TOTAL_CANONICAL_SLOTS and not record.error
    ]
    print("present_slot_count:", file=output, flush=True)
    for present, count in sorted(present_distribution.items()):
        print(f"  {present}/{TOTAL_CANONICAL_SLOTS}   {count}", file=output, flush=True)
    print("missing_slot_count:", file=output, flush=True)
    for missing, count in sorted(missing_distribution.items()):
        print(f"  {missing}/{TOTAL_CANONICAL_SLOTS}   {count}", file=output, flush=True)
    print(
        f"all_20_slots_present: {len(complete_records)}/{denominator}",
        file=output,
        flush=True,
    )
    if summary.warning_records:
        print("records with missing slots:", file=output, flush=True)
        for audit_record in summary.records:
            if not audit_record.warning:
                continue
            missing = ",".join(audit_record.missing_slots) or "-"
            error_text = f" error={audit_record.error}" if audit_record.error else ""
            print(
                f"  {audit_record.record_id} site={audit_record.site} "
                f"({audit_record.present_count}/{TOTAL_CANONICAL_SLOTS}) "
                f"missing={missing}{error_text}",
                file=output,
                flush=True,
            )
    else:
        print("records with missing slots: none", file=output, flush=True)
    affected_records = [record for record in summary.records if record.resp_source_conflicts]
    affected_pairs = Counter(
        conflict
        for record in summary.records
        for conflict in record.resp_source_conflicts
    )
    print("RESP selected raw-channel consistency:", file=output, flush=True)
    print(f"affected_record_count: {len(affected_records)}", file=output, flush=True)
    print("affected_slot_pairs:", file=output, flush=True)
    if affected_pairs:
        for (raw_index, left, right), count in sorted(affected_pairs.items()):
            print(
                f"  {left} <-> {right} (raw_channel_index={raw_index})   {count}",
                file=output,
                flush=True,
            )
    else:
        print("  none", file=output, flush=True)
    print("example_record_ids:", file=output, flush=True)
    example_ids = []
    for record in affected_records:
        if record.record_id not in example_ids:
            example_ids.append(record.record_id)
        if len(example_ids) >= 10:
            break
    for record_id in example_ids or ["none"]:
        print(f"  {record_id}", file=output, flush=True)
    if report_path is not None:
        report_file = write_channel_insufficient_report(summary, report_path)
        print(f"channel_insufficient_report: {report_file}", file=output, flush=True)
    return summary


def _write_string(group: h5py.Group, name: str, value: str) -> None:
    group.create_dataset(
        name,
        data=np.asarray(value, dtype=h5py.string_dtype(encoding="utf-8")),
    )


def _write_strings(group: h5py.Group, name: str, values: Iterable[str]) -> None:
    group.create_dataset(
        name,
        data=np.asarray(tuple(values), dtype=h5py.string_dtype(encoding="utf-8")),
    )


def _write_json_strings(
    group: h5py.Group, name: str, values: Iterable[Iterable[str]]
) -> None:
    _write_strings(
        group,
        name,
        (json.dumps(tuple(value), ensure_ascii=False, separators=(",", ":")) for value in values),
    )


def _write_header_metadata(group: h5py.Group, header: EdfHeader) -> None:
    _write_string(group, "start_datetime", header.start_datetime)
    group.create_dataset("duration_sec", data=np.float64(header.duration_sec))


def write_schema_cache(
    output_path: Path | str,
    *,
    record_id: str,
    site: str,
    psg_header: EdfHeader,
    caisr_header: EdfHeader | None,
    spo2_scale_factor: int = 1,
    annotation_source: str | None = None,
) -> Path:
    """根据 PSG header 写入 30 秒 PSG-relative timebase、metadata 和 source provenance。"""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    complete_epoch_count = int(math.floor(psg_header.duration_sec / EPOCH_DURATION_SEC))
    epoch_index = np.arange(complete_epoch_count, dtype=np.int64)
    if int(spo2_scale_factor) not in (1, 100):
        raise ValueError("spo2_scale_factor must be 1 or 100")
    if annotation_source is None:
        annotation_source = "caisr" if caisr_header is not None else "unavailable"
    if annotation_source not in {"caisr", "manual", "unavailable"}:
        raise ValueError(f"unsupported annotation_source: {annotation_source!r}")
    source_plans = build_source_plans(
        psg_header.channel_names,
        psg_header.native_sampling_rates_hz,
        psg_header.native_sample_counts,
        psg_header.physical_dimensions,
    )

    with h5py.File(output, "w") as handle:
        handle.attrs["record_id"] = str(record_id)
        handle.attrs["site"] = str(site)
        handle.attrs["cache_schema_version"] = CACHE_SCHEMA_VERSION
        handle.attrs["preprocessing_version"] = PREPROCESSING_VERSION
        handle.attrs["complete_epoch_count"] = complete_epoch_count
        handle.attrs["epoch_duration_sec"] = EPOCH_DURATION_SEC

        timebase = handle.create_group("timebase")
        timebase.create_dataset("epoch_index", data=epoch_index)
        epoch_start = epoch_index.astype(np.float64) * EPOCH_DURATION_SEC
        timebase.create_dataset("epoch_start_sec", data=epoch_start)
        timebase.create_dataset("epoch_end_sec", data=epoch_start + EPOCH_DURATION_SEC)

        metadata = handle.create_group("metadata")
        psg_metadata = metadata.create_group("psg")
        _write_header_metadata(psg_metadata, psg_header)

        caisr_metadata = metadata.create_group("caisr")
        if caisr_header is None:
            _write_string(caisr_metadata, "start_datetime", "")
            caisr_metadata.create_dataset("duration_sec", data=np.float64(np.nan))
            _write_strings(caisr_metadata, "channel_names", ())
            caisr_metadata.create_dataset("native_sampling_rate_hz", data=np.asarray((), dtype=np.float64))
            caisr_metadata.create_dataset("native_sample_count", data=np.asarray((), dtype=np.int64))
            _write_strings(caisr_metadata, "physical_dimension", ())
            file_available = False
        else:
            _write_header_metadata(caisr_metadata, caisr_header)
            _write_strings(caisr_metadata, "channel_names", caisr_header.channel_names)
            caisr_metadata.create_dataset(
                "native_sampling_rate_hz",
                data=np.asarray(caisr_header.native_sampling_rates_hz, dtype=np.float64),
            )
            caisr_metadata.create_dataset(
                "native_sample_count",
                data=np.asarray(caisr_header.native_sample_counts, dtype=np.int64),
            )
            _write_strings(caisr_metadata, "physical_dimension", caisr_header.physical_dimensions)
            file_available = True
        _write_string(caisr_metadata, "source_path", "")
        _write_string(caisr_metadata, "annotation_source", annotation_source)
        _write_string(caisr_metadata, "alignment_method", "psg_relative_sample0")
        caisr_metadata.create_dataset("datetime_offset_used", data=np.bool_(False))
        caisr_metadata.create_dataset("file_available", data=np.bool_(file_available))

        quality = handle.create_group("quality")
        channel_present = quality.create_group("channel_present")
        for modality in MODALITY_ORDER:
            modality_metadata = metadata.create_group(modality)
            plans = tuple(source_plans[name] for name in CANONICAL_CHANNEL_NAMES[modality])
            _write_strings(
                modality_metadata,
                "channel_names",
                CANONICAL_CHANNEL_NAMES[modality],
            )
            _write_json_strings(
                modality_metadata,
                "candidate_source_labels",
                (plan.candidate_source_labels for plan in plans),
            )
            _write_json_strings(
                modality_metadata,
                "selected_source_labels",
                (plan.selected_source_labels for plan in plans),
            )
            _write_strings(
                modality_metadata,
                "channel_source",
                (plan.channel_source for plan in plans),
            )
            _write_strings(
                modality_metadata,
                "derivation_method",
                (plan.derivation_method for plan in plans),
            )
            _write_strings(
                modality_metadata,
                "reference_system",
                (plan.reference_system for plan in plans),
            )
            modality_metadata.create_dataset(
                "native_sampling_rate_hz",
                data=np.asarray(
                    [plan.native_sampling_rate_hz for plan in plans],
                    dtype=np.float64,
                ),
            )
            modality_metadata.create_dataset(
                "native_sample_count",
                data=np.asarray(
                    [plan.native_sample_count for plan in plans],
                    dtype=np.int64,
                ),
            )
            _write_strings(
                modality_metadata,
                "physical_dimension",
                (plan.physical_dimension for plan in plans),
            )
            if modality == "spo2":
                _write_string(modality_metadata, "canonical_unit", "%")
                modality_metadata.create_dataset(
                    "scale_factor", data=np.int64(spo2_scale_factor)
                )
            channel_present.create_dataset(
                modality,
                data=np.asarray([plan.channel_present for plan in plans], dtype=np.bool_),
            )

        # 为后续 waveform、QC 和 annotation 阶段预留对应的顶层 group。
        handle.create_group("signals")
        handle.create_group("annotations").create_group("caisr")

    return output



_ANNOTATION_DATASET_NAMES = (
    "stage",
    "stage_probability",
    "stage_probability_valid",
    "arousal",
    "arousal_probability",
    "arousal_probability_valid",
    "respiratory",
    "limb",
)


def _write_annotation_dataset(group: h5py.Group, name: str, values: np.ndarray) -> None:
    """以固定 dtype 和按 epoch chunk 写入一个 CAISR dataset。"""

    array = np.asarray(values)
    if array.ndim == 0 or array.shape[0] == 0:
        group.create_dataset(name, data=array)
        return
    chunk_shape = (1,) + tuple(array.shape[1:])
    group.create_dataset(
        name,
        data=array,
        dtype=array.dtype,
        chunks=chunk_shape,
        compression="gzip",
        compression_opts=4,
        shuffle=True,
    )


def _set_caisr_metadata(
    handle: h5py.File,
    *,
    metadata: dict[str, object] | None,
) -> None:
    """更新已存在的 metadata/caisr provenance，不创建第二套 metadata group。"""

    group = handle["metadata/caisr"]
    payload = dict(metadata or {})
    if "path" in payload:
        source_path = str(payload["path"])
        if "source_path" in group:
            group["source_path"][()] = source_path
        else:
            _write_string(group, "source_path", source_path)
    if "start_datetime" in payload:
        value = str(payload["start_datetime"])
        if "start_datetime" in group:
            group["start_datetime"][()] = value
        else:
            _write_string(group, "start_datetime", value)
    if "duration_sec" in payload:
        value = float(payload["duration_sec"])
        if "duration_sec" in group:
            group["duration_sec"][()] = value
        else:
            group.create_dataset("duration_sec", data=np.float64(value))
    for name, value in (
        ("alignment_method", "psg_relative_sample0"),
        ("datetime_offset_used", False),
        ("file_available", bool(payload.get("file_available", False))),
    ):
        if name in group:
            group[name][()] = value
        elif isinstance(value, str):
            _write_string(group, name, value)
        else:
            group.create_dataset(name, data=np.bool_(value))
    annotation_source = str(
        payload.get(
            "annotation_source",
            "caisr" if bool(payload.get("file_available", False)) else "unavailable",
        )
    )
    if annotation_source not in {"caisr", "manual", "unavailable"}:
        raise ValueError(f"unsupported annotation_source metadata: {annotation_source!r}")
    if "annotation_source" in group:
        group["annotation_source"][()] = annotation_source
    else:
        _write_string(group, "annotation_source", annotation_source)


def write_caisr_annotations(
    output_path: Path | str,
    result: CaisrAnnotationResult,
    *,
    metadata: dict[str, object] | None = None,
) -> Path:
    """把已对齐的 CAISR arrays 写入 canonical annotations/caisr。"""

    output = Path(output_path)
    arrays = {
        "stage": np.asarray(result.stage, dtype=np.int8),
        "stage_probability": np.asarray(result.stage_probability, dtype=np.float32),
        "stage_probability_valid": np.asarray(result.stage_probability_valid, dtype=np.bool_),
        "arousal": np.asarray(result.arousal, dtype=np.int8),
        "arousal_probability": np.asarray(result.arousal_probability, dtype=np.float32),
        "arousal_probability_valid": np.asarray(result.arousal_probability_valid, dtype=np.bool_),
        "respiratory": np.asarray(result.respiratory, dtype=np.int8),
        "limb": np.asarray(result.limb, dtype=np.int8),
    }
    with h5py.File(output, "a") as handle:
        annotation_group = handle["annotations/caisr"]
        if annotation_group.keys():
            raise ValueError("annotations/caisr group is not empty")
        for name in _ANNOTATION_DATASET_NAMES:
            _write_annotation_dataset(annotation_group, name, arrays[name])
        _set_caisr_metadata(handle, metadata=metadata)
    validate_caisr_annotations(output)
    return output


def write_caisr_annotations_from_file(
    output_path: Path | str,
    caisr_path: Path | str | None,
    *,
    psg_duration_sec: float,
    record_id: str,
    annotation_source: str | None = None,
) -> Path:
    """读取 CAISR/人工 EDF 并写入 canonical annotation；缺失时填充 sentinel。"""

    epochs = int(math.floor(float(psg_duration_sec) / EPOCH_DURATION_SEC))
    if caisr_path is None or not Path(caisr_path).exists():
        result = missing_caisr_annotations(epochs)
        selected_source = "unavailable"
        metadata = {
            "file_available": False,
            "path": "" if caisr_path is None else str(caisr_path),
            "start_datetime": "",
            "duration_sec": np.nan,
            "annotation_source": selected_source,
        }
    else:
        selected_source = annotation_source or "caisr"
        if selected_source not in {"caisr", "manual"}:
            raise ValueError(f"annotation_source must be caisr or manual for an existing EDF, got {selected_source!r}")
        result, metadata = read_caisr_annotations(
            caisr_path,
            psg_duration_sec=psg_duration_sec,
            record_id=record_id,
            annotation_source=selected_source,
        )
    return write_caisr_annotations(output_path, result, metadata=metadata)


def validate_caisr_annotations(path: Path | str) -> None:
    """验证 canonical CAISR tree 的 shape、dtype、code domain 和 probability domain。"""

    with h5py.File(path, "r") as handle:
        epochs = int(handle.attrs["complete_epoch_count"])
        group = handle["annotations/caisr"]
        if set(group.keys()) != set(_ANNOTATION_DATASET_NAMES):
            raise ValueError(f"unexpected annotations/caisr datasets: {sorted(group.keys())}")
        expected = {
            "stage": ((epochs,), np.dtype("int8")),
            "stage_probability": ((epochs, 5), np.dtype("float32")),
            "stage_probability_valid": ((epochs,), np.dtype("bool")),
            "arousal": ((epochs, 60), np.dtype("int8")),
            "arousal_probability": ((epochs, 60, 2), np.dtype("float32")),
            "arousal_probability_valid": ((epochs, 60), np.dtype("bool")),
            "respiratory": ((epochs, 30), np.dtype("int8")),
            "limb": ((epochs, 30), np.dtype("int8")),
        }
        for name, (shape, dtype) in expected.items():
            dataset = group[name]
            if dataset.shape != shape or dataset.dtype != dtype:
                raise ValueError(
                    f"{name} expected shape={shape} dtype={dtype}, "
                    f"got shape={dataset.shape} dtype={dataset.dtype}"
                )
        domains = {
            "stage": {-1, 1, 2, 3, 4, 5},
            "arousal": {-1, 0, 1},
            "respiratory": {-1, 0, 1, 2, 3, 4, 5},
            "limb": {-1, 0, 1, 2},
        }
        for name, allowed in domains.items():
            values = set(np.unique(group[name][:]).tolist())
            if not values.issubset(allowed):
                raise ValueError(f"{name} has invalid codes: {sorted(values - allowed)}")
        stage_probabilities = group["stage_probability"][:]
        stage_valid_expected = (
            np.isfinite(stage_probabilities).all(axis=1)
            & (stage_probabilities >= 0.0).all(axis=1)
            & (stage_probabilities <= 1.0).all(axis=1)
            & (np.abs(stage_probabilities.sum(axis=1) - 1.0) <= 0.01)
            & (group["stage"][:] != -1)
        )
        if not np.array_equal(group["stage_probability_valid"][:], stage_valid_expected):
            raise ValueError("stage_probability_valid does not match probability contract")
        arousal_probabilities = group["arousal_probability"][:]
        arousal_valid_expected = (
            np.isfinite(arousal_probabilities).all(axis=-1)
            & (arousal_probabilities >= 0.0).all(axis=-1)
            & (arousal_probabilities <= 1.0).all(axis=-1)
            & (np.abs(arousal_probabilities.sum(axis=-1) - 1.0) <= 0.01)
        )
        if not np.array_equal(group["arousal_probability_valid"][:], arousal_valid_expected):
            raise ValueError("arousal_probability_valid does not match probability contract")
        for name in ("stage_probability", "arousal_probability"):
            values = group[name][:]
            if np.isinf(values).any():
                raise ValueError(f"{name} contains +/-Inf")
            finite = np.isfinite(values)
            if ((values[finite] < 0.0) | (values[finite] > 1.0)).any():
                raise ValueError(f"{name} finite values must be within [0,1]")


def write_waveform_signals(
    output_path: Path | str,
    signals: dict[str, np.ndarray],
    *,
    quality: dict[str, np.ndarray] | None = None,
) -> Path:
    """将已经完成主路径处理的 modality arrays 写入 signals group。

    正式 waveform dataset 统一使用 float32、NaN 缺失值、按 epoch 的 HDF5 chunk
    和 gzip compression。本函数不做任何 signal 修复、归一化或量化。
    """

    output = Path(output_path)
    expected_modalities = tuple(MODALITY_ORDER)
    missing = [modality for modality in expected_modalities if modality not in signals]
    extra = [modality for modality in signals if modality not in expected_modalities]
    if missing or extra:
        raise ValueError(f"signal modalities mismatch: missing={missing}, extra={extra}")

    with h5py.File(output, "a") as handle:
        group = handle["signals"]
        if group.keys():
            raise ValueError("signals group is not empty")
        for modality in expected_modalities:
            values = np.asarray(signals[modality], dtype=np.float32)
            expected_channels = len(CANONICAL_CHANNEL_NAMES[modality])
            if values.ndim != 3 or values.shape[1] != expected_channels:
                raise ValueError(
                    f"{modality} signal shape must be (T,{expected_channels},samples), "
                    f"got {values.shape}"
                )
            if np.isinf(values).any():
                raise ValueError(f"{modality} signal contains +/-Inf")
            if values.shape[0] == 0 or values.shape[2] == 0:
                raise ValueError(f"{modality} signal has an empty time axis: {values.shape}")
            group.create_dataset(
                modality,
                data=values,
                dtype=np.dtype("float32"),
                chunks=(1, values.shape[1], values.shape[2]),
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )
        if quality:
            quality_group = handle["quality"]
            if "spo2_second_valid" in quality:
                valid = np.asarray(quality["spo2_second_valid"], dtype=np.bool_)
                spo2_shape = handle["signals/spo2"].shape
                expected_shape = (int(spo2_shape[0]), 30)
                if valid.shape != expected_shape:
                    raise ValueError(
                        "spo2_second_valid shape must be "
                        f"{expected_shape}, got {valid.shape}"
                    )
                quality_group.create_dataset(
                    "spo2_second_valid",
                    data=valid,
                    dtype=np.dtype("bool"),
                    chunks=(1, 30),
                    compression="gzip",
                    compression_opts=4,
                    shuffle=True,
                )

            qc_names = (
                "channel_invalid_5s",
                "channel_flat_5s",
                "channel_extreme_5s",
                "channel_valid_fraction_30s",
            )
            provided_qc = [
                name for name in qc_names
                if any(f"{name}/{modality}" in quality for modality in expected_modalities)
            ]
            if provided_qc and len(provided_qc) != len(qc_names):
                missing_qc = [name for name in qc_names if name not in provided_qc]
                raise ValueError(
                    "quality channel QC datasets must be provided together; "
                    f"missing={missing_qc}"
                )
            for qc_name in qc_names if provided_qc else ():
                qc_group = quality_group.create_group(qc_name)
                for modality in expected_modalities:
                    key = f"{qc_name}/{modality}"
                    if key not in quality:
                        raise ValueError(f"missing quality dataset: {key}")
                    channel_count = len(CANONICAL_CHANNEL_NAMES[modality])
                    epoch_count = int(handle[f"signals/{modality}"].shape[0])
                    values = np.asarray(quality[key])
                    if qc_name == "channel_valid_fraction_30s":
                        expected_shape = (epoch_count, channel_count)
                        if values.shape != expected_shape:
                            raise ValueError(
                                f"{key} shape must be {expected_shape}, got {values.shape}"
                            )
                        values = np.asarray(values, dtype=np.float32)
                        if not np.isfinite(values).all() or (values < 0.0).any() or (values > 1.0).any():
                            raise ValueError(f"{key} must be finite and within [0,1]")
                        dtype = np.dtype("float32")
                        chunks = (1, channel_count)
                    else:
                        expected_shape = (epoch_count, channel_count, 6)
                        if values.shape != expected_shape:
                            raise ValueError(
                                f"{key} shape must be {expected_shape}, got {values.shape}"
                            )
                        values = np.asarray(values, dtype=np.bool_)
                        dtype = np.dtype("bool")
                        chunks = (1, channel_count, 6)
                    qc_group.create_dataset(
                        modality,
                        data=values,
                        dtype=dtype,
                        chunks=chunks,
                        compression="gzip",
                        compression_opts=4,
                        shuffle=True,
                    )
    return output


def build_waveform_cache(
    psg_path: Path | str,
    caisr_path: Path | str | None,
    output_path: Path | str,
    *,
    record_id: str,
    site: str,
    annotation_source: str | None = None,
) -> tuple[Path, object]:
    """读取一个 record 的 selected waveform 并写入主路径 signals。"""

    import pyedflib
    from waveform_preprocessing import preprocess_selected_sources

    psg_header = read_edf_header(psg_path)
    caisr_header = read_edf_header(caisr_path) if caisr_path is not None and Path(caisr_path).exists() else None
    selected_annotation_source = annotation_source
    if selected_annotation_source is None:
        selected_annotation_source = "caisr" if caisr_header is not None else "unavailable"
    source_plans = build_source_plans(
        psg_header.channel_names,
        psg_header.native_sampling_rates_hz,
        psg_header.native_sample_counts,
        psg_header.physical_dimensions,
    )
    n_epochs = int(math.floor(psg_header.duration_sec / EPOCH_DURATION_SEC))
    reader = pyedflib.EdfReader(str(psg_path))
    try:
        result = preprocess_selected_sources(
            source_plans,
            psg_header,
            read_signal=reader.readSignal,
            n_epochs=n_epochs,
        )
    finally:
        reader.close()
    output = write_schema_cache(
        output_path,
        record_id=record_id,
        site=site,
        psg_header=psg_header,
        caisr_header=caisr_header,
        spo2_scale_factor=result.scale_factors.get("spo2", 1),
        annotation_source=selected_annotation_source,
    )
    write_waveform_signals(output, result.signals, quality=result.quality)
    write_caisr_annotations_from_file(
        output,
        caisr_path,
        psg_duration_sec=psg_header.duration_sec,
        record_id=record_id,
        annotation_source=selected_annotation_source,
    )
    return output, result


def print_signal_reopen_report(path: Path | str) -> None:
    """重新打开 waveform cache，打印 signals shape/dtype/chunk/compression。"""

    cache_path = Path(path)
    with h5py.File(cache_path, "r") as handle:
        print("signals reopen report:")
        for modality in MODALITY_ORDER:
            dataset = handle[f"signals/{modality}"]
            print(
                f"  {modality}: shape={dataset.shape} dtype={dataset.dtype} "
                f"chunks={dataset.chunks} compression={dataset.compression}"
            )
    print(f"file_size_bytes: {cache_path.stat().st_size}")

def build_schema_cache(
    psg_path: Path | str,
    caisr_path: Path | str | None,
    output_path: Path | str,
    *,
    record_id: str,
    site: str,
) -> Path:
    """读取 PSG/CAISR header 并构建一个单 record 的 cache schema smoke 文件。"""

    psg_header = read_edf_header(psg_path)
    caisr_header = read_edf_header(caisr_path) if caisr_path is not None and Path(caisr_path).exists() else None
    return write_schema_cache(
        output_path,
        record_id=record_id,
        site=site,
        psg_header=psg_header,
        caisr_header=caisr_header,
    )


def _display_value(value: object) -> object:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def print_record_metadata(
    *,
    record_id: str,
    site: str,
    psg_path: Path | str,
    caisr_path: Path | str,
    psg_header: EdfHeader,
    caisr_header: EdfHeader,
) -> None:
    """打印单个 record 的 PSG/CAISR header、raw labels 和逐通道 metadata。"""

    print("record_metadata:")
    print(f"record_id: {record_id}")
    print(f"site: {site}")
    print(f"PSG path: {psg_path}")
    print(f"CAISR path: {caisr_path}")
    for name, header in (("PSG", psg_header), ("CAISR", caisr_header)):
        print(f"{name} header:")
        print(f"  start_datetime: {header.start_datetime}")
        print(f"  duration_sec: {header.duration_sec}")
        print(f"  channel_count: {len(header.channel_names)}")
        print(f"{name} raw channels:" if name == "PSG" else "CAISR channels:")
        for index, (label, rate, count, dimension) in enumerate(
            zip(
                header.channel_names,
                header.native_sampling_rates_hz,
                header.native_sample_counts,
                header.physical_dimensions,
            )
        ):
            print(
                f"  {index} | {label} | fs={float(rate)} | "
                f"samples={int(count)} | physical_dimension={dimension}"
            )


def _format_labels(values: Iterable[str]) -> str:
    return json.dumps(tuple(values), ensure_ascii=False, separators=(",", ":"))


def _format_indices(values: Iterable[int]) -> str:
    return json.dumps(tuple(values), ensure_ascii=False, separators=(",", ":"))


def print_source_plan(source_plans: dict[str, SourcePlan]) -> None:
    """打印 20 个 canonical slot 的 candidate、selected source 和解析异常。"""

    print("source-plan:")
    print("modality | canonical | present | candidates | selected | source_type | derivation | fs | samples | unit")
    for modality in MODALITY_ORDER:
        for canonical in CANONICAL_CHANNEL_NAMES[modality]:
            plan = source_plans[canonical]
            rate = "nan" if not math.isfinite(plan.native_sampling_rate_hz) else f"{plan.native_sampling_rate_hz:g}"
            print(
                f"{modality} | {canonical} | {int(plan.channel_present)} | "
                f"{_format_labels(plan.candidate_source_labels)} | "
                f"{_format_labels(plan.selected_source_labels)} | "
                f"{plan.channel_source} | {plan.derivation_method} | "
                f"{rate} | {plan.native_sample_count} | {plan.physical_dimension}"
            )
    print("RESP source-plan (7 slots):")
    print("modality | canonical | present | candidates | selected | source_type | derivation | fs | samples | unit")
    for canonical in CANONICAL_CHANNEL_NAMES["resp"]:
        plan = source_plans[canonical]
        rate = "nan" if not math.isfinite(plan.native_sampling_rate_hz) else f"{plan.native_sampling_rate_hz:g}"
        print(
            f"resp | {canonical} | {int(plan.channel_present)} | "
            f"{_format_labels(plan.candidate_source_labels)} | "
            f"{_format_labels(plan.selected_source_labels)} | "
            f"{plan.channel_source} | {plan.derivation_method} | "
            f"{rate} | {plan.native_sample_count} | {plan.physical_dimension}"
        )
    print("source-resolution anomalies:")
    anomaly_count = 0
    for modality in MODALITY_ORDER:
        for canonical in CANONICAL_CHANNEL_NAMES[modality]:
            plan = source_plans[canonical]
            for anomaly in plan.anomalies:
                print(f"  {canonical}: {anomaly}")
                anomaly_count += 1
    if anomaly_count == 0:
        print("  none")
    print("present counts:")
    for modality in MODALITY_ORDER:
        count = sum(source_plans[name].channel_present for name in CANONICAL_CHANNEL_NAMES[modality])
        print(f"  {modality}: {count}/{len(CANONICAL_CHANNEL_NAMES[modality])}")


def print_schema_report(path: Path | str) -> None:
    """打印 cache tree、attrs、timebase、canonical metadata 和 CAISR channel metadata。"""

    cache_path = Path(path)
    with h5py.File(cache_path, "r") as handle:
        print("HDF5 tree:")
        print("/")

        def visit(name: str, item: h5py.Group | h5py.Dataset) -> None:
            suffix = "" if isinstance(item, h5py.Group) else f" shape={item.shape} dtype={item.dtype}"
            print(f"/{name}{suffix}")

        handle.visititems(visit)

        print("attrs:")
        for key, value in handle.attrs.items():
            print(f"  {key}: {_display_value(value)}")

        indices = handle["timebase/epoch_index"][:]
        starts = handle["timebase/epoch_start_sec"][:]
        ends = handle["timebase/epoch_end_sec"][:]
        positions = list(range(min(5, len(indices))))
        positions.extend(range(max(5, len(indices) - 5), len(indices)))
        positions = list(dict.fromkeys(positions))
        print("timebase first/last rows:")
        for position in positions:
            print(
                f"  {int(indices[position])}: "
                f"{float(starts[position])} -> {float(ends[position])} sec"
            )

        print("canonical channel names and provenance dataset shapes/dtypes:")
        provenance_names = (
            "channel_names",
            "candidate_source_labels",
            "selected_source_labels",
            "channel_source",
            "derivation_method",
            "reference_system",
            "native_sampling_rate_hz",
            "native_sample_count",
            "physical_dimension",
        )
        for modality in MODALITY_ORDER:
            modality_group = handle[f"metadata/{modality}"]
            values = modality_group["channel_names"].asstr()[:]
            print(f"  {modality}: {values.tolist()}")
            for name in provenance_names:
                dataset = modality_group[name]
                print(f"    {name}: shape={dataset.shape} dtype={dataset.dtype}")
            present = handle[f"quality/channel_present/{modality}"]
            print(
                f"    quality/channel_present: shape={present.shape} "
                f"dtype={present.dtype} count={int(np.asarray(present[:], dtype=bool).sum())}"
            )

        caisr = handle["metadata/caisr"]
        print("CAISR channel metadata:")
        names = caisr["channel_names"].asstr()[:]
        rates = caisr["native_sampling_rate_hz"][:]
        counts = caisr["native_sample_count"][:]
        dimensions = caisr["physical_dimension"].asstr()[:]
        for name, rate, count, dimension in zip(names, rates, counts, dimensions):
            print(
                f"  {name}: fs={float(rate)}, samples={int(count)}, "
                f"physical_dimension={dimension}"
            )

    print(f"file_size_bytes: {cache_path.stat().st_size}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--psg", type=Path)
    parser.add_argument("--caisr", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--record-id")
    parser.add_argument("--site")
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="使用 header metadata 和原始标签检查模式。",
    )
    parser.add_argument(
        "--waveform-smoke",
        action="store_true",
        help="读取单个正常 record 的 selected waveform，执行主路径并写入 signals。",
    )
    parser.add_argument(
        "--source-plan-only",
        action="store_true",
        help="只读取 PSG header 并打印 20 个 canonical source plan，不写 cache。",
    )
    parser.add_argument(
        "--audit-source-slots",
        action="store_true",
        help="扫描 PSG 数据目录中的 20 个 canonical slot，并输出 warning 汇总。",
    )
    parser.add_argument(
        "--psg-root",
        type=Path,
        default=DEFAULT_PSG_ROOT,
        help="包含 site 子目录的 PSG EDF 根目录。",
    )
    parser.add_argument(
        "--expected-records",
        type=int,
        default=6600,
        help="期望扫描的 record 数量，默认 6600。",
    )
    parser.add_argument(
        "--audit-report",
        type=Path,
        default=DEFAULT_CHANNEL_INSUFFICIENT_PATH,
        help="20-slot audit 摘要输出路径，默认写入 channel_insufficient.txt。",
    )
    args = parser.parse_args()
    if args.audit_source_slots:
        audit_source_slots(
            args.psg_root,
            expected_records=args.expected_records,
            report_path=args.audit_report,
        )
        return
    required_single_record = {
        "--psg": args.psg,
        "--record-id": args.record_id,
        "--site": args.site,
    }
    if args.metadata_only:
        required_single_record["--caisr"] = args.caisr
    missing_arguments = [name for name, value in required_single_record.items() if value is None]
    if missing_arguments:
        parser.error(
            "single-record mode requires: " + ", ".join(missing_arguments)
        )
    psg_header = read_edf_header(args.psg)
    if args.source_plan_only:
        source_plans = build_source_plans(
            psg_header.channel_names,
            psg_header.native_sampling_rates_hz,
            psg_header.native_sample_counts,
            psg_header.physical_dimensions,
        )
        print(f"record_id: {args.record_id}")
        print(f"site: {args.site}")
        print(f"PSG path: {args.psg}")
        print_source_plan(source_plans)
        conflicts = find_resp_source_conflicts(source_plans)
        print("RESP selected raw-channel consistency:")
        print(f"affected_record_count: {int(bool(conflicts))}")
        print("affected_slot_pairs:")
        for raw_index, left, right in conflicts:
            print(f"  {left} <-> {right} (raw_channel_index={raw_index})")
        if not conflicts:
            print("  none")
        return
    if args.waveform_smoke:
        output, waveform_result = build_waveform_cache(
            args.psg,
            args.caisr,
            args.output,
            record_id=args.record_id,
            site=args.site,
        )
        print_source_plan(
            build_source_plans(
                psg_header.channel_names,
                psg_header.native_sampling_rates_hz,
                psg_header.native_sample_counts,
                psg_header.physical_dimensions,
            )
        )
        from waveform_preprocessing import print_raw_inspection, print_waveform_processing_report
        print_raw_inspection(waveform_result)
        print_waveform_processing_report(waveform_result)
        print_schema_report(output)
        print_signal_reopen_report(output)
        return
    caisr_header = read_edf_header(args.caisr) if args.caisr is not None and args.caisr.exists() else None
    if args.metadata_only:
        print_record_metadata(
            record_id=args.record_id,
            site=args.site,
            psg_path=args.psg,
            caisr_path=args.caisr,
            psg_header=psg_header,
            caisr_header=caisr_header,
        )
        return
    if args.output is None:
        parser.error("--output is required unless --metadata-only is set")
    source_plans = build_source_plans(
        psg_header.channel_names,
        psg_header.native_sampling_rates_hz,
        psg_header.native_sample_counts,
        psg_header.physical_dimensions,
    )
    output = write_schema_cache(
        args.output,
        record_id=args.record_id,
        site=args.site,
        psg_header=psg_header,
        caisr_header=caisr_header,
    )
    print_source_plan(source_plans)
    print_schema_report(output)


if __name__ == "__main__":
    main()


# cd /database/home/gaohaojie/workspace/psg

# PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
# conda run -n sleepfm_env python waveform_cache.py \
#   --metadata-only \
#   --record-id sub-I0002150002662_ses-1 \
#   --site I0002 \
#   --psg /database2/physionet2026_kaggle/data2/physiological_data/I0002/sub-I0002150002662_ses-1.edf \
#   --caisr /database2/physionet2026_kaggle/data2/algorithmic_annotations/I0002/sub-I0002150002662_ses-1_caisr_annotations.edf




# cd /database/home/gaohaojie/workspace/psg

# PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
# conda run -n sleepfm_env python waveform_cache.py \
#   --audit-source-slots \
#   --psg-root /database2/physionet2026_kaggle/data2/physiological_data \
#   --expected-records 6600




# cd /database/home/gaohaojie/workspace/psg

# if [ -e waveform_cache/I0002/sub-I0002150002662_ses-1.h5 ]; then
#   echo "Refusing to overwrite existing cache"
#   exit 2
# fi

# PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
# /database/home/gaohaojie/.conda/envs/sleepfm_env/bin/python waveform_cache.py \
#   --waveform-smoke \
#   --record-id sub-I0002150002662_ses-1 \
#   --site I0002 \
#   --psg /database2/physionet2026_kaggle/data2/physiological_data/I0002/sub-I0002150002662_ses-1.edf \
#   --caisr /database2/physionet2026_kaggle/data2/algorithmic_annotations/I0002/sub-I0002150002662_ses-1_caisr_annotations.edf \
#   --output waveform_cache/I0002/sub-I0002150002662_ses-1.h5
