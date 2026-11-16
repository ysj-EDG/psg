#!/usr/bin/env python3
"""只读统计 canonical v1 HDF5 cache 的记录级与站点级数值分布。

该工具不读取 EDF、不修改 HDF5、不重新计算 QC。它直接使用 canonical cache 中
已有的 waveform、source provenance、quality mask 与 CAISR annotations，生成后续
WARN 阈值讨论所需的稳健参考分布和暂定离群候选；输出不构成正式审计结论。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import h5py
import numpy as np


SCRIPT_VERSION = "0.1"
SCHEMA_PROFILE = "canonical_v1"
EXPECTED_SITE_COUNTS = {"I0002": 319, "I0006": 1142, "S0001": 5139}
SMOKE_RECORDS = {
    "I0002": "sub-I0002150026011_ses-2",
    "I0006": "sub-I0006179001674_ses-1",
    "S0001": "sub-S0001111674855_ses-1",
}
MODALITY_CHANNELS = {
    "eeg": ("EEG_F3", "EEG_F4", "EEG_C3", "EEG_C4", "EEG_O1", "EEG_O2"),
    "eog": ("EOG_E1", "EOG_E2"),
    "ecg": ("ECG",),
    "emg": ("CHIN_EMG", "LEG_EMG_LEFT", "LEG_EMG_RIGHT"),
    "resp": (
        "NASAL_PRESSURE",
        "THERMAL_AIRFLOW",
        "AIRFLOW_UNSPECIFIED",
        "PAP_FLOW",
        "PAP_PRESSURE",
        "THORACIC_EFFORT",
        "ABDOMINAL_EFFORT",
    ),
    "spo2": ("SPO2",),
}
MODALITY_SAMPLES = {"eeg": 3840, "eog": 3840, "ecg": 6000, "emg": 6000, "resp": 750, "spo2": 30}
MODALITY_ORDER = tuple(MODALITY_CHANNELS)
QC_NAMES = (
    "channel_invalid_5s",
    "channel_flat_5s",
    "channel_extreme_5s",
    "channel_valid_fraction_30s",
)
POSITIVE_LOG_METRICS = {"std", "rms", "p99_abs", "max_abs", "abs_mean_over_rms"}
BOUNDED_METRICS = {
    "channel_present",
    "finite_waveform_fraction",
    "zero_fraction",
    "valid_coverage",
    "invalid_5s_fraction",
    "flat_5s_fraction",
    "extreme_5s_fraction",
    "valid_fraction_30s_median",
    "valid_fraction_30s_p10",
    "native_support_fraction",
    "spo2_valid_second_fraction",
    "spo2_interpolated_fraction",
    "spo2_flat_fraction",
    "file_available",
    "stage_valid_fraction",
    "stage_unknown_fraction",
    "stage_prob_valid_fraction",
    "arousal_valid_fraction",
    "arousal_probability_valid_fraction",
    "resp_valid_fraction",
    "limb_valid_fraction",
    "stage_n1_fraction",
    "stage_n2_fraction",
    "stage_n3_fraction",
    "stage_rem_fraction",
    "stage_wake_fraction",
    "arousal_positive_fraction",
    "resp_positive_fraction",
    "limb_positive_fraction",
}
PROFILE_FIELDS = (
    "record_id",
    "site",
    "modality",
    "channel",
    "metric",
    "value",
    "unit",
    "source_physical_dimension",
    "source_route",
    "channel_present",
    "finite_sample_count",
    "quantile_sample_count",
)
DISTRIBUTION_FIELDS = (
    "site",
    "modality",
    "channel",
    "metric",
    "unit",
    "source_physical_dimension",
    "source_route",
    "n_original",
    "n_reference",
    "n_provisionally_excluded",
    "excluded_fraction",
    "mean_original",
    "median_original",
    "MAD_original",
    "IQR_original",
    "P01_original",
    "P05_original",
    "P25_original",
    "P75_original",
    "P95_original",
    "P99_original",
    "min_original",
    "max_original",
    "median_reference",
    "MAD_reference",
    "IQR_reference",
    "P01_reference",
    "P05_reference",
    "P25_reference",
    "P75_reference",
    "P95_reference",
    "P99_reference",
    "reference_status",
    "outlier_transform",
    "robust_scale_reference",
    "z6_candidate_count",
    "z8_excluded_count",
    "z10_candidate_count",
)
CANDIDATE_FIELDS = (
    "site",
    "modality",
    "channel",
    "metric",
    "unit",
    "source_physical_dimension",
    "source_route",
    "reference_n",
    "candidate_method",
    "candidate_lower",
    "candidate_upper",
    "estimated_flagged_n",
    "estimated_flagged_fraction",
    "reference_stability",
)
OUTLIER_FIELDS = (
    "record_id",
    "site",
    "modality",
    "channel",
    "metric",
    "unit",
    "source_physical_dimension",
    "source_route",
    "observed_value",
    "site_reference_median",
    "deviation_score",
    "candidate_reason",
    "provisionally_excluded",
)


class ProfileSchemaError(ValueError):
    """HDF5 不符合当前 canonical v1 统计契约。"""


class StreamingAccumulator:
    """单个 waveform channel 的流式有限值统计器。"""

    def __init__(self) -> None:
        self.total_count = 0
        self.finite_count = 0
        self.zero_count = 0
        self.sum = 0.0
        self.sumsq = 0.0
        self.abs_sum = 0.0
        self.max_abs = 0.0
        self.minimum = math.inf
        self.maximum = -math.inf
        self.quantile_parts: list[np.ndarray] = []

    def update(self, values: np.ndarray) -> None:
        flat = np.asarray(values, dtype=np.float32).reshape(-1)
        self.total_count += int(flat.size)
        finite = np.isfinite(flat)
        selected = flat[finite].astype(np.float64, copy=False)
        if not selected.size:
            return
        self.finite_count += int(selected.size)
        self.zero_count += int(np.count_nonzero(selected == 0.0))
        self.sum += float(selected.sum(dtype=np.float64))
        self.sumsq += float(np.square(selected, dtype=np.float64).sum(dtype=np.float64))
        absolute = np.abs(selected)
        self.abs_sum += float(absolute.sum(dtype=np.float64))
        self.max_abs = max(self.max_abs, float(absolute.max()))
        self.minimum = min(self.minimum, float(selected.min()))
        self.maximum = max(self.maximum, float(selected.max()))

    def add_quantile_sample(self, values: np.ndarray) -> None:
        """记录确定性分位数样本，不重复计入全夜 moments。"""
        flat = np.asarray(values, dtype=np.float32).reshape(-1)
        finite = np.isfinite(flat)
        if finite.any():
            self.quantile_parts.append(np.abs(flat[finite]).astype(np.float32, copy=False))

    def finish(self) -> dict[str, float | int]:
        if not self.finite_count:
            return {
                "finite_waveform_fraction": 0.0,
                "mean": math.nan,
                "std": math.nan,
                "rms": math.nan,
                "p99_abs": math.nan,
                "max_abs": math.nan,
                "zero_fraction": math.nan,
                "abs_mean_over_rms": math.nan,
                "min": math.nan,
                "max": math.nan,
                "finite_sample_count": 0,
                "quantile_sample_count": 0,
            }
        mean = self.sum / self.finite_count
        second_moment = self.sumsq / self.finite_count
        std = math.sqrt(max(0.0, second_moment - mean * mean))
        rms = math.sqrt(max(0.0, second_moment))
        if self.quantile_parts:
            sample = np.concatenate(self.quantile_parts)
            p99 = float(np.quantile(sample, 0.99))
            sample_count = int(sample.size)
        else:
            p99 = math.nan
            sample_count = 0
        return {
            "finite_waveform_fraction": self.finite_count / self.total_count if self.total_count else 0.0,
            "mean": mean,
            "std": std,
            "rms": rms,
            "p99_abs": p99,
            "max_abs": self.max_abs,
            "zero_fraction": self.zero_count / self.finite_count,
            "abs_mean_over_rms": (self.abs_sum / self.finite_count) / rms if rms > 0.0 else math.nan,
            "min": self.minimum,
            "max": self.maximum,
            "finite_sample_count": self.finite_count,
            "quantile_sample_count": sample_count,
        }


def _as_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return _as_text(value.item())
    return str(value)


def _strings(dataset: h5py.Dataset) -> list[str]:
    return [str(value) for value in dataset.asstr()[:].tolist()]


def _scalar_string(dataset: h5py.Dataset) -> str:
    return _as_text(dataset.asstr()[()])


def _finite_fraction(values: np.ndarray, predicate: np.ndarray | None = None) -> float:
    values = np.asarray(values)
    if predicate is None:
        predicate = np.ones(values.shape, dtype=bool)
    selected = np.asarray(predicate, dtype=bool)
    if not selected.any():
        return math.nan
    return float(np.mean(values[selected]))


def _quantile(values: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(np.quantile(values, q)) if values.size else math.nan


def _nan(value: float | int | np.floating[Any] | None) -> float:
    if value is None:
        return math.nan
    output = float(value)
    return output if math.isfinite(output) else math.nan


def _route(source: str, derivation: str, reference: str) -> str:
    return "|".join((source or "", derivation or "", reference or ""))


def _add_metric(
    rows: list[dict[str, Any]],
    *,
    record_id: str,
    site: str,
    modality: str,
    channel: str,
    metric: str,
    value: float | int | None,
    unit: str,
    source_physical_dimension: str = "",
    source_route: str = "",
    channel_present: bool = True,
    finite_sample_count: int = 0,
    quantile_sample_count: int = 0,
) -> None:
    rows.append(
        {
            "record_id": record_id,
            "site": site,
            "modality": modality,
            "channel": channel,
            "metric": metric,
            "value": _nan(value),
            "unit": unit,
            "source_physical_dimension": source_physical_dimension,
            "source_route": source_route,
            "channel_present": int(bool(channel_present)),
            "finite_sample_count": int(finite_sample_count),
            "quantile_sample_count": int(quantile_sample_count),
        }
    )


def _require(handle: h5py.File, path: str) -> h5py.Dataset | h5py.Group:
    if path not in handle:
        raise ProfileSchemaError(f"missing required path: /{path}")
    return handle[path]


def _validate_v1(handle: h5py.File) -> tuple[str, str, int]:
    if _as_text(handle.attrs.get("cache_schema_version", "")) != "1.0":
        raise ProfileSchemaError("cache_schema_version must be '1.0'")
    record_id = _as_text(handle.attrs.get("record_id", ""))
    site = _as_text(handle.attrs.get("site", ""))
    epochs = int(handle.attrs.get("complete_epoch_count", -1))
    if not record_id or site not in EXPECTED_SITE_COUNTS or epochs <= 0:
        raise ProfileSchemaError("invalid root record_id, site, or complete_epoch_count")
    _require(handle, "annotations/caisr")
    _require(handle, "metadata/caisr")
    _require(handle, "quality/channel_present")
    _require(handle, "quality/spo2_second_valid")
    for modality, channels in MODALITY_CHANNELS.items():
        dataset = _require(handle, f"signals/{modality}")
        expected_shape = (epochs, len(channels), MODALITY_SAMPLES[modality])
        if not isinstance(dataset, h5py.Dataset) or dataset.shape != expected_shape or dataset.dtype != np.dtype("float32"):
            raise ProfileSchemaError(
                f"signals/{modality} must be float32 {expected_shape}, got "
                f"{getattr(dataset, 'shape', None)} {getattr(dataset, 'dtype', None)}"
            )
        names = _strings(_require(handle, f"metadata/{modality}/channel_names"))
        if tuple(names) != channels:
            raise ProfileSchemaError(f"metadata/{modality}/channel_names differs from canonical v1")
        present = _require(handle, f"quality/channel_present/{modality}")
        if not isinstance(present, h5py.Dataset) or present.shape != (len(channels),):
            raise ProfileSchemaError(f"quality/channel_present/{modality} has wrong shape")
        for qc_name in QC_NAMES:
            quality = _require(handle, f"quality/{qc_name}/{modality}")
            expected = (epochs, len(channels)) if qc_name.endswith("30s") else (epochs, len(channels), 6)
            if not isinstance(quality, h5py.Dataset) or quality.shape != expected:
                raise ProfileSchemaError(f"quality/{qc_name}/{modality} has wrong shape")
    second_valid = _require(handle, "quality/spo2_second_valid")
    if not isinstance(second_valid, h5py.Dataset) or second_valid.shape != (epochs, 30):
        raise ProfileSchemaError("quality/spo2_second_valid has wrong shape")
    return record_id, site, epochs


def _sample_epoch_indices(epoch_count: int, sample_epochs: int) -> set[int]:
    if epoch_count <= 0 or sample_epochs <= 0:
        return set()
    count = min(epoch_count, sample_epochs)
    return set(np.linspace(0, epoch_count - 1, num=count, dtype=np.int64).tolist())


def _quality_for_channel(handle: h5py.File, modality: str, index: int) -> dict[str, float]:
    invalid = np.asarray(handle[f"quality/channel_invalid_5s/{modality}"][:, index, :], dtype=bool)
    flat = np.asarray(handle[f"quality/channel_flat_5s/{modality}"][:, index, :], dtype=bool)
    extreme = np.asarray(handle[f"quality/channel_extreme_5s/{modality}"][:, index, :], dtype=bool)
    coverage = np.asarray(handle[f"quality/channel_valid_fraction_30s/{modality}"][:, index], dtype=np.float64)
    return {
        "valid_coverage": float(np.mean(coverage)),
        "invalid_5s_fraction": float(np.mean(invalid)),
        "flat_5s_fraction": float(np.mean(flat)),
        "extreme_5s_fraction": float(np.mean(extreme)),
        "valid_fraction_30s_median": _quantile(coverage, 0.50),
        "valid_fraction_30s_p10": _quantile(coverage, 0.10),
    }


def _longest_run(mask: np.ndarray) -> int:
    longest = current = 0
    for value in np.asarray(mask, dtype=bool).reshape(-1):
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _profile_waveforms(
    handle: h5py.File,
    *,
    record_id: str,
    site: str,
    epochs: int,
    sample_epochs: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sample_indices = _sample_epoch_indices(epochs, sample_epochs)
    complete_duration_sec = epochs * 30.0

    for modality, canonical_names in MODALITY_CHANNELS.items():
        dataset = handle[f"signals/{modality}"]
        metadata = handle[f"metadata/{modality}"]
        present = np.asarray(handle[f"quality/channel_present/{modality}"][:], dtype=bool)
        source_units = _strings(metadata["physical_dimension"])
        source_names = _strings(metadata["channel_source"])
        derivations = _strings(metadata["derivation_method"])
        references = _strings(metadata["reference_system"])
        native_rates = np.asarray(metadata["native_sampling_rate_hz"][:], dtype=np.float64)
        native_counts = np.asarray(metadata["native_sample_count"][:], dtype=np.int64)
        canonical_unit = (
            _scalar_string(metadata["canonical_unit"])
            if modality == "spo2" and "canonical_unit" in metadata
            else None
        )
        accumulators = [StreamingAccumulator() for _ in canonical_names]
        spo2_finite_invalid_seconds = 0
        spo2_second_count = 0
        batch_epochs = max(1, min(8, epochs))
        for start in range(0, epochs, batch_epochs):
            end = min(epochs, start + batch_epochs)
            values = np.asarray(dataset[start:end, :, :], dtype=np.float32)
            second_valid_batch = (
                np.asarray(handle["quality/spo2_second_valid"][start:end, :], dtype=bool)
                if modality == "spo2"
                else None
            )
            selected_locals = [index - start for index in sample_indices if start <= index < end]
            for index, is_present in enumerate(present):
                if not is_present:
                    continue
                accumulators[index].update(values[:, index, :])
                if modality == "spo2":
                    assert second_valid_batch is not None
                    spo2_finite_invalid_seconds += int(
                        np.count_nonzero(np.isfinite(values[:, index, :]) & ~second_valid_batch)
                    )
                    spo2_second_count += int(values[:, index, :].size)
                if selected_locals:
                    accumulators[index].add_quantile_sample(values[selected_locals, index, :])

        for index, channel in enumerate(canonical_names):
            source_unit = source_units[index]
            unit = canonical_unit or source_unit
            route = _route(source_names[index], derivations[index], references[index])
            is_present = bool(present[index])
            _add_metric(
                rows,
                record_id=record_id,
                site=site,
                modality=modality,
                channel=channel,
                metric="channel_present",
                value=int(is_present),
                unit="bool",
                source_physical_dimension=source_unit,
                source_route=route,
                channel_present=is_present,
            )
            if not is_present:
                continue
            summary = accumulators[index].finish()
            metrics = (
                "mean",
                "std",
                "rms",
                "p99_abs",
                "max_abs",
                "zero_fraction",
                "finite_waveform_fraction",
            )
            if modality in {"eeg", "eog"}:
                metrics += ("abs_mean_over_rms",)
            for metric in metrics:
                _add_metric(
                    rows,
                    record_id=record_id,
                    site=site,
                    modality=modality,
                    channel=channel,
                    metric=metric,
                    value=summary[metric],
                    unit=unit if metric not in {"zero_fraction", "finite_waveform_fraction", "abs_mean_over_rms"} else "fraction",
                    source_physical_dimension=source_unit,
                    source_route=route,
                    channel_present=True,
                    finite_sample_count=int(summary["finite_sample_count"]),
                    quantile_sample_count=int(summary["quantile_sample_count"]),
                )
            for metric, value in _quality_for_channel(handle, modality, index).items():
                _add_metric(
                    rows,
                    record_id=record_id,
                    site=site,
                    modality=modality,
                    channel=channel,
                    metric=metric,
                    value=value,
                    unit="fraction",
                    source_physical_dimension=source_unit,
                    source_route=route,
                    channel_present=True,
                )
            rate = native_rates[index]
            count = native_counts[index]
            duration = (float(count) / rate) if np.isfinite(rate) and rate > 0 and count >= 0 else math.nan
            _add_metric(
                rows,
                record_id=record_id,
                site=site,
                modality=modality,
                channel=channel,
                metric="native_support_duration_sec",
                value=duration,
                unit="sec",
                source_physical_dimension=source_unit,
                source_route=route,
                channel_present=True,
            )
            _add_metric(
                rows,
                record_id=record_id,
                site=site,
                modality=modality,
                channel=channel,
                metric="native_support_fraction",
                value=(min(duration, complete_duration_sec) / complete_duration_sec if np.isfinite(duration) and complete_duration_sec > 0 else math.nan),
                unit="fraction",
                source_physical_dimension=source_unit,
                source_route=route,
                channel_present=True,
            )

            if modality == "spo2":
                second_valid = np.asarray(handle["quality/spo2_second_valid"][:], dtype=bool)
                invalid_seconds = ~second_valid
                for metric, value in {
                    "spo2_mean": summary["mean"],
                    "spo2_std": summary["std"],
                    "spo2_min": summary["min"],
                    "spo2_max": summary["max"],
                    "spo2_valid_second_fraction": float(second_valid.mean()),
                    "spo2_interpolated_fraction": (
                        spo2_finite_invalid_seconds / spo2_second_count
                        if spo2_second_count
                        else math.nan
                    ),
                    "spo2_longest_invalid_run_sec": float(_longest_run(invalid_seconds)),
                    "spo2_flat_fraction": _quality_for_channel(handle, modality, index)["flat_5s_fraction"],
                }.items():
                    _add_metric(
                        rows,
                        record_id=record_id,
                        site=site,
                        modality=modality,
                        channel=channel,
                        metric=metric,
                        value=value,
                        unit=("%" if metric in {"spo2_mean", "spo2_std", "spo2_min", "spo2_max"} else "fraction" if "fraction" in metric else "sec"),
                        source_physical_dimension=source_unit,
                        source_route=route,
                        channel_present=True,
                        finite_sample_count=int(summary["finite_sample_count"]),
                        quantile_sample_count=int(summary["quantile_sample_count"]),
                    )
    return rows


def _profile_caisr(handle: h5py.File, *, record_id: str, site: str, epochs: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metadata = handle["metadata/caisr"]
    available = bool(metadata["file_available"][()])
    base = dict(record_id=record_id, site=site, modality="caisr", channel="CAISR", unit="fraction", source_route="psg_relative_sample0", channel_present=available)
    _add_metric(rows, metric="file_available", value=int(available), unit="bool", **{key: value for key, value in base.items() if key != "unit"})
    metric_names = (
        "stage_valid_fraction",
        "stage_unknown_fraction",
        "stage_prob_valid_fraction",
        "arousal_valid_fraction",
        "arousal_probability_valid_fraction",
        "resp_valid_fraction",
        "limb_valid_fraction",
        "stage_n1_fraction",
        "stage_n2_fraction",
        "stage_n3_fraction",
        "stage_rem_fraction",
        "stage_wake_fraction",
        "arousal_positive_fraction",
        "resp_positive_fraction",
        "limb_positive_fraction",
    )
    if not available:
        for metric in metric_names:
            _add_metric(rows, metric=metric, value=math.nan, **base)
        return rows

    group = handle["annotations/caisr"]
    stage = np.asarray(group["stage"][:], dtype=np.int8)
    if stage.shape != (epochs,):
        raise ProfileSchemaError("annotations/caisr/stage has wrong shape")
    stage_valid = stage != -1
    probability_valid = np.asarray(group["stage_probability_valid"][:], dtype=bool)
    arousal = np.asarray(group["arousal"][:], dtype=np.int8)
    arousal_probability_valid = np.asarray(group["arousal_probability_valid"][:], dtype=bool)
    respiratory = np.asarray(group["respiratory"][:], dtype=np.int8)
    limb = np.asarray(group["limb"][:], dtype=np.int8)
    if arousal.shape != (epochs, 60) or respiratory.shape != (epochs, 30) or limb.shape != (epochs, 30):
        raise ProfileSchemaError("CAISR categorical annotation shape mismatch")
    if probability_valid.shape != (epochs,) or arousal_probability_valid.shape != (epochs, 60):
        raise ProfileSchemaError("CAISR probability-valid shape mismatch")
    arousal_valid = arousal != -1
    respiratory_valid = respiratory != -1
    limb_valid = limb != -1
    values = {
        "stage_valid_fraction": float(stage_valid.mean()),
        "stage_unknown_fraction": float((~stage_valid).mean()),
        "stage_prob_valid_fraction": float(probability_valid.mean()),
        "arousal_valid_fraction": float(arousal_valid.mean()),
        "arousal_probability_valid_fraction": float(arousal_probability_valid.mean()),
        "resp_valid_fraction": float(respiratory_valid.mean()),
        "limb_valid_fraction": float(limb_valid.mean()),
        "stage_n3_fraction": _finite_fraction(stage == 1, stage_valid),
        "stage_n2_fraction": _finite_fraction(stage == 2, stage_valid),
        "stage_n1_fraction": _finite_fraction(stage == 3, stage_valid),
        "stage_rem_fraction": _finite_fraction(stage == 4, stage_valid),
        "stage_wake_fraction": _finite_fraction(stage == 5, stage_valid),
        "arousal_positive_fraction": _finite_fraction(arousal == 1, arousal_valid),
        "resp_positive_fraction": _finite_fraction(respiratory > 0, respiratory_valid),
        "limb_positive_fraction": _finite_fraction(limb > 0, limb_valid),
    }
    for metric, value in values.items():
        _add_metric(rows, metric=metric, value=value, **base)
    return rows


def profile_record(
    path: Path | str,
    *,
    schema_profile: str = SCHEMA_PROFILE,
    sample_epochs: int = 48,
) -> list[dict[str, Any]]:
    """流式统计一条 canonical-v1 cache，绝不写回输入 HDF5。"""

    if schema_profile != SCHEMA_PROFILE:
        raise ValueError(f"unsupported schema profile: {schema_profile}")
    path = Path(path)
    with h5py.File(path, "r") as handle:
        record_id, site, epochs = _validate_v1(handle)
        rows = _profile_waveforms(
            handle,
            record_id=record_id,
            site=site,
            epochs=epochs,
            sample_epochs=sample_epochs,
        )
        rows.extend(_profile_caisr(handle, record_id=record_id, site=site, epochs=epochs))
    return rows


def _summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    median = _quantile(values, 0.50)
    mad = float(np.median(np.abs(values - median))) if values.size else math.nan
    return {
        "mean": float(values.mean()) if values.size else math.nan,
        "median": median,
        "mad": mad,
        "iqr": _quantile(values, 0.75) - _quantile(values, 0.25),
        "p01": _quantile(values, 0.01),
        "p05": _quantile(values, 0.05),
        "p25": _quantile(values, 0.25),
        "p75": _quantile(values, 0.75),
        "p95": _quantile(values, 0.95),
        "p99": _quantile(values, 0.99),
        "min": float(values.min()) if values.size else math.nan,
        "max": float(values.max()) if values.size else math.nan,
    }


def _robust_fit(values: np.ndarray, metric: str) -> tuple[np.ndarray, float, float, str, str]:
    """返回变换后的 values、center、scale、变换和稳定性原因。"""

    values = np.asarray(values, dtype=np.float64)
    if metric in POSITIVE_LOG_METRICS:
        transformed = np.log1p(np.maximum(values, 0.0))
        transform = "log1p"
    else:
        transformed = values
        transform = "identity"
    center = float(np.median(transformed))
    mad = float(np.median(np.abs(transformed - center)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 1e-12:
        iqr = _quantile(transformed, 0.75) - _quantile(transformed, 0.25)
        scale = iqr / 1.349 if np.isfinite(iqr) and iqr > 1e-12 else math.nan
        status = "iqr_fallback" if np.isfinite(scale) else "degenerate_scale"
    else:
        status = "mad"
    return transformed, center, scale, transform, status


def _as_output_number(value: float | int | str | bool) -> float | int | str | bool:
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return ""
    return value


def build_reference_tables(
    per_record_rows: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """按 site×slot×metric×unit×route 建立原始与稳健参考分布。"""

    grouped: dict[tuple[str, str, str, str, str, str, str], tuple[list[str], list[float]]] = {}
    for row in per_record_rows:
        try:
            value = float(row["value"])
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value):
            continue
        key = tuple(
            str(row.get(name, ""))
            for name in (
                "site",
                "modality",
                "channel",
                "metric",
                "unit",
                "source_physical_dimension",
                "source_route",
            )
        )
        record_ids, grouped_values = grouped.setdefault(key, ([], []))
        record_ids.append(sys.intern(str(row["record_id"])))
        grouped_values.append(value)

    distributions: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    outliers: list[dict[str, Any]] = []
    for key in sorted(grouped):
        site, modality, channel, metric, unit, source_physical_dimension, route = key
        record_ids, grouped_values = grouped[key]
        values = np.asarray(grouped_values, dtype=np.float64)
        original = _summary(values)
        n = int(values.size)
        bounded = metric in BOUNDED_METRICS
        transformed, center, scale, transform, scale_status = _robust_fit(values, metric)
        unstable_reason = ""
        excluded = np.zeros(n, dtype=bool)
        z = np.full(n, np.nan, dtype=np.float64)
        if bounded and (np.mean((values == 0.0) | (values == 1.0)) >= 0.80 or original["iqr"] <= 1e-12):
            unstable_reason = "bounded_mass"
        elif not np.isfinite(scale) or scale <= 1e-12:
            unstable_reason = scale_status
        else:
            z = np.abs((transformed - center) / scale)
            excluded = z > 8.0
            if n < 20:
                unstable_reason = "n_lt_20"
            elif int((~excluded).sum()) < 20:
                unstable_reason = "reference_n_lt_20_after_exclusion"

        reference = values[~excluded]
        reference_summary = _summary(reference)
        reference_status = "REFERENCE_STABLE" if not unstable_reason else f"REFERENCE_UNSTABLE:{unstable_reason}"
        z6 = int(np.count_nonzero(z > 6.0)) if np.isfinite(z).any() else 0
        z8 = int(np.count_nonzero(excluded))
        z10 = int(np.count_nonzero(z > 10.0)) if np.isfinite(z).any() else 0
        distributions.append(
            {
                "site": site,
                "modality": modality,
                "channel": channel,
                "metric": metric,
                "unit": unit,
                "source_physical_dimension": source_physical_dimension,
                "source_route": route,
                "n_original": n,
                "n_reference": int(reference.size),
                "n_provisionally_excluded": z8,
                "excluded_fraction": z8 / n if n else math.nan,
                "mean_original": original["mean"],
                "median_original": original["median"],
                "MAD_original": original["mad"],
                "IQR_original": original["iqr"],
                "P01_original": original["p01"],
                "P05_original": original["p05"],
                "P25_original": original["p25"],
                "P75_original": original["p75"],
                "P95_original": original["p95"],
                "P99_original": original["p99"],
                "min_original": original["min"],
                "max_original": original["max"],
                "median_reference": reference_summary["median"],
                "MAD_reference": reference_summary["mad"],
                "IQR_reference": reference_summary["iqr"],
                "P01_reference": reference_summary["p01"],
                "P05_reference": reference_summary["p05"],
                "P25_reference": reference_summary["p25"],
                "P75_reference": reference_summary["p75"],
                "P95_reference": reference_summary["p95"],
                "P99_reference": reference_summary["p99"],
                "reference_status": reference_status,
                "outlier_transform": transform,
                "robust_scale_reference": scale,
                "z6_candidate_count": z6,
                "z8_excluded_count": z8,
                "z10_candidate_count": z10,
            }
        )

        if not unstable_reason and np.isfinite(scale):
            for z_limit in (6, 8, 10):
                low_t = center - z_limit * scale
                high_t = center + z_limit * scale
                if transform == "log1p":
                    lower = max(0.0, math.expm1(low_t))
                    upper = math.expm1(high_t)
                else:
                    lower, upper = low_t, high_t
                flagged = (values < lower) | (values > upper)
                candidates.append(
                    {
                        "site": site,
                        "modality": modality,
                        "channel": channel,
                        "metric": metric,
                        "unit": unit,
                        "source_physical_dimension": source_physical_dimension,
                        "source_route": route,
                        "reference_n": int(reference.size),
                        "candidate_method": f"{transform}_median_scale_z{z_limit}",
                        "candidate_lower": lower,
                        "candidate_upper": upper,
                        "estimated_flagged_n": int(flagged.sum()),
                        "estimated_flagged_fraction": float(flagged.mean()),
                        "reference_stability": reference_status,
                    }
                )
        candidates.append(
            {
                "site": site,
                "modality": modality,
                "channel": channel,
                "metric": metric,
                "unit": unit,
                "source_physical_dimension": source_physical_dimension,
                "source_route": route,
                "reference_n": int(reference.size),
                "candidate_method": "reference_quantile_p01_p99",
                "candidate_lower": reference_summary["p01"],
                "candidate_upper": reference_summary["p99"],
                "estimated_flagged_n": int(((values < reference_summary["p01"]) | (values > reference_summary["p99"])).sum()) if reference.size else 0,
                "estimated_flagged_fraction": float(((values < reference_summary["p01"]) | (values > reference_summary["p99"])).mean()) if reference.size else math.nan,
                "reference_stability": reference_status,
            }
        )
        if np.isfinite(z).any():
            for index in np.flatnonzero(z > 6.0):
                outliers.append(
                    {
                        "record_id": record_ids[int(index)],
                        "site": site,
                        "modality": modality,
                        "channel": channel,
                        "metric": metric,
                        "unit": unit,
                        "source_physical_dimension": source_physical_dimension,
                        "source_route": route,
                        "observed_value": values[int(index)],
                        "site_reference_median": reference_summary["median"],
                        "deviation_score": z[int(index)],
                        "candidate_reason": f"{transform}_robust_z_gt_6",
                        "provisionally_excluded": bool(excluded[int(index)]),
                    }
                )
    return distributions, candidates, outliers


def _write_tsv_atomic(path: Path, fields: Sequence[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _as_output_number(row.get(field, "")) for field in fields})
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _record_profile_path(records_dir: Path, site: str, record_id: str) -> Path:
    return records_dir / site / f"{record_id}.tsv"


def _write_record_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    _write_tsv_atomic(path, PROFILE_FIELDS, rows)


def _profile_job(args: tuple[str, str, str, int]) -> dict[str, str]:
    path_text, record_output_text, schema_profile, sample_epochs = args
    path = Path(path_text)
    output = Path(record_output_text)
    try:
        rows = profile_record(path, schema_profile=schema_profile, sample_epochs=sample_epochs)
        _write_record_rows(output, rows)
        return {"status": "SUCCESS", "path": str(path), "profile": str(output)}
    except Exception as exc:
        return {
            "status": "PROFILE_ERROR",
            "path": str(path),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


def discover_records(cache_root: Path | str, sample_per_site: int | None = None) -> list[Path]:
    """以固定站点和排序顺序发现 cache；smoke 优先选已验证的记录。"""

    root = Path(cache_root)
    selected: list[Path] = []
    for site in EXPECTED_SITE_COUNTS:
        all_paths = sorted((root / site).glob("*.h5"))
        if sample_per_site is None:
            selected.extend(all_paths)
            continue
        if sample_per_site <= 0:
            raise ValueError("sample_per_site must be positive")
        preferred = root / site / f"{SMOKE_RECORDS[site]}.h5"
        ordered = ([preferred] if preferred.is_file() else []) + [path for path in all_paths if path != preferred]
        selected.extend(ordered[:sample_per_site])
    return selected


def _merge_record_profiles(records_dir: Path, output_path: Path) -> int:
    record_files = sorted(records_dir.glob("*/*.tsv"))
    tmp = output_path.with_name(output_path.name + ".tmp")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    with tmp.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=PROFILE_FIELDS, delimiter="\t")
        writer.writeheader()
        for path in record_files:
            with path.open("r", encoding="utf-8", newline="") as source:
                reader = csv.DictReader(source, delimiter="\t")
                for row in reader:
                    writer.writerow({field: row.get(field, "") for field in PROFILE_FIELDS})
                    row_count += 1
        destination.flush()
        os.fsync(destination.fileno())
    os.replace(tmp, output_path)
    return row_count


def _iter_profile_rows(path: Path) -> Iterable[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle, delimiter="\t")


def _site_counts(paths: Sequence[Path]) -> dict[str, int]:
    counts = Counter(path.parent.name for path in paths)
    return {site: int(counts[site]) for site in EXPECTED_SITE_COUNTS}


def _write_summary(
    path: Path,
    *,
    discovered_counts: dict[str, int],
    errors: list[dict[str, str]],
    distributions: list[dict[str, Any]],
    outliers: list[dict[str, Any]],
    provisional: bool,
) -> None:
    coverage = [row for row in distributions if row["metric"] == "channel_present"]
    unstable = [row for row in distributions if str(row["reference_status"]).startswith("REFERENCE_UNSTABLE")]
    by_site_outliers = Counter(row["site"] for row in outliers)
    lines = [
        "# Canonical HDF5 site profile",
        "",
        f"- status: {'provisional' if provisional else 'complete'}",
        f"- records discovered: {sum(discovered_counts.values())}",
        f"- profile errors: {len(errors)}",
        "",
        "## Site completion",
        "",
        "| site | profiled HDF5 | expected canonical v1 |",
        "| --- | ---: | ---: |",
    ]
    lines.extend(f"| {site} | {discovered_counts[site]} | {EXPECTED_SITE_COUNTS[site]} |" for site in EXPECTED_SITE_COUNTS)
    lines.extend(("", "## Canonical channel coverage", "", "| site | modality | channel | source unit | route | mean presence |", "| --- | --- | --- | --- | --- | ---: |"))
    for row in coverage:
        lines.append(
            f"| {row['site']} | {row['modality']} | {row['channel']} | {row['unit']} | {row['source_route']} | {float(row['mean_original']):.4f} |"
        )
    lines.extend(("", "## Reference stability", "", f"- stable metric strata: {len(distributions) - len(unstable)}", f"- REFERENCE_UNSTABLE strata: {len(unstable)}"))
    for row in unstable[:30]:
        lines.append(
            f"  - {row['site']} {row['modality']}/{row['channel']} {row['metric']} [{row['unit']}; {row['source_route']}]: {row['reference_status']}"
        )
    lines.extend(("", "## Provisional outlier candidates", "", f"- candidates (robust z > 6): {len(outliers)}"))
    lines.extend(f"  - {site}: {by_site_outliers[site]}" for site in EXPECTED_SITE_COUNTS)
    lines.append("")
    lines.append("Candidate bounds are exploratory only. No WARN_LIMITS were written by this profile.")
    _write_text_atomic(path, "\n".join(lines) + "\n")


def scan_cache(
    *,
    cache_root: Path | str,
    output_dir: Path | str,
    schema_profile: str,
    workers: int,
    sample_per_site: int | None,
    allow_partial: bool,
    sample_epochs: int = 48,
) -> dict[str, Any]:
    """可恢复地扫描 cache，返回本次运行的文件与计数摘要。"""

    if schema_profile != SCHEMA_PROFILE:
        raise ValueError(f"unsupported schema profile: {schema_profile}")
    if workers < 1 or workers > 2:
        raise ValueError("workers must be within [1, 2] to avoid competing with cache extraction")
    root = Path(cache_root).resolve()
    output = Path(output_dir).resolve()
    required_free_bytes = 5 * 1024**3
    available_bytes = shutil.disk_usage(output.parent).free
    if available_bytes < required_free_bytes:
        raise RuntimeError(
            f"insufficient output space: available={available_bytes} bytes, "
            f"required_at_least={required_free_bytes} bytes"
        )
    paths = discover_records(root, sample_per_site=sample_per_site)
    discovered_counts = _site_counts(paths)
    full_requested = sample_per_site is None
    if full_requested and not allow_partial and discovered_counts != EXPECTED_SITE_COUNTS:
        raise RuntimeError(
            f"cache is incomplete: discovered={discovered_counts}, expected={EXPECTED_SITE_COUNTS}; "
            "rerun with --allow-partial only for provisional output"
        )
    output.mkdir(parents=True, exist_ok=True)
    records_dir = output / "records"
    jobs = []
    skipped = 0
    for path in paths:
        profile_path = _record_profile_path(records_dir, path.parent.name, path.stem)
        if profile_path.is_file():
            skipped += 1
        else:
            jobs.append((str(path), str(profile_path), schema_profile, int(sample_epochs)))

    errors: list[dict[str, str]] = []
    if workers == 1:
        results = (_profile_job(job) for job in jobs)
        for result in results:
            if result["status"] != "SUCCESS":
                errors.append(result)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_profile_job, job) for job in jobs]
            for future in as_completed(futures):
                result = future.result()
                if result["status"] != "SUCCESS":
                    errors.append(result)

    per_record_path = output / "per_record_profile.tsv"
    per_record_row_count = _merge_record_profiles(records_dir, per_record_path)
    distributions, candidates, outliers = build_reference_tables(_iter_profile_rows(per_record_path))
    _write_tsv_atomic(output / "site_metric_distribution.tsv", DISTRIBUTION_FIELDS, distributions)
    _write_tsv_atomic(output / "site_metric_threshold_candidates.tsv", CANDIDATE_FIELDS, candidates)
    _write_tsv_atomic(output / "record_outlier_candidates.tsv", OUTLIER_FIELDS, outliers)
    error_text = "\n\n".join(
        f"{item['path']}\n{item['error']}\n{item.get('traceback', '')}" for item in errors
    )
    _write_text_atomic(output / "profile_errors.txt", error_text + ("\n" if error_text else ""))
    provisional = bool(sample_per_site is not None or allow_partial or discovered_counts != EXPECTED_SITE_COUNTS or errors)
    metadata = {
        "cache_root": str(root),
        "schema_profile": schema_profile,
        "site_counts": discovered_counts,
        "expected_site_counts": EXPECTED_SITE_COUNTS,
        "metric_definitions": {
            "waveform": ["mean", "std", "rms", "p99_abs", "max_abs", "zero_fraction", "valid_coverage", "invalid_5s_fraction", "flat_5s_fraction", "extreme_5s_fraction", "valid_fraction_30s_median", "valid_fraction_30s_p10"],
            "spo2": ["spo2_valid_second_fraction", "spo2_interpolated_fraction", "spo2_longest_invalid_run_sec", "spo2_flat_fraction"],
            "caisr": ["stage_valid_fraction", "stage_unknown_fraction", "stage_prob_valid_fraction", "arousal_valid_fraction", "resp_valid_fraction", "limb_valid_fraction"],
        },
        "sampling_strategy": {"p99_abs": f"all samples in {sample_epochs} evenly spaced epochs per record/channel", "sample_epochs": sample_epochs},
        "outlier_reference_method": {"transform": "log1p for positive amplitude metrics; identity otherwise", "provisional_exclusion": "absolute robust z > 8", "candidate_listing": "absolute robust z > 6", "sensitivity": [6, 8, 10]},
        "parameters": {"workers": workers, "sample_per_site": sample_per_site, "allow_partial": allow_partial, "resumed_records": skipped, "newly_profiled_records": len(jobs)},
        "script_version": SCRIPT_VERSION,
        "scan_timestamp": datetime.now(timezone.utc).isoformat(),
        "complete_or_provisional": "provisional" if provisional else "complete",
    }
    _write_summary(output / "site_profile_summary.md", discovered_counts=discovered_counts, errors=errors, distributions=distributions, outliers=outliers, provisional=provisional)
    _write_text_atomic(
        output / "profile_run_metadata.json",
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return {
        "discovered_counts": discovered_counts,
        "skipped_records": skipped,
        "profiled_records": len(jobs),
        "errors": errors,
        "per_record_rows": per_record_row_count,
        "distribution_rows": len(distributions),
        "candidate_rows": len(candidates),
        "outlier_rows": len(outliers),
        "provisional": provisional,
        "output_dir": str(output),
    }


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--schema-profile", default=SCHEMA_PROFILE, choices=(SCHEMA_PROFILE,))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--sample-per-site", type=int)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--sample-epochs", type=int, default=48)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    result = scan_cache(
        cache_root=args.cache_root,
        output_dir=args.output_dir,
        schema_profile=args.schema_profile,
        workers=args.workers,
        sample_per_site=args.sample_per_site,
        allow_partial=args.allow_partial,
        sample_epochs=args.sample_epochs,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if not result["errors"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
