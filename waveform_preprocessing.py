"""PSG canonical waveform 主路径预处理器。

本模块负责把冻结的 canonical source plan 转换为单 record 的多模态主路径输出：
读取 selected EDF source 并记录 raw 统计，对 EEG/EOG/ECG/EMG/RESP 执行各自的带通、
目标采样率转换和连续 30 秒 epoch 整理；SpO2 转交专用处理器完成每秒聚合、尺度
统一、valid mask 和线性插值。结果同时携带 waveform、source inspection、slot
状态、SpO2 quality mask 与尺度 provenance，供 cache writer 和单 record 检查器使用。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence
import warnings

import numpy as np

from channel_sources import CANONICAL_BY_MODALITY, SourcePlan
from waveform_primitives import (
    bandpass_filter,
    interpolate_nonfinite,
    pad_or_trim,
    resample_finite_signal,
)


EPOCH_DURATION_SEC = 30
TARGET_SAMPLING_RATES = {
    "eeg": 128,
    "eog": 128,
    "ecg": 200,
    "emg": 200,
    "resp": 25,
    "spo2": 1,
}
BANDPASS_RANGES_HZ = {
    "eeg": (0.3, 35.0),
    "eog": (0.3, 35.0),
    "ecg": (0.5, 50.0),
    "emg": (10.0, 90.0),
    "resp": (0.05, 5.0),
}


# CAISR annotation 作为 PSG cache 的 canonical annotation preprocessing contract。
# 这里直接复用本模块已有的 PSG-relative timebase，不维护第二套 epoch timeline。
EXPECTED_SAMPLING_RATES_HZ = {
    "stage_caisr": 1.0 / 30.0,
    "caisr_prob_n1": 1.0 / 30.0,
    "caisr_prob_n2": 1.0 / 30.0,
    "caisr_prob_n3": 1.0 / 30.0,
    "caisr_prob_r": 1.0 / 30.0,
    "caisr_prob_w": 1.0 / 30.0,
    "arousal_caisr": 2.0,
    "caisr_prob_no-ar": 2.0,
    "caisr_prob_arous": 2.0,
    "resp_caisr": 1.0,
    "limb_caisr": 1.0,
}

STAGE_PROBABILITY_ORDER = (
    "caisr_prob_n3",
    "caisr_prob_n2",
    "caisr_prob_n1",
    "caisr_prob_r",
    "caisr_prob_w",
)

MANUAL_RESPIRATORY_MAPPING = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 4,
    6: 4,
    7: 5,
    8: 1,
    9: 4,
}
MANUAL_LIMB_MAPPING = {0: 0, 2: 2}
MANUAL_LIMB_MAPPING.update({code: 1 for code in range(1, 13) if code != 2})

# (digital_min, digital_max, physical_min, physical_max) from one EDF channel.
ProbabilityCalibration = tuple[float, float, float, float]


@dataclass(frozen=True)
class CaisrAnnotationResult:
    """固定 canonical shape 的 CAISR arrays。"""

    stage: np.ndarray
    stage_probability: np.ndarray
    stage_probability_valid: np.ndarray
    arousal: np.ndarray
    arousal_probability: np.ndarray
    arousal_probability_valid: np.ndarray
    respiratory: np.ndarray
    limb: np.ndarray


def _complete_epoch_count(psg_duration_sec: float) -> int:
    duration = float(psg_duration_sec)
    if not np.isfinite(duration) or duration < 0:
        raise ValueError(f"invalid PSG duration: {psg_duration_sec!r}")
    return int(np.floor(duration / 30.0))


def _validate_sampling_rates(
    sampling_rates: Mapping[str, float] | None,
) -> dict[str, float]:
    rates = dict(EXPECTED_SAMPLING_RATES_HZ)
    if sampling_rates is not None:
        rates.update({str(name): float(rate) for name, rate in sampling_rates.items()})
    for name, expected in EXPECTED_SAMPLING_RATES_HZ.items():
        actual = rates[name]
        if not np.isfinite(actual) or not np.isclose(actual, expected, rtol=0.0, atol=1e-7):
            raise ValueError(
                f"{name} sampling rate must be {expected:g} Hz, got {actual!r}"
            )
        samples_per_epoch = 30.0 * actual
        if not np.isclose(samples_per_epoch, round(samples_per_epoch), atol=1e-7):
            raise ValueError(f"{name} has non-integral samples_per_30s: {actual!r}")
    return rates


def _as_flat(values: np.ndarray | None) -> np.ndarray:
    if values is None:
        return np.empty(0, dtype=np.float64)
    return np.asarray(values).reshape(-1)


def _categorical(
    values: np.ndarray | None,
    *,
    target_count: int,
    allowed_codes: tuple[int, ...],
    source_name: str,
    record_id: str,
    invalid_codes_to_missing: bool = False,
) -> np.ndarray:
    """验证 categorical code，并以 -1 表示缺失或无效 code。"""

    raw = _as_flat(values)
    output = np.full(target_count, -1, dtype=np.int8)
    copy_count = min(target_count, raw.size)
    if copy_count == 0:
        return output
    selected = raw[:copy_count]
    finite = np.isfinite(selected)
    if not finite.all():
        # categorical nonfinite 不是一个可解释 code；保持缺失 sentinel。
        selected_indices = np.flatnonzero(~finite)
        if selected_indices.size:
            selected = selected.copy()
            selected[selected_indices] = np.nan
    selected_float = selected.astype(np.float64, copy=False)
    rounded = np.zeros(copy_count, dtype=np.float64)
    rounded[finite] = np.rint(selected_float[finite])
    integral = finite & np.isclose(selected_float, rounded, atol=1e-6)
    unknown = finite & ~integral
    codes = rounded.astype(np.int64, copy=False)
    unknown |= finite & ~np.isin(codes, allowed_codes)
    if unknown.any():
        if not invalid_codes_to_missing:
            indices = np.flatnonzero(unknown)
            counts: dict[int, int] = {}
            for value in codes[indices].tolist():
                counts[int(value)] = counts.get(int(value), 0) + 1
            raise ValueError(
                f"{record_id}: {source_name} unexpected categorical values "
                f"{counts}; first_indices={indices[:10].tolist()}"
            )
    valid = finite & integral & ~unknown
    output_indices = np.arange(copy_count)[valid]
    output[output_indices] = codes[valid].astype(np.int8)
    return output


def _map_manual_codes(values: np.ndarray | None, mapping: Mapping[int, int]) -> np.ndarray | None:
    """把人工 EDF 的原始 subtype 映射为 canonical categorical code。"""

    if values is None:
        return None
    raw = _as_flat(values)
    output = np.full(raw.shape, np.nan, dtype=np.float64)
    for source_code, canonical_code in mapping.items():
        output[np.isfinite(raw) & np.isclose(raw, source_code, atol=1e-6)] = canonical_code
    return output


def _probability_tolerance(
    source_name: str,
    probability_calibrations: Mapping[str, ProbabilityCalibration] | None,
) -> float:
    """根据该 EDF probability channel 的量化步长返回半步容差。"""

    if not probability_calibrations or source_name not in probability_calibrations:
        return 0.0
    digital_min, digital_max, physical_min, physical_max = probability_calibrations[source_name]
    denominator = float(digital_max) - float(digital_min)
    if not np.isfinite(denominator) or denominator == 0.0:
        raise ValueError(f"{source_name}: EDF digital range is invalid")
    physical_step = abs(
        (float(physical_max) - float(physical_min)) / denominator
    )
    if not np.isfinite(physical_step):
        raise ValueError(f"{source_name}: EDF physical step is invalid")
    return 0.5 * physical_step


def _probability(
    values: np.ndarray | None,
    *,
    target_count: int,
    source_name: str,
    probability_calibrations: Mapping[str, ProbabilityCalibration] | None = None,
) -> np.ndarray:
    """按 EDF 半量化步长 canonicalize probability；不做归一化。"""

    raw = _as_flat(values).astype(np.float32, copy=False)
    output = np.full(target_count, np.nan, dtype=np.float32)
    copy_count = min(target_count, raw.size)
    if copy_count:
        selected = raw[:copy_count]
        finite = np.isfinite(selected)
        # CAISR 的 9 是 unavailable sentinel，必须先于容差和 clip 处理。
        sentinel = finite & (selected == np.float32(9.0))
        tolerance = _probability_tolerance(source_name, probability_calibrations)
        valid = (
            finite
            & ~sentinel
            & (selected >= -tolerance)
            & (selected <= 1.0 + tolerance)
        )
        valid_indices = np.flatnonzero(valid)
        if valid_indices.size:
            output[valid_indices] = np.clip(selected[valid_indices], 0.0, 1.0)
    return output


def preprocess_caisr_arrays(
    sources: Mapping[str, np.ndarray],
    *,
    psg_duration_sec: float,
    record_id: str,
    sampling_rates: Mapping[str, float] | None = None,
    probability_calibrations: Mapping[str, ProbabilityCalibration] | None = None,
    annotation_source: str = "caisr",
) -> CaisrAnnotationResult:
    """将 CAISR 或人工 annotation arrays 对齐到固定 PSG-relative canonical shape。"""

    if annotation_source not in {"caisr", "manual"}:
        raise ValueError(f"unsupported annotation_source: {annotation_source!r}")
    _validate_sampling_rates(sampling_rates)
    epochs = _complete_epoch_count(psg_duration_sec)
    is_manual = annotation_source == "manual"
    stage_values = sources.get("stage_expert" if is_manual else "stage_caisr")
    arousal_values = sources.get("arousal_expert" if is_manual else "arousal_caisr")
    respiratory_values = sources.get("resp_expert" if is_manual else "resp_caisr")
    limb_values = sources.get("limb_expert" if is_manual else "limb_caisr")
    if is_manual:
        respiratory_values = _map_manual_codes(respiratory_values, MANUAL_RESPIRATORY_MAPPING)
        limb_values = _map_manual_codes(limb_values, MANUAL_LIMB_MAPPING)
    stage = _categorical(
        stage_values,
        target_count=epochs,
        allowed_codes=(1, 2, 3, 4, 5),
        source_name="stage_caisr",
        record_id=record_id,
        # Matcha semantics: every finite stage code outside 1-5 is invalid.
        invalid_codes_to_missing=True,
    )

    stage_probabilities = np.column_stack(
        [
            _probability(
                None if is_manual else sources.get(name),
                target_count=epochs,
                source_name=name,
                probability_calibrations=probability_calibrations,
            )
            for name in STAGE_PROBABILITY_ORDER
        ]
    ).astype(np.float32, copy=False)
    stage_probability_valid = (
        np.isfinite(stage_probabilities).all(axis=1)
        & (stage_probabilities >= 0.0).all(axis=1)
        & (stage_probabilities <= 1.0).all(axis=1)
        & (np.abs(stage_probabilities.sum(axis=1) - 1.0) <= 0.01)
        & (stage != -1)
    )
    stage_probabilities[~stage_probability_valid] = np.nan

    arousal_count = epochs * 60
    arousal = _categorical(
        arousal_values,
        target_count=arousal_count,
        allowed_codes=(0, 1),
        source_name="arousal_caisr",
        record_id=record_id,
        invalid_codes_to_missing=True,
    ).reshape(epochs, 60)
    arousal_probability = np.stack(
        (
            _probability(
                None if is_manual else sources.get("caisr_prob_no-ar"),
                target_count=arousal_count,
                source_name="caisr_prob_no-ar",
                probability_calibrations=probability_calibrations,
            ),
            _probability(
                None if is_manual else sources.get("caisr_prob_arous"),
                target_count=arousal_count,
                source_name="caisr_prob_arous",
                probability_calibrations=probability_calibrations,
            ),
        ),
        axis=-1,
    ).astype(np.float32, copy=False).reshape(epochs, 60, 2)
    arousal_probability_valid = (
        np.isfinite(arousal_probability).all(axis=-1)
        & (arousal_probability >= 0.0).all(axis=-1)
        & (arousal_probability <= 1.0).all(axis=-1)
        & (np.abs(arousal_probability.sum(axis=-1) - 1.0) <= 0.01)
    )

    respiratory = _categorical(
        respiratory_values,
        target_count=epochs * 30,
        allowed_codes=(0, 1, 2, 3, 4, 5),
        source_name="resp_caisr",
        record_id=record_id,
        invalid_codes_to_missing=True,
    ).reshape(epochs, 30)
    limb = _categorical(
        limb_values,
        target_count=epochs * 30,
        allowed_codes=(0, 1, 2),
        source_name="limb_caisr",
        record_id=record_id,
        invalid_codes_to_missing=True,
    ).reshape(epochs, 30)
    return CaisrAnnotationResult(
        stage=stage,
        stage_probability=stage_probabilities,
        stage_probability_valid=stage_probability_valid.astype(np.bool_),
        arousal=arousal,
        arousal_probability=arousal_probability,
        arousal_probability_valid=arousal_probability_valid.astype(np.bool_),
        respiratory=respiratory,
        limb=limb,
    )


def read_caisr_annotations(
    path: Path | str,
    *,
    psg_duration_sec: float,
    record_id: str,
    annotation_source: str = "caisr",
) -> tuple[CaisrAnnotationResult, dict[str, object]]:
    """读取 CAISR 或人工 EDF 的指定 channels，并返回结果及 header provenance。"""

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
        if len(set(labels)) != len(labels):
            raise ValueError(f"{record_id}: duplicate CAISR channel labels prevent reliable mapping")
        by_label = {label: index for index, label in enumerate(labels)}
        label_map = {
            "stage_caisr": "stage_expert",
            "arousal_caisr": "arousal_expert",
            "resp_caisr": "resp_expert",
            "limb_caisr": "limb_expert",
        } if annotation_source == "manual" else {}
        sources: dict[str, np.ndarray] = {}
        sampling_rates: dict[str, float] = {}
        probability_calibrations: dict[str, ProbabilityCalibration] = {}
        for label, expected in EXPECTED_SAMPLING_RATES_HZ.items():
            input_label = label_map.get(label, label)
            index = by_label.get(input_label)
            if index is not None:
                sampling_rates[label] = rates[index]
                source_key = input_label if annotation_source == "manual" else label
                sources[source_key] = np.asarray(reader.readSignal(index), dtype=np.float64)
                if label.startswith("caisr_prob_"):
                    probability_calibrations[label] = (
                        float(reader.getDigitalMinimum(index)),
                        float(reader.getDigitalMaximum(index)),
                        float(reader.getPhysicalMinimum(index)),
                        float(reader.getPhysicalMaximum(index)),
                    )
        result = preprocess_caisr_arrays(
            sources,
            psg_duration_sec=psg_duration_sec,
            record_id=record_id,
            sampling_rates=sampling_rates,
            probability_calibrations=probability_calibrations,
            annotation_source=annotation_source,
        )
        metadata = {
            "path": str(path),
            "start_datetime": str(reader.getStartdatetime()),
            "duration_sec": float(reader.file_duration),
            "channel_names": labels,
            "native_sampling_rates_hz": rates,
            "native_sample_counts": counts,
            "physical_dimensions": dimensions,
            "file_available": True,
            "annotation_source": annotation_source,
        }
        return result, metadata
    finally:
        reader.close()


def missing_caisr_annotations(epochs: int) -> CaisrAnnotationResult:
    """CAISR EDF 缺失时返回固定 schema 的全缺失 arrays。"""

    if epochs < 0:
        raise ValueError("epochs must be nonnegative")
    return CaisrAnnotationResult(
        stage=np.full(epochs, -1, dtype=np.int8),
        stage_probability=np.full((epochs, 5), np.nan, dtype=np.float32),
        stage_probability_valid=np.zeros(epochs, dtype=np.bool_),
        arousal=np.full((epochs, 60), -1, dtype=np.int8),
        arousal_probability=np.full((epochs, 60, 2), np.nan, dtype=np.float32),
        arousal_probability_valid=np.zeros((epochs, 60), dtype=np.bool_),
        respiratory=np.full((epochs, 30), -1, dtype=np.int8),
        limb=np.full((epochs, 30), -1, dtype=np.int8),
    )



@dataclass(frozen=True)
class RawSourceInspection:
    """单个 selected raw EDF source 的 filter 前检查结果。"""

    canonical_name: str
    source_index: int
    raw_label: str
    native_sampling_rate_hz: float
    header_sample_count: int
    actual_read_length: int
    finite_count: int
    finite_fraction: float
    nan_count: int
    posinf_count: int
    neginf_count: int
    zero_count: int
    zero_fraction: float
    finite_min: float
    finite_p1: float
    finite_p5: float
    finite_median: float
    finite_p95: float
    finite_p99: float
    finite_max: float
    physical_dimension: str

    @property
    def has_nonfinite(self) -> bool:
        return bool(self.nan_count or self.posinf_count or self.neginf_count)


@dataclass(frozen=True)
class SlotProcessResult:
    """一个 canonical slot 的 filter/resample/epoch 结果摘要。"""

    modality: str
    canonical_name: str
    source_indices: tuple[int, ...]
    raw_lengths: tuple[int, ...]
    filtered_length: int
    resampled_length: int
    processed: bool
    reason: str
    finite_fraction: float
    first_epoch_has_finite: bool
    last_epoch_has_finite: bool


@dataclass(frozen=True)
class Spo2PreprocessResult:
    """Matcha 兼容的 SpO2 一秒聚合、valid mask 和插值结果。"""

    waveform: np.ndarray
    second_valid: np.ndarray
    scale_factor: int


def preprocess_spo2(
    raw_values: np.ndarray,
    *,
    native_sampling_rate_hz: float,
    n_epochs: int,
) -> Spo2PreprocessResult:
    """按 Matcha ``preprocessing_primitives.process_spo2`` 生成 canonical SpO2。

    该路径只做完整 PSG epoch 的 trim/pad、每秒 ``nanmedian``、尺度统一、
    生理有效性判定和 Matcha 语义的 nonfinite interpolation；不调用
    ``resample_poly``，也不做 normalization、quantization 或 clipping。
    """

    if n_epochs < 0:
        raise ValueError("n_epochs must be non-negative")
    rate = float(native_sampling_rate_hz)
    rounded_rate = int(round(rate))
    if not np.isfinite(rate) or rounded_rate <= 0 or not np.isclose(rate, rounded_rate, atol=1e-6):
        raise ValueError(f"SpO2 sampling rate is not integer-like: {native_sampling_rate_hz}")

    n_seconds = int(n_epochs) * EPOCH_DURATION_SEC
    values = pad_or_trim(
        np.asarray(raw_values, dtype=np.float64).reshape(-1),
        n_seconds * rounded_rate,
    )
    second_matrix = values.reshape(n_seconds, rounded_rate)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        per_second = np.nanmedian(second_matrix, axis=1)
    positive = per_second[np.isfinite(per_second) & (per_second > 0.0)]
    scale_indicator = float(np.median(positive)) if positive.size else float("nan")
    if np.isfinite(scale_indicator) and scale_indicator <= 1.5:
        scale_factor = 100
        per_second = per_second * scale_factor
    else:
        scale_factor = 1

    second_valid = (
        np.isfinite(per_second)
        & (per_second >= 50.0)
        & (per_second <= 100.5)
    )
    invalid_as_nan = np.where(second_valid, per_second, np.nan)
    interpolated = interpolate_nonfinite(invalid_as_nan)
    waveform = interpolated.reshape(int(n_epochs), EPOCH_DURATION_SEC).astype(
        np.float32, copy=False
    )
    valid = second_valid.reshape(int(n_epochs), EPOCH_DURATION_SEC)
    return Spo2PreprocessResult(waveform, valid, scale_factor)




@dataclass(frozen=True)
class ChannelQCResult:
    """一个 canonical channel 的 raw-support/flat/extreme QC 结果。"""

    invalid_5s: np.ndarray
    flat_5s: np.ndarray
    extreme_5s: np.ndarray
    valid_fraction_30s: np.ndarray


QC_SUBWINDOW_SEC = 5
QC_SUBWINDOWS_PER_EPOCH = EPOCH_DURATION_SEC // QC_SUBWINDOW_SEC
QC_INVALID_SUPPORT_THRESHOLD = 0.99
QC_FLAT_STD_THRESHOLD = 1e-12
QC_FLAT_ZERO_RATIO_THRESHOLD = 0.995
QC_FLAT_DIFFERENCE_RATIO_THRESHOLD = 0.9995
QC_EXTREME_ROBUST_Z = {"eeg": 7.0, "eog": 7.0, "emg": 8.0, "resp": 7.0}
QC_ABSOLUTE_PEAK_THRESHOLDS = {"eeg": 50.0, "eog": 50.0, "emg": 100.0, "resp": 50.0}
QC_TRANSIENT_PEAK_THRESHOLDS = {"eeg": 15.0, "eog": 15.0, "emg": 30.0, "resp": 15.0}
QC_TRANSIENT_CREST_THRESHOLDS = {"eeg": 8.0, "eog": 8.0, "emg": 10.0, "resp": 8.0}
QC_TRANSIENT_DIFF_PEAK_THRESHOLDS = {"eeg": 20.0, "eog": 20.0, "emg": 40.0, "resp": 20.0}


def structural_missing_qc(n_epochs: int) -> ChannelQCResult:
    """返回 structural-missing slot 的固定 QC contract。"""

    if n_epochs < 0:
        raise ValueError("n_epochs must be non-negative")
    shape = (int(n_epochs), QC_SUBWINDOWS_PER_EPOCH)
    return ChannelQCResult(
        invalid_5s=np.ones(shape, dtype=bool),
        flat_5s=np.zeros(shape, dtype=bool),
        extreme_5s=np.zeros(shape, dtype=bool),
        valid_fraction_30s=np.zeros(int(n_epochs), dtype=np.float32),
    )


def _raw_qc_target_lengths(native_sampling_rate_hz: float, n_epochs: int) -> tuple[int, int]:
    rate = float(native_sampling_rate_hz)
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError(f"native sampling rate must be finite and positive: {rate!r}")
    target_length = int(round(n_epochs * EPOCH_DURATION_SEC * rate))
    samples_per_5s_float = QC_SUBWINDOW_SEC * rate
    samples_per_5s = int(round(samples_per_5s_float))
    if not np.isclose(samples_per_5s_float, samples_per_5s, atol=1e-5):
        raise ValueError(f"sampling rate does not map to 5-second windows: {rate}")
    return target_length, samples_per_5s


def _pad_or_trim_support(values: np.ndarray, target_length: int) -> np.ndarray:
    values = np.asarray(values, dtype=bool).reshape(-1)
    output = np.zeros(int(target_length), dtype=bool)
    output[: min(values.size, output.size)] = values[: output.size]
    return output


def _pad_or_trim_raw_for_qc(values: np.ndarray, target_length: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    output = np.full(int(target_length), np.nan, dtype=np.float64)
    output[: min(values.size, output.size)] = values[: output.size]
    return output


def _raw_window_metrics(windows: np.ndarray) -> dict[str, np.ndarray]:
    windows = np.asarray(windows, dtype=np.float64)
    finite = np.isfinite(windows)
    finite_count = finite.sum(axis=-1)
    safe = np.where(finite, windows, 0.0)
    standard_deviation = np.sqrt(
        np.divide(
            np.square(np.where(finite, windows - np.divide(
                safe.sum(axis=-1), finite_count,
                out=np.zeros_like(finite_count, dtype=np.float64), where=finite_count > 0
            )[:, None], 0.0)).sum(axis=-1),
            finite_count,
            out=np.zeros_like(finite_count, dtype=np.float64),
            where=finite_count > 0,
        )
    )
    zero_count = (finite & np.isclose(windows, 0.0, atol=1e-8)).sum(axis=-1)
    zero_ratio = np.divide(
        zero_count,
        finite_count,
        out=np.ones_like(finite_count, dtype=np.float64),
        where=finite_count > 0,
    )
    if windows.shape[-1] > 1:
        left = windows[..., :-1]
        right = windows[..., 1:]
        valid_pairs = np.isfinite(left) & np.isfinite(right)
        valid_pair_count = valid_pairs.sum(axis=-1)
        equal_pair_count = (valid_pairs & (left == right)).sum(axis=-1)
        flat_difference_ratio = np.divide(
            equal_pair_count,
            valid_pair_count,
            out=np.ones_like(valid_pair_count, dtype=np.float64),
            where=valid_pair_count > 0,
        )
        difference = np.where(valid_pairs, right - left, 0.0)
        difference_rms = np.sqrt(
            np.divide(
                np.square(difference).sum(axis=-1),
                valid_pair_count,
                out=np.zeros_like(valid_pair_count, dtype=np.float64),
                where=valid_pair_count > 0,
            )
        )
        difference_peak = np.max(np.abs(difference), axis=-1)
    else:
        flat_difference_ratio = np.ones(windows.shape[0], dtype=np.float64)
        difference_rms = np.zeros(windows.shape[0], dtype=np.float64)
        difference_peak = np.zeros(windows.shape[0], dtype=np.float64)
    return {
        "standard_deviation": standard_deviation,
        "zero_ratio": zero_ratio,
        "flat_difference_ratio": flat_difference_ratio,
        "rms": np.sqrt(
            np.divide(
                np.square(safe).sum(axis=-1),
                finite_count,
                out=np.zeros_like(finite_count, dtype=np.float64),
                where=finite_count > 0,
            )
        ),
        "peak": np.max(np.abs(safe), axis=-1),
        "difference_rms": difference_rms,
        "difference_peak": difference_peak,
        "crest_factor": np.divide(
            np.max(np.abs(safe), axis=-1),
            np.sqrt(np.divide(
                np.square(safe).sum(axis=-1), finite_count,
                out=np.zeros_like(finite_count, dtype=np.float64), where=finite_count > 0
            )),
            out=np.zeros_like(finite_count, dtype=np.float64),
            where=np.sqrt(np.divide(
                np.square(safe).sum(axis=-1), finite_count,
                out=np.zeros_like(finite_count, dtype=np.float64), where=finite_count > 0
            )) > 1e-8,
        ),
    }


def calculate_raw_channel_qc(
    raw_values: Sequence[np.ndarray],
    *,
    native_sampling_rate_hz: float,
    n_epochs: int,
) -> ChannelQCResult:
    """基于 raw canonical support 计算 invalid、flat 和 30 秒 valid fraction。

    paired source 的 support 严格使用 component A AND component B；本函数只读输入，
    不插值、不滤波、不重采样，也不改变任何 waveform 数值。
    """

    if len(raw_values) not in (1, 2):
        raise ValueError(f"raw_values must contain one or two components: {len(raw_values)}")
    target_length, samples_per_5s = _raw_qc_target_lengths(native_sampling_rate_hz, n_epochs)
    components = [np.asarray(value, dtype=np.float64).reshape(-1) for value in raw_values]
    common_length = min(component.size for component in components)
    if len(components) == 1:
        canonical = components[0]
        support = np.isfinite(canonical)
    else:
        with np.errstate(invalid="ignore", over="ignore"):
            canonical = components[0][:common_length] - components[1][:common_length]
        support = np.isfinite(components[0][:common_length]) & np.isfinite(components[1][:common_length])
    support = _pad_or_trim_support(support, target_length)
    canonical = _pad_or_trim_raw_for_qc(canonical, target_length)
    if n_epochs == 0:
        empty = np.zeros((0, QC_SUBWINDOWS_PER_EPOCH), dtype=bool)
        return ChannelQCResult(empty, empty.copy(), empty.copy(), np.zeros((0,), dtype=np.float32))
    support_windows = support.reshape(n_epochs, QC_SUBWINDOWS_PER_EPOCH, samples_per_5s)
    support_fraction = support_windows.mean(axis=-1)
    invalid_5s = support_fraction < QC_INVALID_SUPPORT_THRESHOLD
    raw_windows = canonical.reshape(n_epochs, QC_SUBWINDOWS_PER_EPOCH, samples_per_5s).reshape(
        n_epochs * QC_SUBWINDOWS_PER_EPOCH, samples_per_5s
    )
    metrics = _raw_window_metrics(raw_windows)
    flat_condition = (
        (metrics["standard_deviation"] <= QC_FLAT_STD_THRESHOLD)
        | (metrics["zero_ratio"] >= QC_FLAT_ZERO_RATIO_THRESHOLD)
        | (metrics["flat_difference_ratio"] >= QC_FLAT_DIFFERENCE_RATIO_THRESHOLD)
    ).reshape(n_epochs, QC_SUBWINDOWS_PER_EPOCH)
    # 数据支持不足的窗口只标记 invalid，不重复解释为 flat。
    flat_5s = flat_condition & (~invalid_5s)
    return ChannelQCResult(
        invalid_5s=invalid_5s.astype(bool),
        flat_5s=flat_5s.astype(bool),
        extreme_5s=np.zeros_like(invalid_5s, dtype=bool),
        valid_fraction_30s=support.reshape(n_epochs, -1).mean(axis=1).astype(np.float32),
    )


def calculate_spo2_qc(second_valid: np.ndarray) -> ChannelQCResult:
    """把 SpO2 的 1 Hz second-valid mask 聚合成 P1-E QC shape。"""

    second_valid = np.asarray(second_valid, dtype=bool)
    if second_valid.ndim != 2 or second_valid.shape[1] != EPOCH_DURATION_SEC:
        raise ValueError(f"second_valid must have shape (T,30), got {second_valid.shape}")
    n_epochs = second_valid.shape[0]
    valid_5s = second_valid.reshape(n_epochs, QC_SUBWINDOWS_PER_EPOCH, QC_SUBWINDOW_SEC).mean(axis=-1) >= 0.80
    return ChannelQCResult(
        invalid_5s=(~valid_5s).astype(bool),
        flat_5s=np.zeros_like(valid_5s, dtype=bool),
        extreme_5s=np.zeros_like(valid_5s, dtype=bool),
        valid_fraction_30s=second_valid.mean(axis=1).astype(np.float32),
    )


def _robust_upper_outlier(values: np.ndarray, valid: np.ndarray, robust_z: float) -> np.ndarray:
    output = np.zeros(values.shape, dtype=bool)
    selected = valid & np.isfinite(values)
    selected_values = values[selected]
    if selected_values.size < 20:
        return output
    transformed = np.log1p(np.maximum(selected_values, 0.0))
    median = float(np.median(transformed))
    scale = 1.4826 * float(np.median(np.abs(transformed - median)))
    if not np.isfinite(scale) or scale <= 1e-8:
        scale = float(np.std(transformed))
    if not np.isfinite(scale) or scale <= 1e-8:
        return output
    threshold = median + robust_z * scale
    transformed_all = np.log1p(np.maximum(values, 0.0))
    output[selected] = transformed_all[selected] > threshold
    return output


def calculate_extreme_activity_5s(
    epoch_signal: np.ndarray,
    invalid_5s: np.ndarray,
    flat_5s: np.ndarray,
    modality: str,
    sampling_rate: int,
) -> np.ndarray:
    """在 temporary robust-normalized copy 上计算 soft extreme flag。"""

    invalid_5s = np.asarray(invalid_5s, dtype=bool)
    flat_5s = np.asarray(flat_5s, dtype=bool)
    if invalid_5s.shape != flat_5s.shape or invalid_5s.ndim != 2 or invalid_5s.shape[1] != QC_SUBWINDOWS_PER_EPOCH:
        raise ValueError("invalid_5s and flat_5s must have shape (T,6)")
    if modality == "ecg":
        return np.zeros_like(invalid_5s, dtype=bool)
    epoch_signal = np.asarray(epoch_signal, dtype=np.float64)
    n_epochs = invalid_5s.shape[0]
    expected = n_epochs * EPOCH_DURATION_SEC * int(sampling_rate)
    if epoch_signal.shape != (n_epochs, EPOCH_DURATION_SEC * int(sampling_rate)):
        raise ValueError(f"epoch_signal shape does not match QC mask: {epoch_signal.shape}")
    samples_per_5s = QC_SUBWINDOW_SEC * int(sampling_rate)
    windows = epoch_signal.reshape(n_epochs, QC_SUBWINDOWS_PER_EPOCH, samples_per_5s).reshape(
        n_epochs * QC_SUBWINDOWS_PER_EPOCH, samples_per_5s
    )
    baseline = (~invalid_5s & ~flat_5s).reshape(-1)
    baseline_values = windows[baseline]
    baseline_values = baseline_values[np.isfinite(baseline_values)]
    if baseline_values.size == 0:
        return np.zeros_like(invalid_5s, dtype=bool)
    center = float(np.median(baseline_values))
    scale = 1.4826 * float(np.median(np.abs(baseline_values - center)))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(np.std(baseline_values))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
    normalized = (windows - center) / scale
    metrics = _raw_window_metrics(normalized)
    robust_z = QC_EXTREME_ROBUST_Z[modality]
    extreme = (
        _robust_upper_outlier(metrics["rms"], baseline, robust_z)
        | _robust_upper_outlier(metrics["peak"], baseline, robust_z)
        | _robust_upper_outlier(metrics["difference_rms"], baseline, robust_z)
        | _robust_upper_outlier(metrics["difference_peak"], baseline, robust_z)
    )
    extreme |= metrics["peak"] > QC_ABSOLUTE_PEAK_THRESHOLDS[modality]
    extreme |= (metrics["peak"] > QC_TRANSIENT_PEAK_THRESHOLDS[modality]) & (
        metrics["crest_factor"] > QC_TRANSIENT_CREST_THRESHOLDS[modality]
    )
    extreme |= metrics["difference_peak"] > QC_TRANSIENT_DIFF_PEAK_THRESHOLDS[modality]
    return (extreme & baseline).reshape(n_epochs, QC_SUBWINDOWS_PER_EPOCH)


@dataclass(frozen=True)
class WaveformPreprocessResult:
    """单 record waveform 主路径的数组和可审计摘要。"""

    signals: dict[str, np.ndarray]
    source_inspections: tuple[RawSourceInspection, ...]
    slot_results: dict[str, SlotProcessResult]
    n_epochs: int
    quality: dict[str, np.ndarray] = field(default_factory=dict)
    scale_factors: dict[str, int] = field(default_factory=dict)
    canonical_units: dict[str, str] = field(default_factory=dict)


def derive_canonical_raw(raw_values: Sequence[np.ndarray]) -> np.ndarray:
    """按已冻结的 source plan 规则构造 native canonical waveform。

    direct source 原样返回；paired/bipolar source 保持现有的
    ``positive - negative`` 数值语义。该函数只是把原有内联运算提取为
    inspection 工具可复用的纯函数，不改变任何数值处理行为。
    """

    if len(raw_values) == 1:
        return np.asarray(raw_values[0], dtype=np.float64).reshape(-1)
    if len(raw_values) == 2:
        left = np.asarray(raw_values[0], dtype=np.float64).reshape(-1)
        right = np.asarray(raw_values[1], dtype=np.float64).reshape(-1)
        if left.size != right.size:
            raise ValueError("paired source sample counts must match")
        return left - right
    raise ValueError(f"unsupported selected source count: {len(raw_values)}")


def _percentile(values: np.ndarray, percentile: float) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.percentile(finite, percentile))


def inspect_raw_source(
    *,
    canonical_name: str,
    source_index: int,
    raw_label: str,
    values: np.ndarray,
    native_sampling_rate_hz: float,
    header_sample_count: int,
    physical_dimension: str,
) -> RawSourceInspection:
    """计算 raw source 统计；此函数不修改输入，也不进行 gap 修复。"""

    values = np.asarray(values, dtype=np.float64).reshape(-1)
    finite_mask = np.isfinite(values)
    finite_count = int(finite_mask.sum())
    size = int(values.size)
    zero_count = int(np.count_nonzero(values == 0.0))
    return RawSourceInspection(
        canonical_name=canonical_name,
        source_index=int(source_index),
        raw_label=str(raw_label),
        native_sampling_rate_hz=float(native_sampling_rate_hz),
        header_sample_count=int(header_sample_count),
        actual_read_length=size,
        finite_count=finite_count,
        finite_fraction=float(finite_count / size) if size else 0.0,
        nan_count=int(np.isnan(values).sum()),
        posinf_count=int(np.isposinf(values).sum()),
        neginf_count=int(np.isneginf(values).sum()),
        zero_count=zero_count,
        zero_fraction=float(zero_count / size) if size else 0.0,
        finite_min=_percentile(values, 0.0),
        finite_p1=_percentile(values, 1.0),
        finite_p5=_percentile(values, 5.0),
        finite_median=_percentile(values, 50.0),
        finite_p95=_percentile(values, 95.0),
        finite_p99=_percentile(values, 99.0),
        finite_max=_percentile(values, 100.0),
        physical_dimension=str(physical_dimension),
    )


def _empty_signal(n_epochs: int, target_rate: int) -> np.ndarray:
    return np.full(
        (n_epochs, EPOCH_DURATION_SEC * target_rate),
        np.nan,
        dtype=np.float32,
    )


def _slot_output(
    values: np.ndarray,
    *,
    n_epochs: int,
    target_rate: int,
) -> np.ndarray:
    target_length = n_epochs * EPOCH_DURATION_SEC * target_rate
    output = np.asarray(values, dtype=np.float64).reshape(-1)
    if output.size != target_length:
        raise ValueError(
            f"processed length {output.size} does not match target length {target_length}"
        )
    return output.reshape(n_epochs, EPOCH_DURATION_SEC * target_rate).astype(
        np.float32, copy=False
    )


def preprocess_selected_sources(
    source_plans: Mapping[str, SourcePlan],
    header: object,
    *,
    read_signal: Callable[[int], np.ndarray],
    n_epochs: int,
) -> WaveformPreprocessResult:
    """读取并处理 source plan 中 selected raw channels。

    ``read_signal`` 是唯一的 waveform 读取入口。调用方传入 EDF reader 的
    ``readSignal`` 方法即可；本函数不会遍历或读取未 selected 的 candidates。
    EEG/EOG/ECG/EMG/RESP raw source 只要包含 NaN/Inf，就记录 inspection 并停止
    该 slot 的普通处理，不调用 interpolate_nonfinite，也不引入 gap threshold。
    SpO2 在统一主入口内使用 Matcha 兼容的每秒 nanmedian、尺度统一和线性插值路径。
    """

    if n_epochs < 0:
        raise ValueError("n_epochs must be non-negative")
    labels = tuple(str(value) for value in header.channel_names)
    rates = tuple(float(value) for value in header.native_sampling_rates_hz)
    counts = tuple(int(value) for value in header.native_sample_counts)
    dimensions = tuple(str(value) for value in header.physical_dimensions)
    if not (len(labels) == len(rates) == len(counts) == len(dimensions)):
        raise ValueError("header channel metadata lengths do not match")

    signals: dict[str, np.ndarray] = {}
    inspections: list[RawSourceInspection] = []
    slot_results: dict[str, SlotProcessResult] = {}
    quality: dict[str, np.ndarray] = {
        "spo2_second_valid": np.zeros((n_epochs, 30), dtype=bool)
    }
    scale_factors: dict[str, int] = {"spo2": 1}
    canonical_units: dict[str, str] = {"spo2": "%"}

    for modality, canonical_names in CANONICAL_BY_MODALITY.items():
        target_rate = TARGET_SAMPLING_RATES[modality]
        modality_signal = np.full(
            (n_epochs, len(canonical_names), EPOCH_DURATION_SEC * target_rate),
            np.nan,
            dtype=np.float32,
        )
        modality_invalid_5s = np.ones(
            (n_epochs, len(canonical_names), QC_SUBWINDOWS_PER_EPOCH), dtype=bool
        )
        modality_flat_5s = np.zeros_like(modality_invalid_5s, dtype=bool)
        modality_extreme_5s = np.zeros_like(modality_invalid_5s, dtype=bool)
        modality_valid_fraction_30s = np.zeros(
            (n_epochs, len(canonical_names)), dtype=np.float32
        )
        for slot_index, canonical_name in enumerate(canonical_names):
            plan = source_plans[canonical_name]
            indices = tuple(int(index) for index in plan.selected_source_indices)
            if not plan.channel_present or not indices:
                slot_results[canonical_name] = SlotProcessResult(
                    modality, canonical_name, indices, (), 0, 0, False,
                    "source_unavailable", 0.0, False, False,
                )
                continue

            raw_values: list[np.ndarray] = []
            invalid = False
            raw_lengths: list[int] = []
            for source_index in indices:
                if source_index < 0 or source_index >= len(labels):
                    invalid = True
                    continue
                values = np.asarray(read_signal(source_index), dtype=np.float64).reshape(-1)
                raw_values.append(values)
                raw_lengths.append(int(values.size))
                inspection = inspect_raw_source(
                    canonical_name=canonical_name,
                    source_index=source_index,
                    raw_label=labels[source_index],
                    values=values,
                    native_sampling_rate_hz=rates[source_index],
                    header_sample_count=counts[source_index],
                    physical_dimension=dimensions[source_index],
                )
                inspections.append(inspection)
                # 其他模态遇到 nonfinite 时保持 P1-D 的停止语义；SpO2
                # 则交给专用每秒聚合/插值路径处理，并保留 raw inspection。
                if modality != "spo2":
                    invalid = invalid or inspection.has_nonfinite

            if modality != "spo2":
                if len(raw_values) == len(indices):
                    try:
                        raw_qc = calculate_raw_channel_qc(
                            raw_values,
                            native_sampling_rate_hz=plan.native_sampling_rate_hz,
                            n_epochs=n_epochs,
                        )
                    except ValueError:
                        raw_qc = structural_missing_qc(n_epochs)
                else:
                    raw_qc = structural_missing_qc(n_epochs)
                modality_invalid_5s[:, slot_index, :] = raw_qc.invalid_5s
                modality_flat_5s[:, slot_index, :] = raw_qc.flat_5s
                modality_valid_fraction_30s[:, slot_index] = raw_qc.valid_fraction_30s

            if invalid or len(raw_values) != len(indices):
                slot_results[canonical_name] = SlotProcessResult(
                    modality, canonical_name, indices, tuple(raw_lengths), 0, 0,
                    False, "raw_nonfinite" if invalid else "source_read_error",
                    0.0, False, False,
                )
                continue

            if len(indices) > 1:
                if not np.isfinite(plan.native_sampling_rate_hz):
                    reason = "source_metadata_inconsistent"
                    slot_results[canonical_name] = SlotProcessResult(
                        modality, canonical_name, indices, tuple(raw_lengths), 0, 0,
                        False, reason, 0.0, False, False,
                    )
                    continue
                if len({raw.size for raw in raw_values}) != 1:
                    slot_results[canonical_name] = SlotProcessResult(
                        modality, canonical_name, indices, tuple(raw_lengths), 0, 0,
                        False, "source_sample_count_inconsistent", 0.0, False, False,
                    )
                    continue
                native_values = derive_canonical_raw(raw_values)
            else:
                native_values = derive_canonical_raw(raw_values)

            if modality == "spo2":
                spo2_result = preprocess_spo2(
                    native_values,
                    native_sampling_rate_hz=plan.native_sampling_rate_hz,
                    n_epochs=n_epochs,
                )
                epoch_values = spo2_result.waveform
                modality_signal[:, slot_index, :] = epoch_values
                spo2_qc = calculate_spo2_qc(spo2_result.second_valid)
                modality_invalid_5s[:, slot_index, :] = spo2_qc.invalid_5s
                modality_flat_5s[:, slot_index, :] = spo2_qc.flat_5s
                modality_extreme_5s[:, slot_index, :] = spo2_qc.extreme_5s
                modality_valid_fraction_30s[:, slot_index] = spo2_qc.valid_fraction_30s
                quality["spo2_second_valid"] = spo2_result.second_valid
                scale_factors["spo2"] = int(spo2_result.scale_factor)
                finite_fraction = (
                    float(np.isfinite(epoch_values).mean())
                    if epoch_values.size
                    else 0.0
                )
                slot_results[canonical_name] = SlotProcessResult(
                    modality,
                    canonical_name,
                    indices,
                    tuple(raw_lengths),
                    int(native_values.size),
                    int(epoch_values.size),
                    bool(np.isfinite(epoch_values).any()),
                    "" if np.isfinite(epoch_values).any() else "all_nonfinite",
                    finite_fraction,
                    bool(np.isfinite(epoch_values[0]).any()) if n_epochs else False,
                    bool(np.isfinite(epoch_values[-1]).any()) if n_epochs else False,
                )
                continue

            if modality in BANDPASS_RANGES_HZ:
                low_hz, high_hz = BANDPASS_RANGES_HZ[modality]
                filtered = bandpass_filter(
                    native_values, plan.native_sampling_rate_hz, low_hz, high_hz
                )
            else:
                filtered = native_values.copy()
            filtered_length = int(filtered.size)
            if not np.isfinite(filtered).all():
                slot_results[canonical_name] = SlotProcessResult(
                    modality, canonical_name, indices, tuple(raw_lengths),
                    filtered_length, 0, False, "post_filter_nonfinite", 0.0, False, False,
                )
                continue

            resampled = resample_finite_signal(
                filtered,
                plan.native_sampling_rate_hz,
                target_rate,
                n_epochs * EPOCH_DURATION_SEC * target_rate,
            )
            resampled_length = int(resampled.size)
            if not np.isfinite(resampled).all():
                slot_results[canonical_name] = SlotProcessResult(
                    modality, canonical_name, indices, tuple(raw_lengths),
                    filtered_length, resampled_length, False,
                    "post_resample_nonfinite", 0.0, False, False,
                )
                continue

            epoch_values = _slot_output(
                resampled, n_epochs=n_epochs, target_rate=target_rate
            )
            modality_signal[:, slot_index, :] = epoch_values
            modality_extreme_5s[:, slot_index, :] = calculate_extreme_activity_5s(
                epoch_values,
                modality_invalid_5s[:, slot_index, :],
                modality_flat_5s[:, slot_index, :],
                modality,
                target_rate,
            )
            slot_results[canonical_name] = SlotProcessResult(
                modality, canonical_name, indices, tuple(raw_lengths),
                filtered_length, resampled_length, True, "",
                float(np.isfinite(epoch_values).mean()) if epoch_values.size else 0.0,
                bool(np.isfinite(epoch_values[0]).any()) if n_epochs else False,
                bool(np.isfinite(epoch_values[-1]).any()) if n_epochs else False,
            )
        signals[modality] = modality_signal
        quality[f"channel_invalid_5s/{modality}"] = modality_invalid_5s
        quality[f"channel_flat_5s/{modality}"] = modality_flat_5s
        quality[f"channel_extreme_5s/{modality}"] = modality_extreme_5s
        quality[f"channel_valid_fraction_30s/{modality}"] = np.clip(
            modality_valid_fraction_30s.astype(np.float32), 0.0, 1.0
        )

    return WaveformPreprocessResult(
        signals=signals,
        source_inspections=tuple(inspections),
        slot_results=slot_results,
        n_epochs=n_epochs,
        quality=quality,
        scale_factors=scale_factors,
        canonical_units=canonical_units,
    )


def print_raw_inspection(result: WaveformPreprocessResult) -> None:
    """打印每个 selected source 的 filter 前统计。"""

    print("raw waveform inspection:")
    print(
        "canonical | raw_label | index | native_fs | header_samples | read_length | "
        "finite_count | finite_fraction | nan | +inf | -inf | zero_count | "
        "zero_fraction | finite_min | finite_max | finite_median"
    )
    for item in result.source_inspections:
        print(
            f"{item.canonical_name} | {item.raw_label} | {item.source_index} | "
            f"{item.native_sampling_rate_hz:g} | {item.header_sample_count} | "
            f"{item.actual_read_length} | {item.finite_count} | "
            f"{item.finite_fraction:.6f} | {item.nan_count} | "
            f"{item.posinf_count} | {item.neginf_count} | {item.zero_count} | "
            f"{item.zero_fraction:.6f} | {item.finite_min:g} | "
            f"{item.finite_max:g} | {item.finite_median:g}"
        )

    spo2 = [item for item in result.source_inspections if item.canonical_name == "SPO2"]
    if spo2:
        item = spo2[0]
        print("SpO2 raw summary:")
        print(
            f"fs={item.native_sampling_rate_hz:g} "
            f"finite_fraction={item.finite_fraction:.6f} "
            f"min={item.finite_min:g} p1={item.finite_p1:g} "
            f"p5={item.finite_p5:g} p50={item.finite_median:g} "
            f"p95={item.finite_p95:g} p99={item.finite_p99:g} "
            f"max={item.finite_max:g} unit={item.physical_dimension}"
        )


def print_waveform_processing_report(result: WaveformPreprocessResult) -> None:
    """打印 slot 输出 finite fraction、边界 epoch 和长度变化。"""

    print("waveform processing report:")
    print(
        "canonical | processed | reason | raw_lengths | filtered_length | "
        "resampled_length | output_finite_fraction | first_epoch_finite | "
        "last_epoch_finite"
    )
    for modality, canonical_names in CANONICAL_BY_MODALITY.items():
        for canonical_name in canonical_names:
            item = result.slot_results[canonical_name]
            print(
                f"{canonical_name} | {int(item.processed)} | {item.reason or '-'} | "
                f"{item.raw_lengths} | {item.filtered_length} | "
                f"{item.resampled_length} | {item.finite_fraction:.6f} | "
                f"{int(item.first_epoch_has_finite)} | "
                f"{int(item.last_epoch_has_finite)}"
            )
