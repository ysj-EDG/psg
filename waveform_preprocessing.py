"""PSG canonical waveform 主路径预处理器。

本模块把已经冻结的 canonical source plan 连接到单 record 的 waveform 主路径：
按 selected raw source 读取 EDF，先打印 raw finite/zero/range 统计，再对全 finite
source 做 modality bandpass、target-rate resample 和连续 30 秒 epoch reshape。
它不改变 alias/source selection，也不做 normalization、quantization、clipping、
zero-fill、gap repair、QC 或 annotation/feature extraction。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np

from channel_sources import CANONICAL_BY_MODALITY, SourcePlan
from waveform_primitives import bandpass_filter, resample_finite_signal


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
class WaveformPreprocessResult:
    """单 record waveform 主路径的数组和可审计摘要。"""

    signals: dict[str, np.ndarray]
    source_inspections: tuple[RawSourceInspection, ...]
    slot_results: dict[str, SlotProcessResult]
    n_epochs: int


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
    raw source 只要包含 NaN/Inf，就记录 inspection 并停止该 slot 的普通处理，
    不调用 interpolate_nonfinite，也不引入 gap threshold 或自动修复。
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

    for modality, canonical_names in CANONICAL_BY_MODALITY.items():
        target_rate = TARGET_SAMPLING_RATES[modality]
        modality_signal = np.full(
            (n_epochs, len(canonical_names), EPOCH_DURATION_SEC * target_rate),
            np.nan,
            dtype=np.float32,
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
                invalid = invalid or inspection.has_nonfinite

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
                native_values = raw_values[0] - raw_values[1]
            else:
                native_values = raw_values[0]

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
            slot_results[canonical_name] = SlotProcessResult(
                modality, canonical_name, indices, tuple(raw_lengths),
                filtered_length, resampled_length, True, "",
                float(np.isfinite(epoch_values).mean()) if epoch_values.size else 0.0,
                bool(np.isfinite(epoch_values[0]).any()) if n_epochs else False,
                bool(np.isfinite(epoch_values[-1]).any()) if n_epochs else False,
            )
        signals[modality] = modality_signal

    return WaveformPreprocessResult(
        signals=signals,
        source_inspections=tuple(inspections),
        slot_results=slot_results,
        n_epochs=n_epochs,
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
