"""Canonical EEG 整夜特征分支。

本模块使用 canonical HDF5 中已经预处理的 128 Hz、µV 波形，按 30 秒 epoch
切成六个互不重叠的 5 秒窗口，进行带物理单位的 PSD 和候选整夜特征聚合。
不读取 EDF，不修改波形，不做整夜归一化或幅值裁剪；QC 只决定窗口是否参与
统计，缺失值保持 NaN 并在 FeatureBlock.invalid_reason 中记录原因。
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

import numpy as np
from scipy.signal import periodogram

from ..feature_schema import BranchResult, FeatureBlock, TimeScale


BRANCH_NAME = "eeg"
FEATURE_VERSION = "eeg_night_candidate_v2"
PLANNED_TIME_SCALES = (TimeScale.NIGHT.value,)

# 仅为工程支持门槛，尚不是临床标准，也不根据标签调节。
MIN_VALID_STAGE_SEC = {"N2": 900.0, "N3": 600.0, "REM": 600.0}
STAGE_NAMES_BY_CODE = {1: "N3", 2: "N2", 4: "REM"}
LOG_EPSILON = 1e-12

EEG_FS = 128.0
EPOCH_SECONDS = 30.0
WINDOW_SECONDS = 5.0
WINDOW_SAMPLES = 640
WINDOWS_PER_EPOCH = 6
NFFT = 640

NREM_BANDS = {
    "SO": (0.5, 1.0),
    "DELTA_1_4": (1.0, 4.0),
    "SWA": (0.5, 4.0),
    "SLOW_SIGMA": (11.0, 13.0),
    "FAST_SIGMA": (13.0, 16.0),
    "SIGMA": (11.0, 16.0),
    "FAST_TOTAL": (8.0, 30.0),
    "TOTAL": (0.5, 30.0),
}
REM_BANDS = {
    "DELTA": (0.5, 4.5),
    "THETA": (4.5, 8.0),
    "ALPHA": (8.0, 12.0),
    "SIGMA": (12.0, 15.0),
    "BETA": (15.0, 32.0),
}
ALL_BANDS = {**NREM_BANDS, **{f"REM_{name}": bounds for name, bounds in REM_BANDS.items()}}

FEATURE_NAMES = (
    "eeg_n3_frontal_so_log_power",
    "eeg_n3_frontal_delta_log_power",
    "eeg_n2_frontal_swa_log_power",
    "eeg_n3_frontal_so_delta_log_ratio",
    "eeg_n3_frontal_slow_fast_log_ratio",
    "eeg_n2_central_sigma_relative_power",
    "eeg_n2_frontal_slow_sigma_relative_power",
    "eeg_n2_central_fast_sigma_relative_power",
    "eeg_n2_occipital_fast_sigma_relative_power",
    "eeg_rem_occipital_slowing_log_ratio",
    "eeg_rem_central_slowing_log_ratio",
    "eeg_rem_frontal_slowing_log_ratio",
    "eeg_rem_occipital_theta_alpha_log_ratio",
    "eeg_n3_frontal_occipital_swa_log_gradient",
    "eeg_n2_frontal_occipital_fast_sigma_log_gradient",
    "eeg_rem_occipital_frontal_slowing_difference",
    "eeg_n3_frontal_swa_left_right_log_asymmetry",
    "eeg_n2_central_sigma_left_right_log_asymmetry",
)
CORE_FEATURE_COUNT = 14
EXPERIMENTAL_FEATURE_COUNT = 4

CANONICAL_CHANNELS = (
    "EEG_F3",
    "EEG_F4",
    "EEG_C3",
    "EEG_C4",
    "EEG_O1",
    "EEG_O2",
)
REGIONS = {
    "frontal": ("EEG_F3", "EEG_F4"),
    "central": ("EEG_C3", "EEG_C4"),
    "occipital": ("EEG_O1", "EEG_O2"),
}
STAGE_CODES = {"N3": 1, "N2": 2, "REM": 4}
TARGET_STAGES = tuple(STAGE_CODES.values())


def compute_window_psd(
    signal: np.ndarray,
    *,
    fs: float = EEG_FS,
    window_seconds: float = WINDOW_SECONDS,
    nfft: int = NFFT,
    window: str = "hamming",
    detrend: str = "linear",
    scaling: str = "density",
) -> tuple[np.ndarray, np.ndarray]:
    """计算一个 5 秒窗口的单边 PSD，返回频率 Hz 和功率/Hz。

    输出 PSD 保持输入物理单位的平方/Hz；canonical EEG 输入为 µV，
    因此积分后的频带功率为 µV²。所有输入样本必须 finite。
    """

    values = np.asarray(signal, dtype=np.float64).reshape(-1)
    expected_samples = int(round(float(fs) * float(window_seconds)))
    if expected_samples <= 0 or values.size != expected_samples:
        raise ValueError(
            f"signal must contain exactly {expected_samples} samples, got {values.size}"
        )
    if not np.isfinite(values).all():
        raise ValueError("signal must be finite before PSD calculation")
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("fs must be positive and finite")
    frequencies, psd = periodogram(
        values,
        fs=float(fs),
        window=window,
        nfft=int(nfft),
        detrend=detrend,
        return_onesided=True,
        scaling=scaling,
    )
    return np.asarray(frequencies, dtype=np.float64), np.asarray(psd, dtype=np.float64)


def integrate_band_power(frequencies: np.ndarray, psd: np.ndarray, low_hz: float, high_hz: float) -> float:
    """在线性功率尺度上积分频带，并插值处理非 FFT 网格边界。"""

    frequencies = np.asarray(frequencies, dtype=np.float64).reshape(-1)
    psd = np.asarray(psd, dtype=np.float64).reshape(-1)
    if frequencies.size != psd.size or frequencies.size < 2:
        raise ValueError("frequencies and psd must have the same length >= 2")
    if not (np.isfinite(low_hz) and np.isfinite(high_hz) and low_hz < high_hz):
        raise ValueError("band boundaries must be finite and ordered")
    if np.any(~np.isfinite(frequencies)) or np.any(~np.isfinite(psd)):
        raise ValueError("frequencies and psd must be finite")
    if np.any(np.diff(frequencies) <= 0):
        raise ValueError("frequencies must be strictly increasing")
    if low_hz < frequencies[0] or high_hz > frequencies[-1]:
        raise ValueError("band is outside the frequency axis")
    interior = (frequencies > low_hz) & (frequencies < high_hz)
    band_frequencies = np.concatenate(
        ([float(low_hz)], frequencies[interior], [float(high_hz)])
    )
    band_psd = np.interp(band_frequencies, frequencies, psd)
    integrate = getattr(np, "trapezoid", np.trapz)
    return float(integrate(band_psd, band_frequencies))


def _integrate_band_power_batch(
    frequencies: np.ndarray,
    psd: np.ndarray,
    low_hz: float,
    high_hz: float,
) -> np.ndarray:
    """向量化积分 ``(..., frequency)`` PSD，边界采用线性插值。"""

    frequencies = np.asarray(frequencies, dtype=np.float64).reshape(-1)
    values = np.asarray(psd, dtype=np.float64)
    if values.shape[-1] != frequencies.size:
        raise ValueError("PSD frequency axis does not match frequencies")
    step = float(frequencies[1] - frequencies[0])
    low_position = (float(low_hz) - float(frequencies[0])) / step
    high_position = (float(high_hz) - float(frequencies[0])) / step
    low_index = int(np.floor(low_position))
    high_index = int(np.floor(high_position))
    low_fraction = low_position - low_index
    high_fraction = high_position - high_index
    low_value = values[..., low_index] * (1.0 - low_fraction) + values[..., low_index + 1] * low_fraction
    high_value = values[..., high_index] * (1.0 - high_fraction) + values[..., high_index + 1] * high_fraction
    interior = (frequencies > low_hz) & (frequencies < high_hz)
    interior_values = values[..., interior]
    band_values = np.concatenate(
        (low_value[..., None], interior_values, high_value[..., None]),
        axis=-1,
    )
    band_frequencies = np.concatenate(
        ([float(low_hz)], frequencies[interior], [float(high_hz)])
    )
    integrate = getattr(np, "trapezoid", np.trapz)
    return np.asarray(integrate(band_values, band_frequencies, axis=-1), dtype=np.float64)


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _is_uv_unit(value: Any) -> bool:
    normalized = _text(value).strip().lower().replace("μ", "µ")
    return normalized in {"uv", "µv", "microvolt", "microvolts"}


def _safe_log(value: float) -> float:
    """绝对功率使用 log10；功率比由 _safe_log_ratio 使用自然对数。"""

    if not np.isfinite(value) or value <= LOG_EPSILON:
        return float("nan")
    return float(np.log10(value))


def _safe_log_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator):
        return float("nan")
    if numerator <= LOG_EPSILON or denominator <= LOG_EPSILON:
        return float("nan")
    return float(np.log(numerator / denominator))


def _safe_relative(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator):
        return float("nan")
    if numerator < 0 or denominator <= 0:
        return float("nan")
    return float(numerator / denominator)


def _empty_block(reason: str, *, config: Mapping[str, Any] | None = None) -> BranchResult:
    values = np.full(len(FEATURE_NAMES), np.nan, dtype=np.float32)
    valid = np.zeros(len(FEATURE_NAMES), dtype=bool)
    reasons = np.full(len(FEATURE_NAMES), reason, dtype="U64")
    block = FeatureBlock(
        branch_name=BRANCH_NAME,
        feature_set_name="eeg_night",
        time_scale=TimeScale.NIGHT,
        values=values,
        feature_names=FEATURE_NAMES,
        valid_mask=valid,
        invalid_reason=reasons,
        config=config or {},
        version=FEATURE_VERSION,
    )
    return BranchResult(
        branch_name=BRANCH_NAME,
        blocks=(block,),
        metadata={"feature_version": FEATURE_VERSION, "error": reason},
    )


def _region_indices(channel_names: tuple[str, ...]) -> dict[str, tuple[int, ...]]:
    lookup = {name: index for index, name in enumerate(channel_names)}
    return {
        region: tuple(lookup[name] for name in names if name in lookup)
        for region, names in REGIONS.items()
    }


def _required_reason(
    stage: int,
    regions: tuple[str, ...],
    *,
    stage_counts: Mapping[int, int],
    region_indices: Mapping[str, tuple[int, ...]],
    present: np.ndarray,
    units_ok: np.ndarray,
    region_counts: Mapping[tuple[int, str], int],
    pair_regions: tuple[str, ...] = (),
    pair_counts: Mapping[tuple[int, str], int] | None = None,
    common_regions: tuple[tuple[str, str], ...] = (),
    common_counts: Mapping[tuple[int, str, str], int] | None = None,
    complete_regions: tuple[str, ...] = (),
    stage_support_seconds: Mapping[int, float] | None = None,
) -> str:
    if stage_counts.get(stage, 0) == 0:
        return "stage_unavailable"
    stage_name = STAGE_NAMES_BY_CODE.get(stage)
    for region in complete_regions:
        indices = region_indices.get(region, ())
        if len(indices) < 2 or not all(present[index] for index in indices):
            return "channel_missing"
        if any(not units_ok[index] for index in indices):
            return "unit_unknown"
    for region in regions:
        indices = region_indices.get(region, ())
        present_indices = [index for index in indices if present[index]]
        if not present_indices:
            return "channel_missing"
        if any(not units_ok[index] for index in present_indices):
            return "unit_unknown"
        if region_counts.get((stage, region), 0) == 0:
            return "coverage_insufficient"
    for region in pair_regions:
        indices = region_indices.get(region, ())
        if len(indices) < 2 or not all(present[index] and units_ok[index] for index in indices):
            return "channel_missing" if len(indices) < 2 or not all(present[index] for index in indices) else "unit_unknown"
        if pair_counts is None or pair_counts.get((stage, region), 0) == 0:
            return "coverage_insufficient"
    for first, second in common_regions:
        if common_counts is None or common_counts.get((stage, first, second), 0) == 0:
            return "coverage_insufficient"
    if stage_name is not None and stage_support_seconds is not None:
        if stage_support_seconds.get(stage, 0.0) < MIN_VALID_STAGE_SEC[stage_name]:
            return "coverage_insufficient"
    return ""


def _assign_feature(
    values: np.ndarray,
    valid: np.ndarray,
    reasons: np.ndarray,
    index: int,
    stage: int,
    regions: tuple[str, ...],
    calculation: Callable[[], float],
    *,
    stage_counts: Mapping[int, int],
    region_indices: Mapping[str, tuple[int, ...]],
    present: np.ndarray,
    units_ok: np.ndarray,
    region_counts: Mapping[tuple[int, str], int],
    pair_regions: tuple[str, ...] = (),
    pair_counts: Mapping[tuple[int, str], int] | None = None,
    common_regions: tuple[tuple[str, str], ...] = (),
    common_counts: Mapping[tuple[int, str, str], int] | None = None,
    complete_regions: tuple[str, ...] = (),
    stage_support_seconds: Mapping[int, float] | None = None,
) -> None:
    if stage_support_seconds is None:
        if common_regions:
            support_seconds = min(
                float((common_counts or {}).get((stage, first, second), 0)) * WINDOW_SECONDS
                for first, second in common_regions
            )
        elif pair_regions:
            support_seconds = min(
                float((pair_counts or {}).get((stage, region), 0)) * WINDOW_SECONDS
                for region in pair_regions
            )
        else:
            support_seconds = min(
                float(region_counts.get((stage, region), 0)) * WINDOW_SECONDS
                for region in regions
            ) if regions else 0.0
        stage_support_seconds = {stage: support_seconds}
    reason = _required_reason(
        stage,
        regions,
        stage_counts=stage_counts,
        region_indices=region_indices,
        present=present,
        units_ok=units_ok,
        region_counts=region_counts,
        pair_regions=pair_regions,
        pair_counts=pair_counts,
        common_regions=common_regions,
        common_counts=common_counts,
        complete_regions=complete_regions,
        stage_support_seconds=stage_support_seconds,
    )
    if reason:
        reasons[index] = reason
        return
    try:
        value = float(calculation())
    except Exception:
        reasons[index] = "computation_failed"
        return
    if not np.isfinite(value):
        reasons[index] = "invalid_value"
        return
    values[index] = np.float32(value)
    valid[index] = True
    reasons[index] = ""


def _region_mean(
    sums: Mapping[tuple[int, str, str], float],
    counts: Mapping[tuple[int, str, str], int],
    stage: int,
    region: str,
    band: str,
) -> float:
    count = counts.get((stage, region, band), 0)
    return float(sums[(stage, region, band)] / count) if count else float("nan")


def _pair_mean(
    sums: Mapping[tuple[int, str, str, str], float],
    counts: Mapping[tuple[int, str, str], int],
    stage: int,
    region: str,
    side: str,
    band: str,
) -> float:
    count = counts.get((stage, region), 0)
    return float(sums[(stage, region, side, band)] / count) if count else float("nan")


def _common_region_mean(
    sums: Mapping[tuple[int, str, str, str, str], float],
    counts: Mapping[tuple[int, str, str, str, str], int],
    stage: int,
    first: str,
    second: str,
    region: str,
    band: str,
) -> float:
    key = (stage, first, second, region, band)
    count = counts.get(key, 0)
    return float(sums[key] / count) if count else float("nan")


def _common_rem_slowing(
    sums: Mapping[tuple[int, str, str, str, str], float],
    counts: Mapping[tuple[int, str, str, str, str], int],
    stage: int,
    region: str,
) -> float:
    slow = _common_region_mean(sums, counts, stage, "frontal", "occipital", region, "REM_DELTA") + _common_region_mean(sums, counts, stage, "frontal", "occipital", region, "REM_THETA")
    fast = (
        _common_region_mean(sums, counts, stage, "frontal", "occipital", region, "REM_ALPHA")
        + _common_region_mean(sums, counts, stage, "frontal", "occipital", region, "REM_SIGMA")
        + _common_region_mean(sums, counts, stage, "frontal", "occipital", region, "REM_BETA")
    )
    return _safe_log_ratio(slow, fast)


def _gradient_mean(
    sums: Mapping[tuple[int, str, str, str, str, str], float],
    counts: Mapping[tuple[int, str, str, str, str, str], int],
    stage: int,
    first: str,
    second: str,
    side: str,
    region: str,
    band: str,
) -> float:
    key = (stage, first, second, side, region, band)
    count = counts.get(key, 0)
    return float(sums[key] / count) if count else float("nan")


def _same_side_gradient(
    sums: Mapping[tuple[int, str, str, str, str, str], float],
    counts: Mapping[tuple[int, str, str, str, str, str], int],
    stage: int,
    band: str,
) -> float:
    left = _safe_log_ratio(
        _gradient_mean(sums, counts, stage, "frontal", "occipital", "left", "frontal", band),
        _gradient_mean(sums, counts, stage, "frontal", "occipital", "left", "occipital", band),
    )
    right = _safe_log_ratio(
        _gradient_mean(sums, counts, stage, "frontal", "occipital", "right", "frontal", band),
        _gradient_mean(sums, counts, stage, "frontal", "occipital", "right", "occipital", band),
    )
    if not np.isfinite(left) or not np.isfinite(right):
        return float("nan")
    return float((left + right) / 2.0)


def _gradient_rem_slowing(
    sums: Mapping[tuple[int, str, str, str, str, str], float],
    counts: Mapping[tuple[int, str, str, str, str, str], int],
    stage: int,
    region: str,
    side: str,
) -> float:
    slow = _gradient_mean(sums, counts, stage, "frontal", "occipital", side, region, "REM_DELTA") + _gradient_mean(sums, counts, stage, "frontal", "occipital", side, region, "REM_THETA")
    fast = (
        _gradient_mean(sums, counts, stage, "frontal", "occipital", side, region, "REM_ALPHA")
        + _gradient_mean(sums, counts, stage, "frontal", "occipital", side, region, "REM_SIGMA")
        + _gradient_mean(sums, counts, stage, "frontal", "occipital", side, region, "REM_BETA")
    )
    return _safe_log_ratio(slow, fast)


def _gradient_rem_difference(
    sums: Mapping[tuple[int, str, str, str, str, str], float],
    counts: Mapping[tuple[int, str, str, str, str, str], int],
    stage: int,
) -> float:
    left = _gradient_rem_slowing(sums, counts, stage, "occipital", "left") - _gradient_rem_slowing(sums, counts, stage, "frontal", "left")
    right = _gradient_rem_slowing(sums, counts, stage, "occipital", "right") - _gradient_rem_slowing(sums, counts, stage, "frontal", "right")
    if not np.isfinite(left) or not np.isfinite(right):
        return float("nan")
    return float((left + right) / 2.0)


def _feature_config() -> dict[str, Any]:
    return {
        "feature_version": FEATURE_VERSION,
        "sampling_rate_hz": EEG_FS,
        "window_seconds": WINDOW_SECONDS,
        "nfft": NFFT,
        "window": "hamming",
        "detrend": "linear",
        "scaling": "density",
        "frequency_resolution_hz": EEG_FS / NFFT,
        "nrem_bands_hz": dict(NREM_BANDS),
        "rem_bands_hz": dict(REM_BANDS),
        "physical_unit": "uV^2 for absolute power; dimensionless for ratios",
        "absolute_power_log": "log10",
        "power_ratio_log": "natural_log",
        "log_epsilon": LOG_EPSILON,
        "min_valid_stage_seconds": dict(MIN_VALID_STAGE_SEC),
        "spatial_gradient": "mean of same-side ln(frontal/occipital) ratios; F3/O1 and F4/O2 require joint QC support",
        "qc_rule": "present AND invalid_5s=False AND flat_5s=False AND extreme_5s=False AND finite",
    }


def _read_qc_summary(context: Any) -> dict[str, Any]:
    present = np.asarray(context.channel_present("eeg"), dtype=bool)
    summary: dict[str, Any] = {"channel_present": present.tolist()}
    for quality_name in ("channel_invalid_5s", "channel_flat_5s", "channel_extreme_5s"):
        dataset = context.quality_dataset(quality_name, "eeg")
        values = np.asarray(dataset[:], dtype=bool)
        summary[quality_name] = {
            "shape": tuple(int(value) for value in values.shape),
            "true_count": int(values.sum()),
            "true_fraction": float(values.mean()) if values.size else 0.0,
        }
    return summary


def quality_gate(context: Any, config: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
    """返回 EEG QC 摘要；不引入新的阈值，也不修改 waveform。"""

    return _read_qc_summary(context)


def extract(context: Any, config: Mapping[str, Any] | None = None) -> BranchResult:
    """从一个 SampleContext 提取 18 维候选整夜 EEG 特征。"""

    options = dict(config or {})
    batch_epochs = int(options.get("batch_epochs", 8))
    if batch_epochs <= 0:
        raise ValueError("batch_epochs must be positive")
    feature_config = _feature_config()
    feature_config["batch_epochs"] = batch_epochs

    try:
        signal_dataset = context.dataset("signals/eeg")
        channel_names = context.channel_names("eeg")
        present = np.asarray(context.channel_present("eeg"), dtype=bool)
        dimensions = tuple(_text(value) for value in context.metadata("eeg", "physical_dimension"))
        derivation = tuple(_text(value) for value in context.metadata("eeg", "derivation_method"))
        reference = tuple(_text(value) for value in context.metadata("eeg", "reference_system"))
        selected = tuple(_text(value) for value in context.metadata("eeg", "selected_source_labels"))
        stage_dataset = context.caisr_dataset("stage")
        epoch_index_dataset = context.dataset("timebase/epoch_index")
        epoch_start_dataset = context.dataset("timebase/epoch_start_sec")
        epoch_end_dataset = context.dataset("timebase/epoch_end_sec")
        invalid_dataset = context.quality_dataset("channel_invalid_5s", "eeg")
        flat_dataset = context.quality_dataset("channel_flat_5s", "eeg")
        extreme_dataset = context.quality_dataset("channel_extreme_5s", "eeg")
        valid_fraction_dataset = context.quality_dataset("channel_valid_fraction_30s", "eeg")
    except Exception as exc:
        return _empty_block("missing_input", config=feature_config)

    epoch_count = int(context.epoch_count)
    expected_shape = (epoch_count, len(CANONICAL_CHANNELS), int(WINDOWS_PER_EPOCH * WINDOW_SAMPLES))
    if signal_dataset.shape != expected_shape:
        return _empty_block("shape_mismatch", config=feature_config)
    if len(channel_names) != len(CANONICAL_CHANNELS) or len(present) != len(CANONICAL_CHANNELS):
        return _empty_block("shape_mismatch", config=feature_config)
    if stage_dataset.shape != (epoch_count,):
        return _empty_block("shape_mismatch", config=feature_config)
    if epoch_index_dataset.shape != (epoch_count,) or epoch_start_dataset.shape != (epoch_count,) or epoch_end_dataset.shape != (epoch_count,):
        return _empty_block("shape_mismatch", config=feature_config)
    # 时间轴只做一致性检查；所有聚合仍以 PSG epoch index 为准，不读取 EDF 时间戳。
    epoch_index = np.asarray(epoch_index_dataset[:], dtype=np.int64)
    epoch_start = np.asarray(epoch_start_dataset[:], dtype=np.float64)
    epoch_end = np.asarray(epoch_end_dataset[:], dtype=np.float64)
    if (
        not np.array_equal(epoch_index, np.arange(epoch_count, dtype=np.int64))
        or not np.isfinite(epoch_start).all()
        or not np.isfinite(epoch_end).all()
        or np.any(epoch_end < epoch_start)
        or np.any(np.diff(epoch_start) < 0)
        or np.any(np.diff(epoch_end) < 0)
    ):
        return _empty_block("timebase_mismatch", config=feature_config)
    expected_qc_shape = (epoch_count, len(CANONICAL_CHANNELS), WINDOWS_PER_EPOCH)
    if any(dataset.shape != expected_qc_shape for dataset in (invalid_dataset, flat_dataset, extreme_dataset)):
        return _empty_block("shape_mismatch", config=feature_config)
    if valid_fraction_dataset.shape != (epoch_count, len(CANONICAL_CHANNELS)):
        return _empty_block("shape_mismatch", config=feature_config)

    region_indices = _region_indices(channel_names)
    units_ok = np.asarray([_is_uv_unit(value) for value in dimensions], dtype=bool)
    stage_counts: dict[int, int] = {code: 0 for code in (1, 2, 3, 4, 5, -1)}
    stage_valid_windows: dict[int, int] = {stage: 0 for stage in TARGET_STAGES}
    channel_valid_windows: dict[tuple[int, int], int] = {}
    region_sums: dict[tuple[int, str, str], float] = {}
    region_counts: dict[tuple[int, str, str], int] = {}
    region_window_counts: dict[tuple[int, str], int] = {}
    pair_sums: dict[tuple[int, str, str, str], float] = {}
    pair_counts: dict[tuple[int, str], int] = {}
    common_region_sums: dict[tuple[int, str, str, str, str], float] = {}
    common_region_counts: dict[tuple[int, str, str, str, str], int] = {}
    common_window_counts: dict[tuple[int, str, str], int] = {}
    gradient_sums: dict[tuple[int, str, str, str, str, str], float] = {}
    gradient_counts: dict[tuple[int, str, str, str, str, str], int] = {}
    gradient_window_counts: dict[tuple[int, str, str], int] = {}
    qc_true_counts = {name: 0 for name in ("invalid", "flat", "extreme")}
    finite_window_count = 0

    for batch_start, signal_batch in context.iter_signal_epochs("eeg", batch_epochs=batch_epochs):
        signal_batch = np.asarray(signal_batch, dtype=np.float64)
        batch_size = signal_batch.shape[0]
        batch_end = batch_start + batch_size
        stages = np.asarray(stage_dataset[batch_start:batch_end], dtype=np.int16)
        invalid = np.asarray(invalid_dataset[batch_start:batch_end], dtype=bool)
        flat = np.asarray(flat_dataset[batch_start:batch_end], dtype=bool)
        extreme = np.asarray(extreme_dataset[batch_start:batch_end], dtype=bool)
        _ = np.asarray(valid_fraction_dataset[batch_start:batch_end], dtype=np.float32)
        qc_true_counts["invalid"] += int(invalid.sum())
        qc_true_counts["flat"] += int(flat.sum())
        qc_true_counts["extreme"] += int(extreme.sum())

        windows = signal_batch.reshape(batch_size, len(CANONICAL_CHANNELS), WINDOWS_PER_EPOCH, WINDOW_SAMPLES)
        finite_windows = np.isfinite(windows).all(axis=-1)
        safe_windows = np.where(np.isfinite(windows), windows, 0.0)
        frequencies, psd_batch = periodogram(
            safe_windows,
            fs=EEG_FS,
            window="hamming",
            nfft=NFFT,
            detrend="linear",
            axis=-1,
            return_onesided=True,
            scaling="density",
        )
        band_power_batch = {
            band_name: _integrate_band_power_batch(frequencies, psd_batch, low_hz, high_hz)
            for band_name, (low_hz, high_hz) in ALL_BANDS.items()
        }
        for stage_value, count in zip(*np.unique(stages, return_counts=True)):
            stage_counts[int(stage_value)] = stage_counts.get(int(stage_value), 0) + int(count)
        eligible = present[None, :, None] & units_ok[None, :, None] & (~invalid) & (~flat) & (~extreme) & finite_windows
        for stage in TARGET_STAGES:
            stage_mask = eligible & (stages[:, None, None] == stage)
            stage_valid_windows[stage] += int(np.any(stage_mask, axis=1).sum())
            finite_window_count += int(stage_mask.sum())
            for channel in range(len(CANONICAL_CHANNELS)):
                count = int(stage_mask[:, channel, :].sum())
                if count:
                    channel_valid_windows[(stage, channel)] = channel_valid_windows.get((stage, channel), 0) + count

            for region, indices in region_indices.items():
                if not indices:
                    continue
                region_mask = stage_mask[:, indices, :]
                region_any = np.any(region_mask, axis=1)
                window_count = int(region_any.sum())
                if not window_count:
                    continue
                region_window_counts[(stage, region)] = region_window_counts.get((stage, region), 0) + window_count
                channel_count = region_mask.sum(axis=1)
                for band_name, power_batch in band_power_batch.items():
                    region_values = np.divide(
                        np.where(region_mask, power_batch[:, indices, :], 0.0).sum(axis=1),
                        channel_count,
                        out=np.zeros_like(channel_count, dtype=np.float64),
                        where=channel_count > 0,
                    )
                    key = (stage, region, band_name)
                    region_sums[key] = region_sums.get(key, 0.0) + float(region_values[region_any].sum())
                    region_counts[key] = region_counts.get(key, 0) + window_count

                if len(indices) == 2:
                    pair_mask = np.all(region_mask, axis=1)
                    paired_count = int(pair_mask.sum())
                    if paired_count:
                        pair_counts[(stage, region)] = pair_counts.get((stage, region), 0) + paired_count
                        for side, channel in zip(("left", "right"), indices):
                            for band_name, power_batch in band_power_batch.items():
                                key = (stage, region, side, band_name)
                                pair_sums[key] = pair_sums.get(key, 0.0) + float(power_batch[:, channel, :][pair_mask].sum())

            # 区域差值（实验特征15）保留原有的共同窗口语义：两个区域各有
            # 至少一个可用导联即可；它不冒充下面的完整双侧梯度定义。
            first, second = "frontal", "occipital"
            first_indices = region_indices.get(first, ())
            second_indices = region_indices.get(second, ())
            if first_indices and second_indices:
                first_mask = stage_mask[:, first_indices, :]
                second_mask = stage_mask[:, second_indices, :]
                common_mask = np.any(first_mask, axis=1) & np.any(second_mask, axis=1)
                common_count = int(common_mask.sum())
                if common_count:
                    common_window_counts[(stage, first, second)] = common_window_counts.get((stage, first, second), 0) + common_count
                    for region, indices, region_mask in ((first, first_indices, first_mask), (second, second_indices, second_mask)):
                        channel_count = region_mask.sum(axis=1)
                        for band_name, power_batch in band_power_batch.items():
                            region_values = np.divide(
                                np.where(region_mask, power_batch[:, indices, :], 0.0).sum(axis=1),
                                channel_count,
                                out=np.zeros_like(channel_count, dtype=np.float64),
                                where=channel_count > 0,
                            )
                            key = (stage, first, second, region, band_name)
                            common_region_sums[key] = common_region_sums.get(key, 0.0) + float(region_values[common_mask].sum())
                            common_region_counts[key] = common_region_counts.get(key, 0) + common_count

            # 空间梯度使用同侧组合：F3/O1 和 F4/O2，四个导联必须在
            # 同一 stage、同一 5 秒窗口共同通过 QC；不使用单侧 fallback。
            if len(first_indices) == 2 and len(second_indices) == 2:
                side_indices = {
                    "left": (first_indices[0], second_indices[0]),
                    "right": (first_indices[1], second_indices[1]),
                }
                side_masks = {
                    side: np.all(stage_mask[:, indices, :], axis=1)
                    for side, indices in side_indices.items()
                }
                full_mask = side_masks["left"] & side_masks["right"]
                full_count = int(full_mask.sum())
                if full_count:
                    gradient_window_counts[(stage, first, second)] = gradient_window_counts.get((stage, first, second), 0) + full_count
                    for side, (frontal_index, occipital_index) in side_indices.items():
                        for region, channel in ((first, frontal_index), (second, occipital_index)):
                            for band_name, power_batch in band_power_batch.items():
                                key = (stage, first, second, side, region, band_name)
                                gradient_sums[key] = gradient_sums.get(key, 0.0) + float(power_batch[:, channel, :][full_mask].sum())
                                gradient_counts[key] = gradient_counts.get(key, 0) + full_count


    values = np.full(len(FEATURE_NAMES), np.nan, dtype=np.float32)
    valid = np.zeros(len(FEATURE_NAMES), dtype=bool)
    reasons = np.full(len(FEATURE_NAMES), "coverage_insufficient", dtype="U64")
    region_count_view = {
        key: int(value) for key, value in region_window_counts.items()
    }
    config = feature_config
    # Values are computed from stage/region linear-power means, then transformed.
    _assign_feature(values, valid, reasons, 0, 1, ("frontal",), lambda: _safe_log(_region_mean(region_sums, region_counts, 1, "frontal", "SO")), stage_counts=stage_counts, region_indices=region_indices, present=present, units_ok=units_ok, region_counts=region_count_view)
    _assign_feature(values, valid, reasons, 1, 1, ("frontal",), lambda: _safe_log(_region_mean(region_sums, region_counts, 1, "frontal", "DELTA_1_4")), stage_counts=stage_counts, region_indices=region_indices, present=present, units_ok=units_ok, region_counts=region_count_view)
    _assign_feature(values, valid, reasons, 2, 2, ("frontal",), lambda: _safe_log(_region_mean(region_sums, region_counts, 2, "frontal", "SWA")), stage_counts=stage_counts, region_indices=region_indices, present=present, units_ok=units_ok, region_counts=region_count_view)
    _assign_feature(values, valid, reasons, 3, 1, ("frontal",), lambda: _safe_log_ratio(_region_mean(region_sums, region_counts, 1, "frontal", "SO"), _region_mean(region_sums, region_counts, 1, "frontal", "DELTA_1_4")), stage_counts=stage_counts, region_indices=region_indices, present=present, units_ok=units_ok, region_counts=region_count_view)
    _assign_feature(values, valid, reasons, 4, 1, ("frontal",), lambda: _safe_log_ratio(_region_mean(region_sums, region_counts, 1, "frontal", "SWA"), _region_mean(region_sums, region_counts, 1, "frontal", "FAST_TOTAL")), stage_counts=stage_counts, region_indices=region_indices, present=present, units_ok=units_ok, region_counts=region_count_view)
    _assign_feature(values, valid, reasons, 5, 2, ("central",), lambda: _safe_relative(_region_mean(region_sums, region_counts, 2, "central", "SIGMA"), _region_mean(region_sums, region_counts, 2, "central", "TOTAL")), stage_counts=stage_counts, region_indices=region_indices, present=present, units_ok=units_ok, region_counts=region_count_view)
    _assign_feature(values, valid, reasons, 6, 2, ("frontal",), lambda: _safe_relative(_region_mean(region_sums, region_counts, 2, "frontal", "SLOW_SIGMA"), _region_mean(region_sums, region_counts, 2, "frontal", "TOTAL")), stage_counts=stage_counts, region_indices=region_indices, present=present, units_ok=units_ok, region_counts=region_count_view)
    _assign_feature(values, valid, reasons, 7, 2, ("central",), lambda: _safe_relative(_region_mean(region_sums, region_counts, 2, "central", "FAST_SIGMA"), _region_mean(region_sums, region_counts, 2, "central", "TOTAL")), stage_counts=stage_counts, region_indices=region_indices, present=present, units_ok=units_ok, region_counts=region_count_view)
    _assign_feature(values, valid, reasons, 8, 2, ("occipital",), lambda: _safe_relative(_region_mean(region_sums, region_counts, 2, "occipital", "FAST_SIGMA"), _region_mean(region_sums, region_counts, 2, "occipital", "TOTAL")), stage_counts=stage_counts, region_indices=region_indices, present=present, units_ok=units_ok, region_counts=region_count_view)

    def rem_slowing(region: str) -> float:
        slow = _region_mean(region_sums, region_counts, 4, region, "REM_DELTA") + _region_mean(region_sums, region_counts, 4, region, "REM_THETA")
        fast = (
            _region_mean(region_sums, region_counts, 4, region, "REM_ALPHA")
            + _region_mean(region_sums, region_counts, 4, region, "REM_SIGMA")
            + _region_mean(region_sums, region_counts, 4, region, "REM_BETA")
        )
        return _safe_log_ratio(slow, fast)

    _assign_feature(values, valid, reasons, 9, 4, ("occipital",), lambda: rem_slowing("occipital"), stage_counts=stage_counts, region_indices=region_indices, present=present, units_ok=units_ok, region_counts=region_count_view)
    _assign_feature(values, valid, reasons, 10, 4, ("central",), lambda: rem_slowing("central"), stage_counts=stage_counts, region_indices=region_indices, present=present, units_ok=units_ok, region_counts=region_count_view)
    _assign_feature(values, valid, reasons, 11, 4, ("frontal",), lambda: rem_slowing("frontal"), stage_counts=stage_counts, region_indices=region_indices, present=present, units_ok=units_ok, region_counts=region_count_view)
    _assign_feature(values, valid, reasons, 12, 4, ("occipital",), lambda: _safe_log_ratio(_region_mean(region_sums, region_counts, 4, "occipital", "REM_THETA"), _region_mean(region_sums, region_counts, 4, "occipital", "REM_ALPHA")), stage_counts=stage_counts, region_indices=region_indices, present=present, units_ok=units_ok, region_counts=region_count_view)
    _assign_feature(values, valid, reasons, 13, 1, ("frontal", "occipital"), lambda: _same_side_gradient(gradient_sums, gradient_counts, 1, "SWA"), stage_counts=stage_counts, region_indices=region_indices, present=present, units_ok=units_ok, region_counts=region_count_view, common_regions=(("frontal", "occipital"),), common_counts=gradient_window_counts, complete_regions=("frontal", "occipital"))
    _assign_feature(values, valid, reasons, 14, 2, ("frontal", "occipital"), lambda: _same_side_gradient(gradient_sums, gradient_counts, 2, "FAST_SIGMA"), stage_counts=stage_counts, region_indices=region_indices, present=present, units_ok=units_ok, region_counts=region_count_view, common_regions=(("frontal", "occipital"),), common_counts=gradient_window_counts, complete_regions=("frontal", "occipital"))
    _assign_feature(values, valid, reasons, 15, 4, ("occipital", "frontal"), lambda: _common_rem_slowing(common_region_sums, common_region_counts, 4, "occipital") - _common_rem_slowing(common_region_sums, common_region_counts, 4, "frontal"), stage_counts=stage_counts, region_indices=region_indices, present=present, units_ok=units_ok, region_counts=region_count_view, common_regions=(("frontal", "occipital"),), common_counts=common_window_counts)
    _assign_feature(values, valid, reasons, 16, 1, ("frontal",), lambda: _safe_log_ratio(_pair_mean(pair_sums, pair_counts, 1, "frontal", "left", "SWA"), _pair_mean(pair_sums, pair_counts, 1, "frontal", "right", "SWA")), stage_counts=stage_counts, region_indices=region_indices, present=present, units_ok=units_ok, region_counts=region_count_view, pair_regions=("frontal",), pair_counts=pair_counts)
    _assign_feature(values, valid, reasons, 17, 2, ("central",), lambda: _safe_log_ratio(_pair_mean(pair_sums, pair_counts, 2, "central", "left", "SIGMA"), _pair_mean(pair_sums, pair_counts, 2, "central", "right", "SIGMA")), stage_counts=stage_counts, region_indices=region_indices, present=present, units_ok=units_ok, region_counts=region_count_view, pair_regions=("central",), pair_counts=pair_counts)

    feature_stage = {
        0: 1, 1: 1, 2: 2, 3: 1, 4: 1, 5: 2, 6: 2, 7: 2, 8: 2,
        9: 4, 10: 4, 11: 4, 12: 4, 13: 1, 14: 2, 15: 4, 16: 1, 17: 2,
    }
    feature_region = {
        0: "frontal", 1: "frontal", 2: "frontal", 3: "frontal", 4: "frontal",
        5: "central", 6: "frontal", 7: "central", 8: "occipital",
        9: "occipital", 10: "central", 11: "frontal", 12: "occipital",
    }
    feature_support_seconds = {}
    for index, stage in feature_stage.items():
        if index in (13, 14):
            count = gradient_window_counts.get((stage, "frontal", "occipital"), 0)
        elif index == 15:
            count = common_window_counts.get((stage, "frontal", "occipital"), 0)
        elif index == 16:
            count = pair_counts.get((stage, "frontal"), 0)
        elif index == 17:
            count = pair_counts.get((stage, "central"), 0)
        else:
            count = region_window_counts.get((stage, feature_region[index]), 0)
        feature_support_seconds[FEATURE_NAMES[index]] = float(count * WINDOW_SECONDS)
    metadata = {
        "feature_version": FEATURE_VERSION,
        "feature_names": FEATURE_NAMES,
        "min_valid_stage_seconds": dict(MIN_VALID_STAGE_SEC),
        "stage_counts": {str(key): int(value) for key, value in stage_counts.items()},
        "stage_raw_seconds": {STAGE_NAMES_BY_CODE.get(key, str(key)): float(value * EPOCH_SECONDS) for key, value in stage_counts.items()},
        "stage_valid_window_counts": {str(key): int(value) for key, value in stage_valid_windows.items()},
        "stage_qc_valid_seconds": {STAGE_NAMES_BY_CODE[key]: float(value * WINDOW_SECONDS) for key, value in stage_valid_windows.items()},
        "feature_support_seconds": feature_support_seconds,
        "stage_valid_seconds": {str(key): float(value * WINDOW_SECONDS) for key, value in stage_valid_windows.items()},
        "channel_valid_window_counts": {
            f"stage_{stage}/{channel_names[channel]}": int(count)
            for (stage, channel), count in channel_valid_windows.items()
        },
        "channel_valid_seconds": {
            f"stage_{stage}/{channel_names[channel]}": float(count * WINDOW_SECONDS)
            for (stage, channel), count in channel_valid_windows.items()
        },
        "qc_true_counts": qc_true_counts,
        "finite_window_channel_count": int(finite_window_count),
        "channel_present": present.tolist(),
        "physical_dimension": dimensions,
        "derivation_method": derivation,
        "reference_system": reference,
        "selected_source_labels": selected,
        "unit_known": units_ok.tolist(),
        "region_common_valid_window_counts": {f"stage_{stage}/{region}": int(count) for (stage, region), count in region_window_counts.items()},
        "cross_region_common_valid_window_counts": {f"stage_{stage}/{first}/{second}": int(count) for (stage, first, second), count in common_window_counts.items()},
        "cross_region_bilateral_valid_window_counts": {f"stage_{stage}/{first}/{second}": int(count) for (stage, first, second), count in gradient_window_counts.items()},
        "candidate_feature_groups": {"core": CORE_FEATURE_COUNT, "experimental": EXPERIMENTAL_FEATURE_COUNT},
    }
    block = FeatureBlock(
        branch_name=BRANCH_NAME,
        feature_set_name="eeg_night",
        time_scale=TimeScale.NIGHT,
        values=values,
        feature_names=FEATURE_NAMES,
        valid_mask=valid,
        invalid_reason=reasons,
        config=config,
        version=FEATURE_VERSION,
    )
    return BranchResult(branch_name=BRANCH_NAME, blocks=(block,), metadata=metadata)
