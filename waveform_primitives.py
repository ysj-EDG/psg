"""PSG waveform cache 预处理使用的数值基础算子。

本模块提供 waveform 长度整理、非有限值插值、采样率变换和连续 30 秒 epoch
重塑等通用操作，为各个 PSG modality 的后续 waveform 写入提供统一的数组接口。
这些算子保持独立，便于 cache 预处理流程复用，并将训练与 feature extraction
逻辑置于上层模块。
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt


def pad_or_trim(
    values: np.ndarray,
    target_length: int,
    fill_value: float = np.nan,
) -> np.ndarray:
    """将输入数组整理为 ``target_length`` 个样本，并返回尾部 padding 后的结果。"""

    if target_length < 0:
        raise ValueError(f"target_length must be non-negative: {target_length}")
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size >= target_length:
        return values[:target_length].copy()
    output = np.full(target_length, fill_value, dtype=float)
    output[: values.size] = values
    return output


def interpolate_nonfinite(values: np.ndarray) -> np.ndarray:
    """用线性插值生成有限值数组，作为后续数值运算的输入。"""

    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if finite.all():
        return values.copy()
    if not finite.any():
        return np.zeros_like(values)
    indices = np.arange(values.size)
    output = values.copy()
    output[~finite] = np.interp(indices[~finite], indices[finite], values[finite])
    return output


def resample_signal(
    values: np.ndarray,
    source_rate: float,
    target_rate: int,
    target_length: int,
) -> np.ndarray:
    """把 1-D 数组变换到目标采样率，并整理为指定长度。"""

    rounded_rate = int(round(source_rate))
    if rounded_rate <= 0 or not np.isclose(source_rate, rounded_rate, atol=1e-6):
        raise ValueError(f"source_rate must be positive and integer-like: {source_rate}")
    if target_rate <= 0:
        raise ValueError(f"target_rate must be positive: {target_rate}")
    ratio = Fraction(int(target_rate), rounded_rate)
    resampled = resample_poly(
        interpolate_nonfinite(np.asarray(values, dtype=float).reshape(-1)),
        up=ratio.numerator,
        down=ratio.denominator,
    )
    return pad_or_trim(resampled, target_length)


def resample_finite_signal(
    values: np.ndarray,
    source_rate: float,
    target_rate: int,
    target_length: int,
) -> np.ndarray:
    """只对已确认全 finite 的数组做采样率变换，不执行任何缺失值插值。"""

    values = np.asarray(values, dtype=float).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("resample_finite_signal requires an all-finite input")
    rounded_rate = int(round(source_rate))
    if rounded_rate <= 0 or not np.isclose(source_rate, rounded_rate, atol=1e-6):
        raise ValueError(f"source_rate must be positive and integer-like: {source_rate}")
    if target_rate <= 0:
        raise ValueError(f"target_rate must be positive: {target_rate}")
    ratio = Fraction(int(target_rate), rounded_rate)
    resampled = resample_poly(
        values,
        up=ratio.numerator,
        down=ratio.denominator,
    )
    return pad_or_trim(resampled, target_length)



def reshape_epoch_signal(
    values: np.ndarray,
    *,
    n_epochs: int,
    sampling_rate: int,
    epoch_duration_sec: int = 30,
) -> np.ndarray:
    """把 1-D modality 信号整理并重塑为连续的固定时长 epoch 数组。"""

    if n_epochs < 0 or sampling_rate <= 0 or epoch_duration_sec <= 0:
        raise ValueError("epoch and sampling-rate arguments must be positive")
    samples_per_epoch = epoch_duration_sec * sampling_rate
    return pad_or_trim(values, n_epochs * samples_per_epoch).reshape(
        n_epochs,
        samples_per_epoch,
    )


def bandpass_filter(
    values: np.ndarray,
    sampling_rate: float,
    low_hz: float,
    high_hz: float,
    *,
    order: int = 4,
) -> np.ndarray:
    """对已经确认全为有限值的 physical-amplitude waveform 做 Butterworth 带通。"""

    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("bandpass_filter requires an all-finite input")
    if sampling_rate <= 0 or low_hz <= 0 or high_hz <= low_hz:
        raise ValueError("invalid sampling-rate or bandpass bounds")
    if high_hz >= sampling_rate / 2.0:
        raise ValueError("bandpass upper bound must be below Nyquist")
    if values.size == 0:
        return values.copy()
    sos = butter(
        order,
        (low_hz, high_hz),
        btype="bandpass",
        fs=float(sampling_rate),
        output="sos",
    )
    return sosfiltfilt(sos, values)
