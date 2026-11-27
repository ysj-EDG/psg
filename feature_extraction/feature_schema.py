"""定义统一的特征结果 schema 和验证规则。

输入是已经存在的 canonical HDF5 数据，输出只在内存中表达 feature block。
本模块不读取信号、不计算生理指标、不执行 QC、不写入特征缓存。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

import numpy as np


class TimeScale(str, Enum):
    """框架支持的四种独立时间尺度。"""

    EPOCH = "epoch"
    WINDOW = "window"
    NIGHT = "night"
    RECORD = "record"


class InvalidReason(str, Enum):
    """特征无效原因；阈值和具体判断留给各分支。"""

    CHANNEL_MISSING = "channel_missing"
    COVERAGE_INSUFFICIENT = "coverage_insufficient"
    CAISR_UNAVAILABLE = "caisr_unavailable"
    QUALITY_FAILED = "quality_failed"
    COMPUTATION_FAILED = "computation_failed"
    MISSING_INPUT = "missing_input"
    NOT_IMPLEMENTED = "not_implemented"
    INVALID_VALUE = "invalid_value"


# 只提供稳定的审计分类，不覆盖分支输出的原始 reason 字符串。
_INVALID_REASON_CATEGORY_RULES = {
    "channel_missing": "channel_missing",
    "caisr_unavailable": "annotation_unavailable",
    "stage_unavailable": "stage_unavailable",
    "unit_unknown": "unit_unknown",
    "coverage_insufficient": "support_insufficient",
    "quality_failed": "quality_failed",
    "spectral_support_insufficient": "spectral_support_insufficient",
    "nn_support_insufficient": "nn_support_insufficient",
    "adjacent_pair_support_insufficient": "nn_support_insufficient",
    "triple_support_insufficient": "nn_support_insufficient",
    "no_continuous_block": "continuous_signal_insufficient",
    "no_b_exact_stage_window": "stage_window_unavailable",
    "no_ecg_exact_stage_window": "stage_window_unavailable",
    "hrv_valid_window_count_lt2": "aggregation_support_insufficient",
    "computation_failed": "computation_failed",
    "math_insufficient": "computation_support_insufficient",
    "invalid_value": "invalid_value",
    "missing_input": "missing_input",
    "not_implemented": "not_implemented",
}


def categorize_invalid_reason(reason: str) -> str:
    """将一个或多个原始 reason 映射为稳定审计类别。

    原始 reason 永远由调用方另行保存；未知值返回 ``unmapped``，不静默归类。
    多个以 ``;`` 连接的失败条件按首次出现顺序去重后返回。
    """

    text = str(reason or "")
    if not text:
        return ""
    categories: list[str] = []
    for token in (part.strip() for part in text.split(";")):
        if not token:
            continue
        category = _INVALID_REASON_CATEGORY_RULES.get(token, "unmapped")
        if category not in categories:
            categories.append(category)
    return ";".join(categories) if categories else "unmapped"


def _as_reason_array(value: np.ndarray | None, shape: tuple[int, ...]) -> np.ndarray:
    """构造不截断的 native-Unicode reason 数组。

    旧实现固定使用 U64，会静默截断 ECG 的复合失败原因；这里按实际最长
    字符串分配宽度，但仍拒绝 object/pickle 语义。
    """

    if value is None:
        return np.full(shape, "", dtype="<U1")
    raw = np.asarray(value)
    if raw.shape != shape:
        raise ValueError(f"invalid_reason shape {raw.shape} does not match {shape}")
    if raw.dtype.kind not in {"U", "S"}:
        raise TypeError("invalid_reason must use a string dtype")
    text = raw.astype(str, copy=False)
    width = max(1, max((len(item) for item in text.reshape(-1).tolist()), default=0))
    return np.asarray(text, dtype=f"<U{width}")


@dataclass(frozen=True)
class FeatureBlock:
    """一组同一时间尺度的特征及逐元素有效性。

    CAISR/EEG 及一般分支 values 为 float32；正式 ECG-S2 特征集明确使用 float64。
    无效元素必须为 NaN，并在 `invalid_reason` 中留下原因。
    因此同一 epoch 中一个 channel 或 feature 无效不会牵连其它元素。
    """

    branch_name: str
    feature_set_name: str
    time_scale: TimeScale | str
    values: np.ndarray
    feature_names: tuple[str, ...]
    valid_mask: np.ndarray
    invalid_reason: np.ndarray | None = None
    epoch_index: np.ndarray | None = None
    window_start_sec: np.ndarray | None = None
    window_end_sec: np.ndarray | None = None
    config: Mapping[str, Any] = field(default_factory=dict)
    version: str = "1.0"

    def __post_init__(self) -> None:
        scale = TimeScale(self.time_scale)
        values = np.asarray(self.values)
        valid = np.asarray(self.valid_mask, dtype=bool)
        names = tuple(str(name) for name in self.feature_names)
        allowed_float64 = (
            self.branch_name == "ecg"
            and self.feature_set_name == "stage_hrv_s2"
            and tuple(names) == ("S01_N2_HF_relative", "S02_N3_HF_absolute_ms2")
            and values.dtype == np.dtype("float64")
        )
        if values.dtype != np.dtype("float32") and not allowed_float64:
            raise TypeError(f"FeatureBlock values must be float32 except ECG-S2 float64, got {values.dtype}")
        if values.ndim not in (1, 2):
            raise ValueError(f"FeatureBlock values must be rank 1 or 2, got {values.ndim}")
        expected_rank = 2 if scale in {TimeScale.EPOCH, TimeScale.WINDOW} else 1
        if values.ndim != expected_rank:
            raise ValueError(f"{scale.value} FeatureBlock must have rank {expected_rank}")
        if values.shape[-1] != len(names):
            raise ValueError("feature_names length must equal the feature dimension")
        if valid.shape != values.shape:
            raise ValueError("valid_mask shape must equal values shape")
        if not np.isfinite(values[valid]).all():
            raise ValueError("valid FeatureBlock values must be finite")
        if np.isinf(values).any():
            raise ValueError("FeatureBlock cannot contain Inf")
        if np.any(valid & np.isnan(values)):
            raise ValueError("valid FeatureBlock values cannot be NaN")
        if np.any((~valid) & ~np.isnan(values)):
            raise ValueError("invalid FeatureBlock values must be NaN")
        reasons = _as_reason_array(self.invalid_reason, values.shape)
        if np.any(valid & (reasons != "")):
            raise ValueError("valid features must have an empty invalid_reason")
        if np.any((~valid) & (reasons == "")):
            raise ValueError("every invalid feature must have an invalid_reason")
        if scale == TimeScale.EPOCH:
            if self.epoch_index is not None and np.asarray(self.epoch_index).shape != (values.shape[0],):
                raise ValueError("epoch_index must have one entry per epoch")
        if scale == TimeScale.WINDOW:
            starts = np.asarray(self.window_start_sec) if self.window_start_sec is not None else None
            ends = np.asarray(self.window_end_sec) if self.window_end_sec is not None else None
            if starts is None or ends is None or starts.shape != (values.shape[0],) or ends.shape != (values.shape[0],):
                raise ValueError("window blocks require start and end arrays")
            if not np.isfinite(starts).all() or not np.isfinite(ends).all() or np.any(ends < starts):
                raise ValueError("window time bounds must be finite and ordered")
        object.__setattr__(self, "time_scale", scale)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "valid_mask", valid)
        object.__setattr__(self, "invalid_reason", reasons)


@dataclass(frozen=True)
class BranchResult:
    """一个分支返回的可验证结果；可以包含多个不同 feature set。"""

    branch_name: str
    blocks: tuple[FeatureBlock, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> "BranchResult":
        if not self.branch_name:
            raise ValueError("branch_name cannot be empty")
        if not isinstance(self.blocks, tuple):
            raise TypeError("blocks must be a tuple")
        for block in self.blocks:
            if block.branch_name != self.branch_name:
                raise ValueError("FeatureBlock branch_name does not match BranchResult")
        return self
