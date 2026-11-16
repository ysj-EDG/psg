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


def _as_reason_array(value: np.ndarray | None, shape: tuple[int, ...]) -> np.ndarray:
    if value is None:
        return np.full(shape, "", dtype="U64")
    result = np.asarray(value, dtype="U64")
    if result.shape != shape:
        raise ValueError(f"invalid_reason shape {result.shape} does not match {shape}")
    return result


@dataclass(frozen=True)
class FeatureBlock:
    """一组同一时间尺度的特征及逐元素有效性。

    `values` 始终为 float32；无效元素必须为 NaN，并在 `invalid_reason` 中留下原因。
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
        if values.dtype != np.dtype("float32"):
            raise TypeError(f"FeatureBlock values must be float32, got {values.dtype}")
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
