"""Chin/leg EMG 特征分支接口；extreme flag 不在框架层自动等价为通道无效。"""

from typing import Any, Mapping
from ..feature_interfaces import BranchNotImplementedError

BRANCH_NAME = "emg"
PLANNED_TIME_SCALES = ("epoch", "window", "night")

def extract(context: Any, config: Mapping[str, Any] | None = None):
    raise BranchNotImplementedError("EMG feature extraction is not implemented")

def quality_gate(context: Any, config: Mapping[str, Any] | None = None):
    raise BranchNotImplementedError("EMG-specific quality gating is not implemented")
