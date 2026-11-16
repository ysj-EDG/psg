"""EEG 30 秒 epoch-level 特征分支接口；当前不计算 EEG 特征或伪迹阈值。"""

from typing import Any, Mapping
from ..feature_interfaces import BranchNotImplementedError

BRANCH_NAME = "eeg"
PLANNED_TIME_SCALES = ("epoch",)

def extract(context: Any, config: Mapping[str, Any] | None = None):
    raise BranchNotImplementedError("EEG feature extraction is not implemented")

def quality_gate(context: Any, config: Mapping[str, Any] | None = None):
    raise BranchNotImplementedError("EEG-specific quality gating is not implemented")
