"""ECG/HRV 分支接口；保留 epoch、window、night 选择，不冻结窗口和步长。"""

from typing import Any, Mapping
from ..feature_interfaces import BranchNotImplementedError

BRANCH_NAME = "ecg"
PLANNED_TIME_SCALES = ("epoch", "window", "night")

def extract(context: Any, config: Mapping[str, Any] | None = None):
    raise BranchNotImplementedError("ECG feature extraction is not implemented")

def quality_gate(context: Any, config: Mapping[str, Any] | None = None):
    raise BranchNotImplementedError("ECG-specific quality gating is not implemented")
