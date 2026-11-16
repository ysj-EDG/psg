"""EOG channel feature分支接口；当前只保留输入和 QC hook，不计算特征。"""

from typing import Any, Mapping
from ..feature_interfaces import BranchNotImplementedError

BRANCH_NAME = "eog"
PLANNED_TIME_SCALES = ("epoch", "window")

def extract(context: Any, config: Mapping[str, Any] | None = None):
    raise BranchNotImplementedError("EOG feature extraction is not implemented")

def quality_gate(context: Any, config: Mapping[str, Any] | None = None):
    raise BranchNotImplementedError("EOG-specific quality gating is not implemented")
