"""SpO2 分支接口；未来区分原始有效秒和插值秒，不计算当前阶段特征。"""

from typing import Any, Mapping
from ..feature_interfaces import BranchNotImplementedError

BRANCH_NAME = "spo2"
PLANNED_TIME_SCALES = ("epoch", "window", "night")

def extract(context: Any, config: Mapping[str, Any] | None = None):
    raise BranchNotImplementedError("SpO2 feature extraction is not implemented")

def quality_gate(context: Any, config: Mapping[str, Any] | None = None):
    raise BranchNotImplementedError("SpO2-specific quality gating is not implemented")
