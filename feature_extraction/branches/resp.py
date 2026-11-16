"""RESP 七个 canonical slot 的特征分支接口；保留 source unit/route 和专用 QC hook。"""

from typing import Any, Mapping
from ..feature_interfaces import BranchNotImplementedError

BRANCH_NAME = "resp"
PLANNED_TIME_SCALES = ("epoch", "window", "night")

def extract(context: Any, config: Mapping[str, Any] | None = None):
    raise BranchNotImplementedError("RESP feature extraction is not implemented")

def quality_gate(context: Any, config: Mapping[str, Any] | None = None):
    raise BranchNotImplementedError("RESP-specific quality gating is not implemented")
