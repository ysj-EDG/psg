"""Demographics10 运行时输入分支接口。

目标是从 manifest 指向的 demographics.csv 按 BidsFolder+SessionID 唯一匹配，未来
生成 record-level `(10,)`；不把 demographic10 强制重复写入 HDF5 feature cache。
"""

from typing import Any, Mapping
from ..feature_interfaces import BranchNotImplementedError

BRANCH_NAME = "demographics"
PLANNED_TIME_SCALES = ("record",)
FEATURE_NAMES = (
    "age", "sex_Female", "sex_Male", "sex_OtherUnknown", "race_Asian",
    "race_Black", "race_Others", "race_Unavailable", "race_White", "BMI",
)

def extract(context: Any, config: Mapping[str, Any] | None = None):
    raise BranchNotImplementedError("Demographics10 feature extraction is not implemented")

def quality_gate(context: Any, config: Mapping[str, Any] | None = None):
    raise BranchNotImplementedError("Demographics quality gating is not implemented")
