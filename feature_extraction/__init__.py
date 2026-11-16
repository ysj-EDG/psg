"""Canonical HDF5 手工特征提取框架。

本包只提供 manifest、只读 HDF5 上下文、结果 schema 和八个后续分支的接口。
本阶段不读取 EDF、不计算生理特征、不写 feature cache，也不包含训练或评分逻辑。
"""

from .feature_schema import FeatureBlock, BranchResult, TimeScale
from .feature_interfaces import BranchNotImplementedError, BranchSpec

__all__ = [
    "FeatureBlock",
    "BranchResult",
    "TimeScale",
    "BranchNotImplementedError",
    "BranchSpec",
]
