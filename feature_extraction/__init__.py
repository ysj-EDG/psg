"""Canonical HDF5 手工特征提取框架。

本包只提供 manifest、只读 HDF5 上下文、结果 schema 和八个后续分支的接口。
本阶段不读取 EDF、不计算生理特征、不写 feature cache，也不包含训练或评分逻辑。
"""

from .feature_schema import FeatureBlock, BranchResult, TimeScale
from .feature_interfaces import BranchNotImplementedError, BranchSpec
from .feature_profiles import B1_PROFILE, D7_PROFILE, FeatureProfile, get_profile, schema_hash
from .feature_store import FeatureLoader, FeatureStore, ModalityFeatures, StoredFeatureRecord, validate_record

__all__ = [
    "FeatureBlock",
    "BranchResult",
    "TimeScale",
    "BranchNotImplementedError",
    "BranchSpec",
    "FeatureProfile",
    "B1_PROFILE",
    "D7_PROFILE",
    "get_profile",
    "schema_hash",
    "FeatureStore",
    "FeatureLoader",
    "ModalityFeatures",
    "StoredFeatureRecord",
    "validate_record",
]
