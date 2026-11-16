"""八个分支共享的接口契约和可选质量检查入口。

分支实现只接收 SampleContext 和配置，具体特征公式在后续独立阶段加入。
本模块不读取 HDF5、不计算信号、不定义伪迹阈值。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol

from .feature_schema import BranchResult


class BranchNotImplementedError(NotImplementedError):
    """分支已注册但尚未实现特征计算。"""


class Extractor(Protocol):
    def __call__(self, context: Any, config: Mapping[str, Any] | None = None) -> BranchResult: ...


class QualityGate(Protocol):
    def __call__(self, context: Any, config: Mapping[str, Any] | None = None) -> Any: ...


@dataclass(frozen=True)
class BranchSpec:
    name: str
    description: str
    module: str
    planned_time_scales: tuple[str, ...]
    extractor: Extractor
    quality_gate: QualityGate | None = None
    implementation_order: int = 0


def call_extractor(spec: BranchSpec, context: Any, config: Mapping[str, Any] | None = None) -> BranchResult:
    """调用分支并验证结果；未实现分支的异常保持明确向上传递。"""

    result = spec.extractor(context, config or {})
    if not isinstance(result, BranchResult):
        raise TypeError(f"branch {spec.name} returned {type(result).__name__}, expected BranchResult")
    return result.validate()
