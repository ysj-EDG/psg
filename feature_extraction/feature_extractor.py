"""Canonical HDF5 手工特征提取主入口和八分支注册表。

主入口只负责列出分支、读取 manifest、创建短生命周期 SampleContext、调度分支和
验证 BranchResult。它不计算生理特征、不滤波、不归一化、不做伪迹阈值判断、不写
feature cache，也不会从 manifest 的 EDF provenance 列读取 EDF。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

import h5py

from .feature_interfaces import BranchNotImplementedError, BranchSpec, call_extractor
from .sample_loader import Manifest, SampleContext
from .branches import caisr, demographics, ecg, eeg, emg, eog, resp, spo2


_MODULES = (caisr, eeg, ecg, resp, emg, spo2, demographics, eog)
BRANCH_REGISTRY = {
    module.BRANCH_NAME: BranchSpec(
        name=module.BRANCH_NAME,
        description=module.__doc__.splitlines()[0],
        module=module.__name__,
        planned_time_scales=tuple(module.PLANNED_TIME_SCALES),
        extractor=module.extract,
        quality_gate=module.quality_gate,
        implementation_order=index + 1,
    )
    for index, module in enumerate(_MODULES)
}


def list_branches() -> tuple[BranchSpec, ...]:
    return tuple(BRANCH_REGISTRY[name] for name in BRANCH_REGISTRY)


def describe_branch(name: str) -> str:
    try:
        spec = BRANCH_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"unknown branch: {name}") from exc
    return (
        f"name: {spec.name}\nmodule: {spec.module}\n"
        f"planned_time_scales: {', '.join(spec.planned_time_scales)}\n"
        f"description: {spec.description}\nextractor: registered"
    )


def inspect_h5(path: Path | str) -> str:
    """只读打印 HDF5 keys/attrs；该入口不创建 SampleContext 或执行分支。"""

    output: list[str] = []
    with h5py.File(path, "r") as handle:
        output.append(f"path: {Path(path).resolve()}")
        output.append(f"root_keys: {sorted(handle.keys())}")
        output.append(f"root_attrs: {dict(handle.attrs.items())}")
        output.append(f"datasets: {sum(isinstance(item, h5py.Dataset) for item in handle.values())}")
    return "\n".join(output)


def run_one(manifest_path: Path | str, branch: str, record_id: str, config: Mapping[str, Any] | None = None):
    spec = BRANCH_REGISTRY.get(branch)
    if spec is None:
        raise ValueError(f"unknown branch: {branch}")
    manifest = Manifest.from_tsv(manifest_path)
    with SampleContext(manifest.get(record_id)) as context:
        return call_extractor(spec, context, config)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-branches", action="store_true")
    parser.add_argument("--describe", choices=tuple(BRANCH_REGISTRY))
    parser.add_argument("--inspect-h5", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--branch", choices=tuple(BRANCH_REGISTRY))
    parser.add_argument("--record-id")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.list_branches:
        for spec in list_branches():
            print(f"{spec.name}\t{spec.description}\ttime_scales={','.join(spec.planned_time_scales)}")
        return 0
    if args.describe:
        print(describe_branch(args.describe))
        return 0
    if args.inspect_h5:
        print(inspect_h5(args.inspect_h5))
        return 0
    if not args.manifest or not args.branch or not args.record_id:
        _parser().error("extraction requires --manifest, --branch and --record-id")
    try:
        run_one(args.manifest, args.branch, args.record_id)
    except BranchNotImplementedError as exc:
        print(f"BRANCH_NOT_IMPLEMENTED: {exc}")
        return 3
    except Exception as exc:
        print(f"EXTRACTION_ERROR: {type(exc).__name__}: {exc}")
        return 2
    raise AssertionError("unimplemented branches must not return successfully")


if __name__ == "__main__":
    raise SystemExit(main())
