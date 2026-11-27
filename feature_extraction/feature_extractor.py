"""Canonical HDF5 手工特征提取主入口和八分支注册表。

主入口只负责列出分支、读取 manifest、创建短生命周期 SampleContext、调度分支和
验证 BranchResult。它不计算生理特征、不滤波、不归一化、不做伪迹阈值判断、不写
feature cache，也不会从 manifest 的 EDF provenance 列读取 EDF。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np

from .feature_interfaces import BranchNotImplementedError, BranchSpec, call_extractor
from .feature_profiles import CAISR20_FEATURE_NAMES, ECG_FEATURE_NAMES, EEG12_FEATURE_NAMES, SCHEMA_REVISION, schema_hash
from .feature_schema import BranchResult, FeatureBlock
from .feature_store import ModalityFeatures, StoredFeatureRecord, write_one, validate_record
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



def _json_safe(value: Any) -> Any:
    """将分支诊断压缩成严格 JSON 值；不把 NaN/Inf 写入 metadata。"""

    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def _branch_block(result: BranchResult, modality: str, names: tuple[str, ...]) -> FeatureBlock:
    result.validate()
    candidates = [block for block in result.blocks if tuple(block.feature_names) == names]
    if len(candidates) != 1:
        raise ValueError(f"{modality} result does not contain exactly one expected block")
    block = candidates[0]
    if block.time_scale.value != "night":
        raise ValueError(f"{modality} combined block must be night-level")
    if block.values.shape != (len(names),):
        raise ValueError(f"{modality} block shape mismatch: {block.values.shape}")
    return block


def _source_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"method": "stat", "path": str(path.resolve()), "size_bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def _compact_branch_metadata(result: BranchResult, modality: str) -> dict[str, Any]:
    raw = dict(result.metadata)
    if modality == "caisr":
        keys = (
            "annotation_source", "source_path", "availability", "valid_stage_epoch_count", "stage_probability_rows_used",
            "sleep_epochs", "tst_sec", "record_duration_sec", "arousal_segment_count",
            "respiratory_segment_count", "limb_segment_count", "plm_sleep_event_count",
            "arousal_sleep_event_duration_sec", "respiratory_sleep_event_duration_sec",
            "limb_sleep_event_duration_sec", "plm_sleep_event_duration_sec",
        )
    elif modality == "eeg":
        keys = (
            "feature_version", "feature_names", "selected_feature_names", "skipped_feature_names",
            "selected_band_names", "channel_present", "stage_counts", "stage_valid_seconds",
            "stage_qc_valid_seconds", "feature_support_seconds", "stage_valid_window_counts",
            "region_common_valid_window_counts", "cross_region_common_valid_window_counts",
            "cross_region_bilateral_valid_window_counts", "qc_true_counts", "psd_window_count",
            "psd_band_integrals_evaluated", "annotation_source",
        )
    else:
        keys = (
            "feature_version", "record_status", "stage_timebase_valid", "valid_stage_epoch_count",
            "ecg_exact_n2_windows", "ecg_exact_n3_windows", "ecg_exact_n2_qc_ge120_windows",
            "ecg_exact_n3_qc_ge120_windows", "ecg_exact_n2_hrv_valid_windows",
            "ecg_exact_n3_hrv_valid_windows", "ecg_exact_selected_n2_windows",
            "ecg_exact_selected_n3_windows", "selection_n2_windows", "selection_n3_windows",
            "s01_invalid_reason", "s02_invalid_reason", "ecg_unit_meta",
        )
    return {key: _json_safe(raw[key]) for key in keys if key in raw}


def _metadata_for_combined(context: SampleContext, results: Mapping[str, BranchResult]) -> dict[str, Any]:
    attrs = context.h5.attrs
    record = context.record
    duration = float(attrs.get("complete_epoch_count", context.epoch_count)) * float(attrs.get("epoch_duration_sec", 30.0))
    annotation_source = str(results["caisr"].metadata.get("annotation_source", "unavailable"))
    recipe = {
        "extractor": "canonical_hdf5_to_combined_npz",
        "schema_revision": SCHEMA_REVISION,
        "caisr_feature_version": str(results["caisr"].blocks[0].version),
        "eeg_feature_version": str(results["eeg"].blocks[0].version),
        "ecg_feature_version": str(results["ecg"].blocks[0].version),
        "eeg_selected_feature_names": list(EEG12_FEATURE_NAMES),
    }
    recipe_json = json.dumps(recipe, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    channel_names_fn = getattr(context, "channel_names", None)
    metadata_fn = getattr(context, "metadata", None)
    def context_channel_names(modality: str, fallback: Any) -> list[str]:
        return list(channel_names_fn(modality)) if callable(channel_names_fn) else [str(name) for name in fallback]
    def context_metadata(modality: str, field: str) -> Any:
        return metadata_fn(modality, field) if callable(metadata_fn) else None
    metadata = {
        "record_id": record.record_id,
        "site": record.site,
        "subject_id": record.subject_id,
        "session_id": record.session_id,
        "schema_revision": SCHEMA_REVISION,
        "schema_hash": schema_hash(),
        "source_fingerprint": _source_fingerprint(record.cache_h5_path),
        "canonical_hdf5_source": {
            "path": str(record.cache_h5_path.resolve()),
            "cache_schema_version": _json_safe(attrs.get("cache_schema_version", "")),
            "preprocessing_version": _json_safe(attrs.get("preprocessing_version", "")),
        },
        "cache_schema_version": _json_safe(attrs.get("cache_schema_version", "")),
        "preprocessing_version": _json_safe(attrs.get("preprocessing_version", "")),
        "extraction_recipe_fingerprint": hashlib.sha256(recipe_json.encode("utf-8")).hexdigest(),
        "epoch_count": int(context.epoch_count),
        "record_duration_sec": duration,
        "annotation_source": annotation_source,
        "caisr": {
            **_compact_branch_metadata(results["caisr"], "caisr"),
            "channel_names": context_channel_names("caisr", results["caisr"].blocks[0].feature_names),
        },
        "eeg": {
            **_compact_branch_metadata(results["eeg"], "eeg"),
            "channel_names": context_channel_names("eeg", results["eeg"].blocks[0].feature_names),
            "physical_dimension": _json_safe(context_metadata("eeg", "physical_dimension")),
            "selected_source_labels": _json_safe(context_metadata("eeg", "selected_source_labels")),
        },
        "ecg": {
            **_compact_branch_metadata(results["ecg"], "ecg"),
            "channel_names": context_channel_names("ecg", results["ecg"].blocks[0].feature_names),
            "physical_dimension": _json_safe(context_metadata("ecg", "physical_dimension")),
        },
    }
    return _json_safe(metadata)


def extract_combined_one(manifest_path: Path | str, record_id: str, output_path: Path | str) -> StoredFeatureRecord:
    """从一个 canonical HDF5 提取 CAISR20+EEG12+ECG-S2 并原子写入 NPZ。"""

    manifest = Manifest.from_tsv(manifest_path)
    record = manifest.get(record_id)
    target = Path(output_path)
    if target.name != f"{record_id}.npz" or target.parent.name != record.site:
        raise ValueError("output_path must be <root>/<site>/<record_id>.npz")
    with SampleContext(record) as context:
        results = {
            "caisr": call_extractor(BRANCH_REGISTRY["caisr"], context, {}),
            "eeg": call_extractor(BRANCH_REGISTRY["eeg"], context, {"selected_feature_names": EEG12_FEATURE_NAMES}),
            "ecg": call_extractor(BRANCH_REGISTRY["ecg"], context, {}),
        }
        blocks = {
            "caisr": _branch_block(results["caisr"], "caisr", tuple(CAISR20_FEATURE_NAMES)),
            "eeg": _branch_block(results["eeg"], "eeg", tuple(EEG12_FEATURE_NAMES)),
            "ecg": _branch_block(results["ecg"], "ecg", tuple(ECG_FEATURE_NAMES)),
        }
        stored = write_one(
            target,
            record.record_id,
            record.site,
            {modality: ModalityFeatures(block.feature_names, block.values, block.valid_mask, block.invalid_reason) for modality, block in blocks.items()},
            _metadata_for_combined(context, results),
            overwrite=False,
        )
    return validate_record(target)


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
