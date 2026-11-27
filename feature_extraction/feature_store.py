"""安全的单样本 canonical34 NPZ 存储和模型读取接口。

本模块只处理已经计算好的三组特征块；不读取 HDF5/EDF，不计算特征，不拟合
scaler，不读取标签。所有发布文件都使用 allow_pickle=False、固定身份校验和
同文件系统 atomic rename。
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .feature_profiles import (
    CAISR20_FEATURE_NAMES,
    ECG_FEATURE_NAMES,
    EEG12_FEATURE_NAMES,
    MODALITY_DTYPES,
    MODALITY_NAMES,
    SCHEMA_REVISION,
    schema_hash,
    validate_schema_hash,
    get_profile,
)


NPZ_FIELDS = (
    "record_id", "site",
    "caisr_names", "caisr_values", "caisr_valid_mask", "caisr_invalid_reason",
    "eeg_names", "eeg_values", "eeg_valid_mask", "eeg_invalid_reason",
    "ecg_names", "ecg_values", "ecg_valid_mask", "ecg_invalid_reason",
    "metadata_json",
)
MODALITIES = ("caisr", "eeg", "ecg")


class FeatureStoreError(ValueError):
    """NPZ schema、身份或 metadata 契约错误。"""


@dataclass(frozen=True)
class ModalityFeatures:
    names: tuple[str, ...]
    values: np.ndarray
    valid_mask: np.ndarray
    invalid_reason: np.ndarray

    def __post_init__(self) -> None:
        names = tuple(str(value) for value in self.names)
        values = np.asarray(self.values)
        valid_mask = np.asarray(self.valid_mask)
        invalid_reason = np.asarray(self.invalid_reason)
        if values.ndim != 1 or valid_mask.shape != values.shape or invalid_reason.shape != values.shape:
            raise FeatureStoreError("modality arrays must be one-dimensional with matching shape")
        if len(names) != values.size:
            raise FeatureStoreError("feature name count does not match values")
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "valid_mask", valid_mask)
        object.__setattr__(self, "invalid_reason", invalid_reason)


@dataclass(frozen=True)
class StoredFeatureRecord:
    record_id: str
    site: str
    modalities: Mapping[str, ModalityFeatures]
    metadata: Mapping[str, Any]


def _expected_names(modality: str) -> tuple[str, ...]:
    try:
        return tuple(MODALITY_NAMES[modality])
    except KeyError as exc:
        raise FeatureStoreError(f"unknown modality: {modality}") from exc


def _validate_modality(modality: str, block: ModalityFeatures) -> None:
    expected_names = _expected_names(modality)
    if block.names != expected_names:
        raise FeatureStoreError(f"{modality} feature name/order mismatch")
    expected_dtype = np.dtype(MODALITY_DTYPES[modality])
    if block.values.dtype != expected_dtype:
        raise FeatureStoreError(f"{modality} values must have dtype {expected_dtype}, got {block.values.dtype}")
    if block.values.shape != (len(expected_names),):
        raise FeatureStoreError(f"{modality} values must have shape {(len(expected_names),)}")
    if block.valid_mask.dtype != np.dtype(bool):
        raise FeatureStoreError(f"{modality} valid_mask must have dtype bool")
    if block.invalid_reason.dtype.kind != "U":
        raise FeatureStoreError(f"{modality} invalid_reason must use native Unicode dtype")
    if np.isinf(block.values).any():
        raise FeatureStoreError(f"{modality} values contain Inf")
    if np.any(block.valid_mask & ~np.isfinite(block.values)):
        raise FeatureStoreError(f"{modality} valid values must be finite")
    if np.any(~block.valid_mask & ~np.isnan(block.values)):
        raise FeatureStoreError(f"{modality} invalid values must be NaN")
    if np.any(block.valid_mask & (block.invalid_reason != "")):
        raise FeatureStoreError(f"{modality} valid values must have empty invalid_reason")
    if np.any(~block.valid_mask & (block.invalid_reason == "")):
        raise FeatureStoreError(f"{modality} invalid values require invalid_reason")


def _strict_metadata(record_id: str, site: str, metadata: Mapping[str, Any] | None) -> tuple[dict[str, Any], str]:
    if metadata is None:
        payload: dict[str, Any] = {}
    elif isinstance(metadata, Mapping):
        payload = dict(metadata)
    else:
        raise FeatureStoreError("metadata must be a mapping")
    for key, expected in (("record_id", record_id), ("site", site), ("schema_revision", SCHEMA_REVISION)):
        if key in payload and str(payload[key]) != expected:
            raise FeatureStoreError(f"metadata {key} does not match record identity/schema")
        payload[key] = expected
    expected_hash = schema_hash()
    if "schema_hash" in payload:
        validate_schema_hash(str(payload["schema_hash"]))
    payload["schema_hash"] = expected_hash
    try:
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise FeatureStoreError(f"metadata is not strict JSON: {exc}") from exc
    if len(encoded.encode("utf-8")) > 65536:
        raise FeatureStoreError("metadata_json exceeds 65536 UTF-8 bytes")
    return payload, encoded


def _unicode_scalar(value: str) -> np.ndarray:
    text = str(value)
    return np.asarray(text, dtype=f"<U{max(1, len(text))}")


def _unicode_vector(values: Sequence[str]) -> np.ndarray:
    texts = [str(value) for value in values]
    width = max([1, *[len(value) for value in texts]])
    return np.asarray(texts, dtype=f"<U{width}")


def _path_identity(path: Path, record_id: str, site: str) -> None:
    if path.suffix != ".npz" or path.name != f"{record_id}.npz":
        raise FeatureStoreError("NPZ filename must be <record_id>.npz")
    if path.parent.name != site:
        raise FeatureStoreError("NPZ parent directory must equal site")


def _scalar_text(array: np.ndarray, field: str) -> str:
    if array.dtype.kind != "U" or array.shape != ():
        raise FeatureStoreError(f"{field} must be a native Unicode scalar")
    return str(array.item())


def _read_archive(path: Path, *, validate_path: bool = True) -> StoredFeatureRecord:
    try:
        with np.load(path, allow_pickle=False) as archive:
            fields = tuple(sorted(archive.files))
            if fields != tuple(sorted(NPZ_FIELDS)):
                missing = sorted(set(NPZ_FIELDS) - set(fields))
                extra = sorted(set(fields) - set(NPZ_FIELDS))
                raise FeatureStoreError(f"NPZ fields mismatch; missing={missing}, extra={extra}")
            record_id = _scalar_text(np.asarray(archive["record_id"]), "record_id")
            site = _scalar_text(np.asarray(archive["site"]), "site")
            if not record_id or not site or "/" in record_id or "\\" in record_id:
                raise FeatureStoreError("invalid record identity")
            if validate_path:
                _path_identity(path, record_id, site)
            metadata_raw = _scalar_text(np.asarray(archive["metadata_json"]), "metadata_json")
            if len(metadata_raw.encode("utf-8")) > 65536:
                raise FeatureStoreError("metadata_json exceeds 65536 UTF-8 bytes")
            try:
                metadata = json.loads(metadata_raw, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise FeatureStoreError(f"invalid metadata_json: {exc}") from exc
            if not isinstance(metadata, dict):
                raise FeatureStoreError("metadata_json must encode an object")
            if str(metadata.get("record_id", "")) != record_id or str(metadata.get("site", "")) != site:
                raise FeatureStoreError("metadata identity mismatch")
            if str(metadata.get("schema_revision", "")) != SCHEMA_REVISION:
                raise FeatureStoreError("unknown schema revision")
            validate_schema_hash(str(metadata.get("schema_hash", "")))
            blocks: dict[str, ModalityFeatures] = {}
            for modality in MODALITIES:
                names_array = np.asarray(archive[f"{modality}_names"]).copy()
                if names_array.dtype.kind != "U" or names_array.ndim != 1:
                    raise FeatureStoreError(f"{modality}_names must be a native Unicode vector")
                names = tuple(str(value) for value in names_array.tolist())
                block = ModalityFeatures(
                    names,
                    np.asarray(archive[f"{modality}_values"]).copy(),
                    np.asarray(archive[f"{modality}_valid_mask"]).copy(),
                    np.asarray(archive[f"{modality}_invalid_reason"]).copy(),
                )
                _validate_modality(modality, block)
                blocks[modality] = block
    except FeatureStoreError:
        raise
    except Exception as exc:
        raise FeatureStoreError(f"unable to read NPZ {path}: {type(exc).__name__}: {exc}") from exc
    return StoredFeatureRecord(record_id, site, blocks, metadata)


def _coerce_blocks(blocks: Mapping[str, ModalityFeatures]) -> dict[str, ModalityFeatures]:
    if set(blocks) != set(MODALITIES):
        raise FeatureStoreError(f"exactly these modalities are required: {MODALITIES}")
    result = {modality: blocks[modality] for modality in MODALITIES}
    for modality, block in result.items():
        if not isinstance(block, ModalityFeatures):
            raise FeatureStoreError(f"{modality} must be ModalityFeatures")
        _validate_modality(modality, block)
    return result


def write_one(
    path: Path | str,
    record_id: str,
    site: str,
    blocks: Mapping[str, ModalityFeatures],
    metadata: Mapping[str, Any] | None = None,
    *,
    overwrite: bool = False,
) -> StoredFeatureRecord:
    """安全写入一个 NPZ；默认拒绝覆盖已有发布文件。"""

    target = Path(path)
    record_id, site = str(record_id), str(site)
    if not record_id or "/" in record_id or "\\" in record_id or not site:
        raise FeatureStoreError("record_id/site is invalid")
    _path_identity(target, record_id, site)
    if target.exists() and not overwrite:
        raise FileExistsError(target)
    validated = _coerce_blocks(blocks)
    metadata_payload, metadata_json = _strict_metadata(record_id, site, metadata)
    arrays: dict[str, np.ndarray] = {
        "record_id": _unicode_scalar(record_id),
        "site": _unicode_scalar(site),
        "metadata_json": _unicode_scalar(metadata_json),
    }
    for modality in MODALITIES:
        block = validated[modality]
        arrays[f"{modality}_names"] = _unicode_vector(block.names)
        arrays[f"{modality}_values"] = block.values
        arrays[f"{modality}_valid_mask"] = block.valid_mask
        arrays[f"{modality}_invalid_reason"] = _unicode_vector(block.invalid_reason.tolist())
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent, delete=False) as handle:
            temp_name = handle.name
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        _read_archive(Path(temp_name), validate_path=False)
        if target.exists() and not overwrite:
            raise FileExistsError(target)
        os.replace(temp_name, target)
        temp_name = None
        return _read_archive(target)
    finally:
        if temp_name is not None:
            try:
                Path(temp_name).unlink()
            except FileNotFoundError:
                pass


def read_one(path: Path | str) -> StoredFeatureRecord:
    return _read_archive(Path(path), validate_path=True)


def validate_record(path: Path | str) -> StoredFeatureRecord:
    """完整读取并校验一个已发布 NPZ；成功时返回校验后的记录。"""

    return _read_archive(Path(path), validate_path=True)


def load_records(root: Path | str, record_ids: Sequence[str]) -> list[StoredFeatureRecord]:
    ids = [str(value) for value in record_ids]
    if len(ids) != len(set(ids)):
        raise ValueError("record_ids must be unique")
    root_path = Path(root)
    records: list[StoredFeatureRecord] = []
    for record_id in ids:
        matches = sorted(root_path.glob(f"*/{record_id}.npz"))
        if not matches:
            raise KeyError(record_id)
        if len(matches) != 1:
            raise FeatureStoreError(f"record_id maps to multiple NPZ files: {record_id}")
        records.append(read_one(matches[0]))
    return records


def build_model_view(records: Sequence[StoredFeatureRecord], profile: str | Any) -> dict[str, Any]:
    selected_profile = get_profile(profile) if isinstance(profile, str) else profile
    if not records:
        raise ValueError("at least one record is required")
    base_values = []
    base_masks = []
    hrv_values = []
    hrv_masks = []
    record_ids = []
    sites = []
    for record in records:
        if not isinstance(record, StoredFeatureRecord):
            raise TypeError("records must contain StoredFeatureRecord")
        caisr = record.modalities["caisr"]
        eeg = record.modalities["eeg"]
        ecg = record.modalities["ecg"]
        value_by_name = {**dict(zip(caisr.names, caisr.values)), **dict(zip(eeg.names, eeg.values))}
        mask_by_name = {**dict(zip(caisr.names, caisr.valid_mask)), **dict(zip(eeg.names, eeg.valid_mask))}
        base_values.append([value_by_name[name] for name in selected_profile.base_feature_names])
        base_masks.append([mask_by_name[name] for name in selected_profile.base_feature_names])
        hrv_values.append([dict(zip(ecg.names, ecg.values))[name] for name in selected_profile.hrv_feature_names])
        hrv_masks.append([dict(zip(ecg.names, ecg.valid_mask))[name] for name in selected_profile.hrv_feature_names])
        record_ids.append(record.record_id)
        sites.append(record.site)
    result = {
        "base_feature_names": selected_profile.base_feature_names,
        "base_values": np.asarray(base_values, dtype=np.float32),
        "base_valid_mask": np.asarray(base_masks, dtype=bool),
        "hrv_feature_names": selected_profile.hrv_feature_names,
        "hrv_values": np.asarray(hrv_values, dtype=np.float64),
        "hrv_valid_mask": np.asarray(hrv_masks, dtype=bool),
        "record_id": tuple(record_ids),
        "site": tuple(sites),
    }
    # 显式模型输入别名；原有 key 保留以兼容历史读取方。
    result.update({
        "X_base_raw": result["base_values"],
        "M_base": result["base_valid_mask"],
        "X_hrv_raw": result["hrv_values"],
        "M_hrv": result["hrv_valid_mask"],
    })
    return result


class FeatureStore:
    """按 site/record_id 管理 combined NPZ 的薄封装。"""

    def __init__(self, root: Path | str):
        self.root = Path(root)

    def path_for(self, record_id: str, site: str) -> Path:
        return self.root / str(site) / f"{record_id}.npz"

    def write_one(self, record_id: str, site: str, blocks: Mapping[str, ModalityFeatures], metadata: Mapping[str, Any] | None = None, *, overwrite: bool = False) -> StoredFeatureRecord:
        return write_one(self.path_for(record_id, site), record_id, site, blocks, metadata, overwrite=overwrite)

    def read_one(self, record_id: str, site: str) -> StoredFeatureRecord:
        """按显式 site/record_id 读取一条 NPZ；不会回退到 TSV。"""
        return read_one(self.path_for(record_id, site))

    def load_record(self, record_id: str, site: str) -> StoredFeatureRecord:
        return self.read_one(record_id, site)

    def load_records_by_site(self, keys: Sequence[tuple[str, str]]) -> list[StoredFeatureRecord]:
        pairs = [(str(record_id), str(site)) for record_id, site in keys]
        if len({record_id for record_id, _ in pairs}) != len(pairs):
            raise FeatureStoreError("duplicate record_id in model input request")
        records = [self.read_one(record_id, site) for record_id, site in pairs]
        if len({record.site for record in records}) != len({site for _, site in pairs}):
            raise FeatureStoreError("NPZ site identity mismatch")
        return records

    def build_model_input(self, keys: Sequence[tuple[str, str]], profile: str | Any) -> dict[str, Any]:
        records = self.load_records_by_site(keys)
        return build_model_view(records, profile)

    def validate_record(self, record_id: str, site: str) -> StoredFeatureRecord:
        return validate_record(self.path_for(record_id, site))

    def load_records(self, record_ids: Sequence[str]) -> list[StoredFeatureRecord]:
        return load_records(self.root, record_ids)

    def build_model_view(self, records: Sequence[StoredFeatureRecord], profile: str | Any) -> dict[str, Any]:
        return build_model_view(records, profile)


class FeatureLoader(FeatureStore):
    """训练读取侧的名称/profile 组装接口，不做 scaler 或标签读取。"""


__all__ = [
    "NPZ_FIELDS", "MODALITIES", "FeatureStoreError", "ModalityFeatures", "StoredFeatureRecord",
    "FeatureStore", "FeatureLoader", "write_one", "read_one", "validate_record", "load_records", "build_model_view",
]
