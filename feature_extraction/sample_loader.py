"""manifest 驱动的只读 canonical HDF5 样本入口。

本模块只打开 manifest 指向的 HDF5 和 demographics.csv。manifest 中的 PSG/CAISR
EDF 路径只作为 provenance 保存，代码没有 EDF reader、EDF fallback 或 cache_record
调用。HDF5 句柄严格绑定在 SampleContext 上下文中，dataset 访问按需读取。
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np


MANIFEST_COLUMNS = (
    "record_id", "site", "subject_id", "session_id", "cache_h5_path",
    "demographics_csv_path", "psg_edf_path", "caisr_edf_path",
)
DEMOGRAPHIC_FIELDS = (
    "age", "sex_Female", "sex_Male", "sex_OtherUnknown", "race_Asian",
    "race_Black", "race_Others", "race_Unavailable", "race_White", "BMI",
)
_RECORD_RE = re.compile(r"^sub-(?P<subject>[^_]+)_ses-(?P<session>[^_]+)$")
_MAX_UNBOUNDED_READ_ELEMENTS = 1_000_000


class ManifestError(ValueError):
    """manifest 或身份匹配不符合框架契约。"""


def _identity_part(value: str) -> str:
    return value.strip().removeprefix("sub-").removeprefix("ses-")


def _resolve(path_text: str, manifest_path: Path) -> Path:
    path = Path(path_text)
    return (path if path.is_absolute() else manifest_path.parent / path).resolve()


@dataclass(frozen=True)
class ManifestRecord:
    record_id: str
    site: str
    subject_id: str
    session_id: str
    cache_h5_path: Path
    demographics_csv_path: Path
    psg_edf_path: str
    caisr_edf_path: str


class Manifest:
    """经过唯一性和身份检查的 manifest 索引。"""

    def __init__(self, path: Path, records: Mapping[str, ManifestRecord]) -> None:
        self.path = path
        self.records = dict(records)

    @classmethod
    def from_tsv(cls, path: Path | str) -> "Manifest":
        manifest_path = Path(path).resolve()
        if not manifest_path.is_file():
            raise FileNotFoundError(f"manifest not found: {manifest_path}")
        records: dict[str, ManifestRecord] = {}
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if tuple(reader.fieldnames or ()) != MANIFEST_COLUMNS:
                raise ManifestError(f"manifest columns must be {MANIFEST_COLUMNS}")
            for line, row in enumerate(reader, start=2):
                values = {name: (row.get(name) or "").strip() for name in MANIFEST_COLUMNS}
                if not values["record_id"] or not values["site"]:
                    raise ManifestError(f"line {line}: record_id and site are required")
                if values["record_id"] in records:
                    raise ManifestError(f"duplicate record_id: {values['record_id']}")
                match = _RECORD_RE.fullmatch(values["record_id"])
                if match is None:
                    raise ManifestError(f"line {line}: invalid record_id {values['record_id']!r}")
                if _identity_part(values["subject_id"]) != match.group("subject"):
                    raise ManifestError(f"line {line}: subject_id does not match record_id")
                if _identity_part(values["session_id"]) != match.group("session"):
                    raise ManifestError(f"line {line}: session_id does not match record_id")
                records[values["record_id"]] = ManifestRecord(
                    record_id=values["record_id"], site=values["site"],
                    subject_id=values["subject_id"], session_id=values["session_id"],
                    cache_h5_path=_resolve(values["cache_h5_path"], manifest_path),
                    demographics_csv_path=_resolve(values["demographics_csv_path"], manifest_path),
                    psg_edf_path=values["psg_edf_path"], caisr_edf_path=values["caisr_edf_path"],
                )
        return cls(manifest_path, records)

    def get(self, record_id: str) -> ManifestRecord:
        try:
            return self.records[record_id]
        except KeyError as exc:
            raise ManifestError(f"record_id not present in manifest: {record_id}") from exc


@dataclass
class SampleContext:
    """一个 record 的短生命周期 HDF5 上下文。"""

    record: ManifestRecord
    _handle: h5py.File | None = None

    def __enter__(self) -> "SampleContext":
        if self._handle is not None:
            raise RuntimeError("SampleContext cannot be entered twice")
        self._handle = h5py.File(self.record.cache_h5_path, "r")
        attrs = self._handle.attrs
        if _text(attrs.get("record_id", "")) != self.record.record_id:
            self.close()
            raise ManifestError("HDF5 record_id does not match manifest")
        if _text(attrs.get("site", "")) != self.record.site:
            self.close()
            raise ManifestError("HDF5 site does not match manifest")
        if _text(attrs.get("cache_schema_version", "")) != "1.0":
            self.close()
            raise ManifestError("HDF5 cache_schema_version is not canonical v1")
        required_groups = ("signals", "quality", "metadata", "annotations", "timebase")
        missing_groups = [name for name in required_groups if name not in self._handle]
        if missing_groups:
            self.close()
            raise ManifestError(f"HDF5 missing canonical groups: {missing_groups}")
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    @property
    def h5(self) -> h5py.File:
        if self._handle is None:
            raise RuntimeError("HDF5 handle is closed; use SampleContext as a context manager")
        return self._handle

    @property
    def epoch_count(self) -> int:
        return int(self.h5.attrs["complete_epoch_count"])

    def dataset(self, path: str) -> h5py.Dataset:
        return self.h5[path]

    def read(self, path: str, selection: Any = None) -> np.ndarray:
        dataset = self.dataset(path)
        if selection is None and dataset.size > _MAX_UNBOUNDED_READ_ELEMENTS:
            raise ValueError(
                f"refusing unbounded read of {path} with {dataset.size} elements; "
                "use dataset() or iter_signal_epochs()"
            )
        return np.asarray(dataset[...] if selection is None else dataset[selection])

    def iter_signal_epochs(self, modality: str, batch_epochs: int = 8):
        """按 epoch batch 提供 waveform，yield 时 context 必须仍处于打开状态。"""

        if batch_epochs <= 0:
            raise ValueError("batch_epochs must be positive")
        dataset = self.dataset(f"signals/{modality}")
        for start in range(0, int(dataset.shape[0]), batch_epochs):
            end = min(int(dataset.shape[0]), start + batch_epochs)
            yield start, np.asarray(dataset[start:end])

    def channel_names(self, modality: str) -> tuple[str, ...]:
        return tuple(_text(value) for value in self.read(f"metadata/{modality}/channel_names"))

    def metadata(self, modality: str, field: str) -> np.ndarray:
        return self.read(f"metadata/{modality}/{field}")

    def channel_present(self, modality: str) -> np.ndarray:
        return self.read(f"quality/channel_present/{modality}").astype(bool, copy=False)

    def quality_dataset(self, quality_name: str, modality: str) -> h5py.Dataset:
        return self.dataset(f"quality/{quality_name}/{modality}")

    def caisr_dataset(self, name: str) -> h5py.Dataset:
        return self.dataset(f"annotations/caisr/{name}")

    def timebase(self) -> dict[str, np.ndarray]:
        return {name: self.read(f"timebase/{name}") for name in ("epoch_index", "epoch_start_sec", "epoch_end_sec")}

    def demographics(self) -> Mapping[str, str]:
        return _demographics_lookup(self.record.demographics_csv_path, self.record)


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


@lru_cache(maxsize=8)
def _demographics_index(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"demographics CSV not found: {path}")
    index: dict[tuple[str, str], dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        required = {"BidsFolder", "SessionID"}
        if not required.issubset(fields):
            raise ManifestError(f"demographics missing columns: {sorted(required - fields)}")
        for row_number, row in enumerate(reader, start=2):
            key = ((row.get("BidsFolder") or "").strip(), (row.get("SessionID") or "").strip())
            if not all(key):
                continue
            if key in index:
                raise ManifestError(f"duplicate demographics key at line {row_number}: {key}")
            index[key] = {field: (row.get(field) or "").strip() for field in DEMOGRAPHIC_FIELDS}
    return index


def _demographics_lookup(path: Path, record: ManifestRecord) -> Mapping[str, str]:
    index = _demographics_index(path)
    subject_options = {record.subject_id, f"sub-{_identity_part(record.subject_id)}"}
    session_options = {record.session_id, f"ses-{_identity_part(record.session_id)}"}
    for key in ((subject, session) for subject in subject_options for session in session_options):
        if key in index:
            return dict(index[key])
    raise ManifestError(f"demographics key not found for {record.record_id}")
