"""统一 34 维特征 schema 与 B1/D7 读取 profile。

本模块只描述名称、单位、dtype、shape 和稳定 schema hash；不读取 HDF5、
不计算特征，也不执行训练折填补或标准化。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


SCHEMA_REVISION = "psg_combined34_v1"

CAISR20_FEATURE_NAMES = (
    "caisr_sleep_tst_sec", "caisr_sleep_se", "caisr_sleep_sol_sec",
    "caisr_sleep_rem_latency_sec", "caisr_sleep_waso_sec", "caisr_sleep_n1_pct",
    "caisr_sleep_n2_pct", "caisr_sleep_n3_pct", "caisr_sleep_rem_pct",
    "caisr_sleep_transition_rate", "caisr_sleep_short_bout_ratio",
    "caisr_sleep_mean_stage_entropy", "caisr_arousal_arousal_index",
    "caisr_arousal_arousal_burden_ratio", "caisr_arousal_mean_arousal_duration",
    "caisr_respiratory_respiratory_event_index",
    "caisr_respiratory_respiratory_burden_ratio",
    "caisr_respiratory_mean_resp_duration", "caisr_limb_limb_movement_index",
    "caisr_limb_plmi",
)
CAISR20_UNITS = (
    "sec", "ratio", "sec", "sec", "sec", "ratio", "ratio", "ratio", "ratio",
    "per_hour", "ratio", "ratio", "per_hour", "ratio", "sec", "per_hour",
    "ratio", "sec", "per_hour", "per_hour",
)

EEG12_FEATURE_NAMES = (
    "eeg_n2_central_sigma_relative_power",
    "eeg_n2_frontal_slow_sigma_relative_power",
    "eeg_n2_central_fast_sigma_relative_power",
    "eeg_n2_occipital_fast_sigma_relative_power",
    "eeg_n3_frontal_occipital_swa_log_gradient",
    "eeg_n2_frontal_occipital_fast_sigma_log_gradient",
    "eeg_rem_occipital_frontal_slowing_difference",
    "eeg_n3_frontal_swa_left_right_log_asymmetry",
    "eeg_n2_central_sigma_left_right_log_asymmetry",
    "eeg_rem_occipital_slowing_log_ratio",
    "eeg_rem_central_slowing_log_ratio",
    "eeg_rem_frontal_slowing_log_ratio",
)
EEG12_UNITS = (
    "ratio", "ratio", "ratio", "ratio", "log_ratio", "log_ratio",
    "log_ratio_difference", "log_ratio", "log_ratio", "log_ratio", "log_ratio", "log_ratio",
)

ECG_FEATURE_NAMES = ("S01_N2_HF_relative", "S02_N3_HF_absolute_ms2")
ECG_UNITS = ("ratio", "ms^2")

MODALITY_NAMES = {"caisr": CAISR20_FEATURE_NAMES, "eeg": EEG12_FEATURE_NAMES, "ecg": ECG_FEATURE_NAMES}
MODALITY_UNITS = {"caisr": CAISR20_UNITS, "eeg": EEG12_UNITS, "ecg": ECG_UNITS}
MODALITY_DTYPES = {"caisr": "float32", "eeg": "float32", "ecg": "float64"}
MODALITY_SHAPES = {"caisr": [20], "eeg": [12], "ecg": [2]}
B1_EEG_FEATURE_NAMES = EEG12_FEATURE_NAMES[:9]


def _schema_contract() -> dict[str, Any]:
    return {
        "schema_revision": SCHEMA_REVISION,
        "modalities": {
            modality: {
                "feature_names": list(MODALITY_NAMES[modality]),
                "units": list(MODALITY_UNITS[modality]),
                "dtype": MODALITY_DTYPES[modality],
                "shape": list(MODALITY_SHAPES[modality]),
                "invalid_contract": "valid finite; invalid NaN with nonempty reason",
            }
            for modality in ("caisr", "eeg", "ecg")
        },
        "string_contract": "numpy native unicode; no object or pickle",
    }


def schema_contract() -> dict[str, Any]:
    """返回可 JSON 序列化的 canonical schema contract 副本。"""

    return json.loads(canonical_schema_json())


def canonical_schema_json() -> str:
    return json.dumps(_schema_contract(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def schema_hash() -> str:
    return hashlib.sha256(canonical_schema_json().encode("utf-8")).hexdigest()


def validate_schema_hash(value: str) -> None:
    if str(value) != schema_hash():
        raise ValueError(f"schema_hash mismatch: expected {schema_hash()}, got {value}")


@dataclass(frozen=True)
class FeatureProfile:
    """模型读取期的名称 profile；不改变统一缓存内容。"""

    name: str
    base_feature_names: tuple[str, ...]
    hrv_feature_names: tuple[str, ...] = ECG_FEATURE_NAMES

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self.base_feature_names + self.hrv_feature_names

    @property
    def base_dim(self) -> int:
        return len(self.base_feature_names)

    @property
    def hrv_dim(self) -> int:
        return len(self.hrv_feature_names)

    @property
    def total_dim(self) -> int:
        return self.base_dim + self.hrv_dim

    def select_indices(self, names: Sequence[str]) -> tuple[int, ...]:
        index = {name: position for position, name in enumerate(names)}
        try:
            return tuple(index[name] for name in self.feature_names)
        except KeyError as exc:
            raise ValueError(f"profile {self.name} feature is missing: {exc.args[0]}") from exc

    def validate_order(self, names: Sequence[str]) -> None:
        actual = tuple(str(name) for name in names)
        if actual != self.feature_names:
            raise ValueError(f"profile {self.name} feature order mismatch")


B1_PROFILE = FeatureProfile("B1", CAISR20_FEATURE_NAMES + B1_EEG_FEATURE_NAMES)
D7_PROFILE = FeatureProfile("D7", CAISR20_FEATURE_NAMES + EEG12_FEATURE_NAMES)
PROFILES: Mapping[str, FeatureProfile] = {"B1": B1_PROFILE, "D7": D7_PROFILE}


def get_profile(name: str) -> FeatureProfile:
    try:
        return PROFILES[str(name).upper()]
    except KeyError as exc:
        raise ValueError(f"unknown feature profile: {name}") from exc


__all__ = [
    "SCHEMA_REVISION", "CAISR20_FEATURE_NAMES", "EEG12_FEATURE_NAMES", "ECG_FEATURE_NAMES",
    "CAISR20_UNITS", "EEG12_UNITS", "ECG_UNITS", "MODALITY_NAMES", "MODALITY_UNITS",
    "MODALITY_DTYPES", "MODALITY_SHAPES", "B1_EEG_FEATURE_NAMES", "FeatureProfile",
    "B1_PROFILE", "D7_PROFILE", "PROFILES", "get_profile", "schema_contract",
    "canonical_schema_json", "schema_hash", "validate_schema_hash",
]
