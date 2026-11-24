"""训练折专用的 CAISR20 v2 特征变换与标准化。"""

from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.preprocessing import StandardScaler


CAISR20_FEATURE_NAMES = (
    "caisr_sleep_tst_sec",
    "caisr_sleep_se",
    "caisr_sleep_sol_sec",
    "caisr_sleep_rem_latency_sec",
    "caisr_sleep_waso_sec",
    "caisr_sleep_n1_pct",
    "caisr_sleep_n2_pct",
    "caisr_sleep_n3_pct",
    "caisr_sleep_rem_pct",
    "caisr_sleep_transition_rate",
    "caisr_sleep_short_bout_ratio",
    "caisr_sleep_mean_stage_entropy",
    "caisr_arousal_arousal_index",
    "caisr_arousal_arousal_burden_ratio",
    "caisr_arousal_mean_arousal_duration",
    "caisr_respiratory_respiratory_event_index",
    "caisr_respiratory_respiratory_burden_ratio",
    "caisr_respiratory_mean_resp_duration",
    "caisr_limb_limb_movement_index",
    "caisr_limb_plmi",
)

SCALER_VERSION = "caisr_v2_train_fold_scaler_v1"

HRV_A14_FEATURE_NAMES = (
    "ecg_hrv_median_nn_w300_equal_mean",
    "ecg_hrv_mcvnn_w300_equal_mean",
    "ecg_hrv_cvnn_w300_equal_mean",
    "ecg_hrv_cvsd_w300_equal_mean",
    "ecg_hrv_pnn20_w300_equal_mean",
    "ecg_hrv_lf_power_w300_equal_mean",
    "ecg_hrv_hf_power_w300_equal_mean",
    "ecg_hrv_hf_relative_lfhf_w300_equal_mean",
    "ecg_hrv_sd1_sd2_w300_equal_mean",
    "ecg_hrv_symbolic_equalprob4_0v_w300_equal_mean",
    "ecg_hrv_symbolic_equalprob4_2uv_w300_equal_mean",
    "ecg_hrv_mean_nn_w300_equal_mean",
    "ecg_hrv_sdnn_w300_equal_mean",
    "ecg_hrv_rmssd_w300_equal_mean",
)
HRV_A14_SCALER_VERSION = "hrv_a14_train_fold_scaler_v1"
HRV_A14_FEATURE_VERSION = "hrv_night_candidate_a14_p1"
CAISR20_HRV14_SCALER_VERSION = "caisr20_hrv14_train_fold_scaler_v1"
HRV3_INDICES = (4, 10, 11)
HRV3_FEATURE_NAMES = tuple(HRV_A14_FEATURE_NAMES[index] for index in HRV3_INDICES)
CAISR20_HRV3_SCALER_VERSION = "caisr20_hrv3_h05_h11_h12_train_fold_scaler_v1"

# Stage-HRV-S2 的正式原始特征。S01 保留原始 HF relative，S02 保留原始
# HF absolute (ms²)；S02 的唯一非线性变换只在训练折 scaler 中执行一次。
STAGE_HRV_S2_FEATURE_NAMES = (
    "S01_N2_HF_relative",
    "S02_N3_HF_absolute_ms2",
)
STAGE_HRV_S2_FEATURE_VERSION = "stage_hrv_s2_v1"
STAGE_HRV_S2_SCALER_VERSION = "stage_hrv_s2_n2rel_n3hf_log1p_train_fold_v1"

EEG18_FEATURE_NAMES = (
    "eeg_n3_frontal_so_log_power",
    "eeg_n3_frontal_delta_log_power",
    "eeg_n2_frontal_swa_log_power",
    "eeg_n3_frontal_so_delta_log_ratio",
    "eeg_n3_frontal_slow_fast_log_ratio",
    "eeg_n2_central_sigma_relative_power",
    "eeg_n2_frontal_slow_sigma_relative_power",
    "eeg_n2_central_fast_sigma_relative_power",
    "eeg_n2_occipital_fast_sigma_relative_power",
    "eeg_rem_occipital_slowing_log_ratio",
    "eeg_rem_central_slowing_log_ratio",
    "eeg_rem_frontal_slowing_log_ratio",
    "eeg_rem_occipital_theta_alpha_log_ratio",
    "eeg_n3_frontal_occipital_swa_log_gradient",
    "eeg_n2_frontal_occipital_fast_sigma_log_gradient",
    "eeg_rem_occipital_frontal_slowing_difference",
    "eeg_n3_frontal_swa_left_right_log_asymmetry",
    "eeg_n2_central_sigma_left_right_log_asymmetry",
)

EEG_SCALER_VERSION = "eeg18_identity_train_fold_scaler_v1"
FUSION_SCALER_VERSION = "caisr20_eeg18_train_fold_scaler_v1"
EEG_SUBSET_SCALER_VERSION = "caisr20_eeg_subset_train_fold_scaler_v1"
B1_EEG_FEATURE_NAMES = tuple(EEG18_FEATURE_NAMES[5:9] + EEG18_FEATURE_NAMES[13:18])
SINGLE_FEATURE_ADDITION_EEG_NAMES = tuple(
    B1_EEG_FEATURE_NAMES + (EEG18_FEATURE_NAMES[index],)
    for index in (0, 1, 2, 3, 4, 9, 10, 11, 12)
)
REM_COMBINATION_EEG_NAMES = tuple(
    B1_EEG_FEATURE_NAMES + tuple(EEG18_FEATURE_NAMES[index] for index in indices)
    for indices in (
        (9, 10), (9, 11), (9, 12), (10, 11), (10, 12), (11, 12),
        (9, 10, 11), (9, 10, 12), (9, 11, 12), (10, 11, 12),
        (9, 10, 11, 12),
    )
)


def _make_rules() -> dict[str, dict[str, Any]]:
    rules: dict[str, dict[str, Any]] = {}
    for name in CAISR20_FEATURE_NAMES:
        rules[name] = {
            "unit_transform": "identity",
            "nonlinear": "identity",
            "robust": False,
        }
    for index in (0, 2, 3, 4):
        rules[CAISR20_FEATURE_NAMES[index]]["unit_transform"] = "sec_to_hour"
        rules[CAISR20_FEATURE_NAMES[index]]["robust"] = True
    for index in (14, 17):
        rules[CAISR20_FEATURE_NAMES[index]]["unit_transform"] = "sec_to_minute"
        rules[CAISR20_FEATURE_NAMES[index]]["nonlinear"] = "log1p"
        rules[CAISR20_FEATURE_NAMES[index]]["robust"] = True
    for index in (9, 12, 15, 18, 19):
        rules[CAISR20_FEATURE_NAMES[index]]["nonlinear"] = "log1p"
        rules[CAISR20_FEATURE_NAMES[index]]["robust"] = True
    return rules


DEFAULT_RULES = _make_rules()


def _identity_rules(feature_names: Sequence[str]) -> dict[str, dict[str, Any]]:
    return {
        name: {"unit_transform": "identity", "nonlinear": "identity", "robust": False}
        for name in feature_names
    }


def rules_for_feature_names(
    feature_names: Sequence[str],
) -> tuple[dict[str, dict[str, Any]], str]:
    """Return the fixed rule set for one approved experiment layout."""

    names = list(feature_names)
    if names == list(HRV_A14_FEATURE_NAMES):
        rules = _identity_rules(HRV_A14_FEATURE_NAMES)
        rules[HRV_A14_FEATURE_NAMES[5]]["nonlinear"] = "log1p"
        rules[HRV_A14_FEATURE_NAMES[6]]["nonlinear"] = "log1p"
        return rules, HRV_A14_SCALER_VERSION
    if names == list(CAISR20_FEATURE_NAMES):
        return {name: dict(rule) for name, rule in DEFAULT_RULES.items()}, SCALER_VERSION
    if names == list(EEG18_FEATURE_NAMES):
        return _identity_rules(EEG18_FEATURE_NAMES), EEG_SCALER_VERSION
    if names == list(CAISR20_FEATURE_NAMES + EEG18_FEATURE_NAMES):
        rules = {name: dict(rule) for name, rule in DEFAULT_RULES.items()}
        rules.update(_identity_rules(EEG18_FEATURE_NAMES))
        return rules, FUSION_SCALER_VERSION
    if names == list(CAISR20_FEATURE_NAMES + HRV_A14_FEATURE_NAMES):
        rules = {name: dict(rule) for name, rule in DEFAULT_RULES.items()}
        hrv_rules = _identity_rules(HRV_A14_FEATURE_NAMES)
        hrv_rules[HRV_A14_FEATURE_NAMES[5]]["nonlinear"] = "log1p"
        hrv_rules[HRV_A14_FEATURE_NAMES[6]]["nonlinear"] = "log1p"
        rules.update(hrv_rules)
        return rules, CAISR20_HRV14_SCALER_VERSION
    if names == list(CAISR20_FEATURE_NAMES + HRV3_FEATURE_NAMES):
        rules = {name: dict(rule) for name, rule in DEFAULT_RULES.items()}
        rules.update(_identity_rules(HRV3_FEATURE_NAMES))
        return rules, CAISR20_HRV3_SCALER_VERSION

    # P3 的六个固定布局：CAISR20 + B1/D7 EEG 子集，后接 S01、S02
    # 的一个或两个维度。保持组件顺序和特征顺序显式，避免把 S02
    # 误当作已变换的诊断值或重复 log1p。
    allowed_stage_hrv_layouts = {
        tuple(CAISR20_FEATURE_NAMES + B1_EEG_FEATURE_NAMES + (STAGE_HRV_S2_FEATURE_NAMES[0],)),
        tuple(CAISR20_FEATURE_NAMES + B1_EEG_FEATURE_NAMES + (STAGE_HRV_S2_FEATURE_NAMES[1],)),
        tuple(CAISR20_FEATURE_NAMES + B1_EEG_FEATURE_NAMES + STAGE_HRV_S2_FEATURE_NAMES),
        tuple(CAISR20_FEATURE_NAMES + B1_EEG_FEATURE_NAMES + EEG18_FEATURE_NAMES[9:12] + (STAGE_HRV_S2_FEATURE_NAMES[0],)),
        tuple(CAISR20_FEATURE_NAMES + B1_EEG_FEATURE_NAMES + EEG18_FEATURE_NAMES[9:12] + (STAGE_HRV_S2_FEATURE_NAMES[1],)),
        tuple(CAISR20_FEATURE_NAMES + B1_EEG_FEATURE_NAMES + EEG18_FEATURE_NAMES[9:12] + STAGE_HRV_S2_FEATURE_NAMES),
    }
    if tuple(names) in allowed_stage_hrv_layouts:
        rules = {name: dict(rule) for name, rule in DEFAULT_RULES.items()}
        suffix = names[len(CAISR20_FEATURE_NAMES):]
        for name in suffix:
            if name in EEG18_FEATURE_NAMES or name == STAGE_HRV_S2_FEATURE_NAMES[0]:
                rules[name] = {"unit_transform": "identity", "nonlinear": "identity", "robust": False}
            elif name == STAGE_HRV_S2_FEATURE_NAMES[1]:
                rules[name] = {"unit_transform": "identity", "nonlinear": "log1p", "robust": False}
        return rules, STAGE_HRV_S2_SCALER_VERSION
    caisr_prefix = list(CAISR20_FEATURE_NAMES)
    eeg_suffix = names[len(caisr_prefix):] if names[:len(caisr_prefix)] == caisr_prefix else []
    if names[:len(caisr_prefix)] == caisr_prefix and eeg_suffix:
        eeg_order = {name: index for index, name in enumerate(EEG18_FEATURE_NAMES)}
        if all(name in eeg_order for name in eeg_suffix) and (
            eeg_suffix == sorted(eeg_suffix, key=eeg_order.get)
            or tuple(eeg_suffix) in SINGLE_FEATURE_ADDITION_EEG_NAMES
            or tuple(eeg_suffix) in REM_COMBINATION_EEG_NAMES
        ):
            rules = {name: dict(rule) for name, rule in DEFAULT_RULES.items()}
            rules.update(_identity_rules(eeg_suffix))
            return rules, EEG_SUBSET_SCALER_VERSION
    raise ValueError("feature names/order must match an approved CAISR20, EEG18, HRV14, fusion, or subset layout")


def build_feature_scaler(feature_names: Sequence[str]) -> "FeatureScaler":
    rules, version = rules_for_feature_names(feature_names)
    return FeatureScaler(feature_names=feature_names, rules=rules, version=version)


class FeatureScaler:
    """按固定 CAISR20 顺序执行 train-fold-only 的缺失填补和标准化。

    ``valid_mask=False`` 与非有限输入都被视为缺失；有限的零值仍是有效输入。
    所有统计量只从 ``fit`` 收到的训练折计算，测试折只经过 ``transform``。
    """

    def __init__(
        self,
        feature_names: Sequence[str] | None = None,
        rules: Mapping[str, Mapping[str, Any]] | None = None,
        version: str | None = None,
    ) -> None:
        names = list(CAISR20_FEATURE_NAMES if feature_names is None else feature_names)
        default_rules, default_version = rules_for_feature_names(names)
        self.feature_names = names
        self.rules = {
            name: dict((rules or default_rules)[name]) for name in self.feature_names
        }
        self.version = default_version if version is None else str(version)
        self.fitted = False

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    def _validate_inputs(
        self, values: np.ndarray, valid_mask: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray]:
        array = np.asarray(values, dtype=float)
        if array.ndim != 2 or array.shape[1] != self.n_features:
            raise ValueError(
                f"expected a 2-D array with {self.n_features} features, got {array.shape}"
            )
        if valid_mask is None:
            mask = np.isfinite(array)
        else:
            mask = np.asarray(valid_mask, dtype=bool)
            if mask.shape != array.shape:
                raise ValueError("valid_mask must have the same shape as values")
        masked = np.array(array, dtype=float, copy=True)
        masked[~mask] = np.nan
        return masked, mask

    def _apply_rules(self, values: np.ndarray) -> np.ndarray:
        transformed = np.full_like(values, np.nan, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            for index, name in enumerate(self.feature_names):
                column = values[:, index]
                rule = self.rules[name]
                if rule["unit_transform"] == "sec_to_hour":
                    column = column / 3600.0
                elif rule["unit_transform"] == "sec_to_minute":
                    column = column / 60.0
                elif rule["unit_transform"] != "identity":
                    raise ValueError(f"unknown unit transform: {rule['unit_transform']}")
                if rule["nonlinear"] == "log1p":
                    column = np.log1p(column)
                elif rule["nonlinear"] != "identity":
                    raise ValueError(f"unknown nonlinear transform: {rule['nonlinear']}")
                transformed[:, index] = column
        return transformed

    def fit(
        self, values: np.ndarray, valid_mask: np.ndarray | None = None
    ) -> "FeatureScaler":
        masked, _ = self._validate_inputs(values, valid_mask)
        transformed = self._apply_rules(masked)
        self._transformed_train = transformed.copy()
        self.robust_center = np.zeros(self.n_features, dtype=float)
        self.robust_scale = np.ones(self.n_features, dtype=float)
        robust_values = transformed.copy()
        self.all_missing_fallback = [False] * self.n_features

        for index, name in enumerate(self.feature_names):
            finite = np.isfinite(transformed[:, index])
            if not np.any(finite):
                self.all_missing_fallback[index] = True
                continue
            if self.rules[name]["robust"]:
                present = transformed[finite, index]
                center = float(np.median(present))
                q25, q75 = np.percentile(present, [25.0, 75.0])
                scale = float(q75 - q25)
                self.robust_center[index] = center
                self.robust_scale[index] = scale if np.isfinite(scale) and scale > 0 else 1.0
                robust_values[finite, index] = (
                    present - self.robust_center[index]
                ) / self.robust_scale[index]

        self.imputer_median = np.zeros(self.n_features, dtype=float)
        filled = robust_values.copy()
        for index in range(self.n_features):
            finite = np.isfinite(robust_values[:, index])
            if np.any(finite):
                self.imputer_median[index] = float(np.median(robust_values[finite, index]))
            else:
                self.all_missing_fallback[index] = True
            filled[~finite, index] = self.imputer_median[index]

        standard = StandardScaler().fit(filled)
        self.standard_mean = standard.mean_.astype(float)
        self.standard_scale = standard.scale_.astype(float)
        self.standard_scale[~np.isfinite(self.standard_scale) | (self.standard_scale == 0)] = 1.0
        self.fitted = True
        return self

    def _transform_to_filled(
        self, values: np.ndarray, valid_mask: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self.fitted:
            raise RuntimeError("FeatureScaler must be fitted before transform")
        masked, supplied_mask = self._validate_inputs(values, valid_mask)
        transformed = self._apply_rules(masked)
        robust = transformed.copy()
        for index, name in enumerate(self.feature_names):
            if self.rules[name]["robust"]:
                finite = np.isfinite(robust[:, index])
                robust[finite, index] = (
                    robust[finite, index] - self.robust_center[index]
                ) / self.robust_scale[index]
        missing = ~supplied_mask | ~np.isfinite(robust)
        filled = robust.copy()
        for index in range(self.n_features):
            filled[missing[:, index], index] = self.imputer_median[index]
        return filled, missing

    def transform(
        self,
        values: np.ndarray,
        valid_mask: np.ndarray | None = None,
        *,
        return_diagnostics: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
        filled, missing = self._transform_to_filled(values, valid_mask)
        output = (filled - self.standard_mean) / self.standard_scale
        if not return_diagnostics:
            return output
        return output, {
            "missing_mask": missing,
            "all_missing_fallback": list(self.all_missing_fallback),
        }

    def fit_transform(
        self, values: np.ndarray, valid_mask: np.ndarray | None = None
    ) -> np.ndarray:
        return self.fit(values, valid_mask).transform(values, valid_mask)  # type: ignore[return-value]

    def explain(
        self, values: np.ndarray, valid_mask: np.ndarray | None = None
    ) -> dict[str, np.ndarray]:
        """返回单条输入经过各阶段的数值，仅用于显式诊断，不重新拟合。"""

        if not self.fitted:
            raise RuntimeError("FeatureScaler must be fitted before explain")
        raw = np.asarray(values, dtype=float).reshape(1, -1)
        mask_array = None if valid_mask is None else np.asarray(valid_mask, dtype=bool).reshape(1, -1)
        masked, supplied_mask = self._validate_inputs(raw, mask_array)
        unit_transformed = self._apply_rules(masked)
        robust = unit_transformed.copy()
        for index, name in enumerate(self.feature_names):
            if self.rules[name]["robust"]:
                finite = np.isfinite(robust[:, index])
                robust[finite, index] = (
                    robust[finite, index] - self.robust_center[index]
                ) / self.robust_scale[index]
        filled, missing = self._transform_to_filled(raw, mask_array)
        standardized = (filled - self.standard_mean) / self.standard_scale
        return {
            "raw": raw[0],
            "valid_mask": supplied_mask[0],
            "unit_transformed": unit_transformed[0],
            "robust_transformed": robust[0],
            "imputed": filled[0],
            "standardized": standardized[0],
            "missing_mask": missing[0],
        }

    def to_dict(self) -> dict[str, Any]:
        if not self.fitted:
            raise RuntimeError("cannot serialize an unfitted FeatureScaler")
        return {
            "version": self.version,
            "feature_names": list(self.feature_names),
            "rules": self.rules,
            "robust_center": self.robust_center.tolist(),
            "robust_scale": self.robust_scale.tolist(),
            "imputer_median": self.imputer_median.tolist(),
            "standard_mean": self.standard_mean.tolist(),
            "standard_scale": self.standard_scale.tolist(),
            "all_missing_fallback": list(self.all_missing_fallback),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FeatureScaler":
        scaler = cls(
            feature_names=payload["feature_names"],
            rules=payload["rules"],
            version=str(payload.get("version", SCALER_VERSION)),
        )
        for field in ("robust_center", "robust_scale", "imputer_median", "standard_mean", "standard_scale"):
            setattr(scaler, field, np.asarray(payload[field], dtype=float))
        scaler.all_missing_fallback = [bool(x) for x in payload["all_missing_fallback"]]
        scaler.fitted = True
        return scaler

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "FeatureScaler":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)


def _load_serialized_scaler(path: str | Path) -> FeatureScaler:
    import joblib

    payload = joblib.load(path)
    if isinstance(payload, dict) and "preprocessor" in payload:
        payload = payload["preprocessor"]
    # 直接执行 `python feature_scaling.py` 时，本文件类名为 `__main__.FeatureScaler`；
    # joblib 中的同一类会以 `feature_scaling.FeatureScaler` 反序列化，因此同时接受
    # 这两个模块身份，但仍检查必要的拟合接口和状态。
    if not isinstance(payload, FeatureScaler) and not (
        payload.__class__.__name__ == "FeatureScaler"
        and hasattr(payload, "transform")
        and bool(getattr(payload, "fitted", False))
    ):
        raise TypeError(f"unsupported preprocessor file: {path}")
    return payload


def _load_feature_row(feature_path: str | Path, record_id: str) -> tuple[np.ndarray, np.ndarray]:
    import json as _json
    import pandas as pd

    frame = pd.read_csv(feature_path, sep="\t", dtype=str, keep_default_na=False)
    rows = frame.loc[frame["record_id"] == record_id]
    if len(rows) != 1:
        raise KeyError(f"expected one frozen feature row for {record_id}, found {len(rows)}")
    row = rows.iloc[0]
    values = _json.loads(row["values"])
    valid = _json.loads(row["valid_mask"])
    return (
        np.asarray([np.nan if x is None else float(x) for x in values], dtype=float),
        np.asarray(valid, dtype=bool),
    )


def _print_vector(label: str, values: np.ndarray, names: Sequence[str]) -> None:
    print(label)
    for name, value in zip(names, values):
        print(f"  {name}: {value!r}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Inspect a fitted CAISR20 scaler")
    parser.add_argument("--preprocessor", required=True, help="preprocessing.joblib or model.sav")
    parser.add_argument("--features", default="feature_extraction/features/caisr/caisr.tsv")
    parser.add_argument("--record-id")
    parser.add_argument("--distribution", action="store_true")
    parser.add_argument(
        "--demographics",
        default="/database2/physionet2026_kaggle/data2/demographics.csv",
    )
    args = parser.parse_args(argv)
    if not args.record_id and not args.distribution:
        parser.error("provide --record-id or --distribution")
    scaler = _load_serialized_scaler(args.preprocessor)
    if args.record_id:
        values, valid = _load_feature_row(args.features, args.record_id)
        explained = scaler.explain(values, valid)
        print(f"record_id: {args.record_id}")
        for label, key in (
            ("raw", "raw"),
            ("valid_mask", "valid_mask"),
            ("unit_transformed", "unit_transformed"),
            ("robust_transformed", "robust_transformed"),
            ("imputed", "imputed"),
            ("standardized", "standardized"),
        ):
            _print_vector(label, explained[key], scaler.feature_names)
    if args.distribution:
        from train_lr import load_caisr_dataset

        dataset = load_caisr_dataset(args.features, args.demographics)
        transformed = scaler.transform(dataset.values, dataset.valid_mask)
        print(f"distribution_records: {len(dataset.records)}")
        for index, name in enumerate(scaler.feature_names):
            column = transformed[:, index]
            print(
                f"{name}\tfinite={np.isfinite(column).sum()}"
                f"\tmin={np.min(column):.9g}\tmedian={np.median(column):.9g}"
                f"\tmax={np.max(column):.9g}"
            )


if __name__ == "__main__":
    main()

