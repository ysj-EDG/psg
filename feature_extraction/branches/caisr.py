"""CAISR20 夜间级特征分支。

本模块只从 canonical HDF5 的 ``annotations/caisr`` 读取已经对齐的标注，计算
固定顺序的 CAISR20 `(20,)` night-level 特征。它不读取 EDF 或 waveform，不写
cache，也不执行滤波、重采样、归一化、QC 或模型/训练逻辑。自动 CAISR、人工
标注和 unavailable 通过 provenance 与 channel metadata 区分；人工标注没有的
概率不会被伪造为 0。
"""

from __future__ import annotations

import argparse
import math
from functools import lru_cache
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

if __package__ in (None, ""):
    _ROOT = Path(__file__).resolve().parents[2]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from feature_extraction.feature_interfaces import BranchNotImplementedError
    from feature_extraction.feature_schema import BranchResult, FeatureBlock, InvalidReason, TimeScale
    from feature_extraction.sample_loader import Manifest, SampleContext, _text
else:
    from ..feature_interfaces import BranchNotImplementedError
    from ..feature_schema import BranchResult, FeatureBlock, InvalidReason, TimeScale
    from ..sample_loader import Manifest, SampleContext, _text


BRANCH_NAME = "caisr"
PLANNED_TIME_SCALES = ("night",)
FEATURE_VERSION = "canonical_v1"
EPOCH_SEC = 30.0
AROUSAL_FS = 2.0
EVENT_FS = 1.0

CAISR20_FEATURE_NAMES = (
    "caisr_sleep_tst_sec", "caisr_sleep_se", "caisr_sleep_sol_sec", "caisr_sleep_rem_latency_sec",  #睡眠总时长TST，TST/记录总时长，从起始到首次睡眠时长，首次睡眠到首次REM时长
    "caisr_sleep_waso_sec", "caisr_sleep_n1_pct", "caisr_sleep_n2_pct", "caisr_sleep_n3_pct",   #入睡后的Wake总时长，N1占比，N2占比，N3占比
    "caisr_sleep_rem_pct", "caisr_sleep_transition_rate", "caisr_sleep_short_bout_ratio",   #REM占比，睡眠阶段转换率，短睡眠段占比
    "caisr_sleep_mean_stage_entropy", "caisr_arousal_arousal_index", "caisr_arousal_arousal_burden_ratio",  #五类睡眠概率平均信息熵，觉醒指数，觉醒负担比率
    "caisr_arousal_mean_arousal_duration", "caisr_respiratory_respiratory_event_index", #觉醒平均持续时间，呼吸事件指数
    "caisr_respiratory_respiratory_burden_ratio", "caisr_respiratory_mean_resp_duration",   #呼吸负担比率，呼吸平均持续时间
    "caisr_limb_limb_movement_index", "caisr_limb_plmi",    #肢体运动指数，周期性肢体运动指数
)
CAISR20_FEATURE_UNITS = (
    "sec", "ratio", "sec", "sec", "sec", "ratio", "ratio", "ratio", "ratio", "per_hour",
    "ratio", "ratio", "per_hour", "ratio", "sec", "per_hour", "ratio", "sec", "per_hour", "per_hour",
)
CAISR20_DIMENSION = 20
assert len(CAISR20_FEATURE_NAMES) == CAISR20_DIMENSION

_STAGE_CODES = (1, 2, 3, 4, 5)
_SLEEP_CODES = (1, 2, 3, 4)
_PROBABILITY_LABELS = {
    "caisr_prob_n3", "caisr_prob_n2", "caisr_prob_n1", "caisr_prob_r", "caisr_prob_w",
}
_CHANNEL_ALIASES = {
    "stage": {"stage_caisr", "stage_expert"},
    "stage_probability": _PROBABILITY_LABELS,
    "arousal": {"arousal_caisr", "arousal_expert"},
    "respiratory": {"resp_caisr", "resp_expert"},
    "limb": {"limb_caisr", "limb_expert"},
}


def _as_1d(value: Any, dtype: Any | None = None) -> np.ndarray:
    array = np.asarray(value if value is not None else [], dtype=dtype)
    return array.reshape(-1)


def _segment_binary(values: Any) -> list[tuple[int, int]]:
    """返回正值连续段；nonfinite、负值和 0 都会打断连续段。"""
    array = _as_1d(values, float)
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for index, item in enumerate(array):
        positive = bool(np.isfinite(item) and item > 0)
        if positive and start is None:
            start = index
        elif not positive and start is not None:
            segments.append((start, index))
            start = None
    if start is not None:
        segments.append((start, len(array)))
    return segments


def _segment_multiclass(values: Any, allowed_codes: tuple[int, ...]) -> list[tuple[int, int, int]]:
    """按 categorical code 拆分正事件段；invalid/0 和 code 变化均断开。"""
    array = _as_1d(values, float)
    allowed = set(allowed_codes)
    segments: list[tuple[int, int, int]] = []
    start: int | None = None
    current: int | None = None
    for index, item in enumerate(array):
        code = int(item) if np.isfinite(item) and float(item).is_integer() else None
        if code not in allowed:
            if start is not None and current is not None:
                segments.append((start, index, current))
            start, current = None, None
            continue
        if start is None:
            start, current = index, code
        elif code != current:
            segments.append((start, index, int(current)))
            start, current = index, code
    if start is not None and current is not None:
        segments.append((start, len(array), int(current)))
    return segments


def _sleep_bouts(stage: np.ndarray) -> list[tuple[int, int]]:
    """返回连续同一 sleep stage 的 bout；Wake、invalid 和 stage 变化均断开。"""
    values = _as_1d(stage, np.int64)
    bouts: list[tuple[int, int]] = []
    start: int | None = None
    current: int | None = None
    for index, value in enumerate(values):
        code = int(value) if int(value) in _SLEEP_CODES else None
        if code is None:
            if start is not None:
                bouts.append((start, index))
            start, current = None, None
        elif start is None:
            start, current = index, code
        elif code != current:
            bouts.append((start, index))
            start, current = index, code
    if start is not None:
        bouts.append((start, len(values)))
    return bouts


def _stage_probability_entropy(probability: np.ndarray, valid_mask: np.ndarray, availability: bool) -> tuple[float, int]:
    if not availability:
        return float("nan"), 0
    values = np.asarray(probability, dtype=np.float64)
    mask = _as_1d(valid_mask, bool)
    if values.ndim != 2 or values.shape[1] != 5:
        return float("nan"), 0
    usable = mask[: len(values)]
    if len(usable) < len(values):
        values = values[: len(usable)]
    usable = usable & np.isfinite(values).all(axis=1) & (values >= 0).all(axis=1)
    if not np.any(usable):
        return float("nan"), 0
    rows = values[usable]
    positive = np.where(rows > 0, rows, 1.0)
    entropy = -np.sum(np.where(rows > 0, rows * np.log(positive), 0.0), axis=1) / math.log(5.0)
    return float(np.mean(entropy)), int(len(rows))


def _empty_outputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.full(CAISR20_DIMENSION, np.nan, dtype=np.float32)
    valid = np.zeros(CAISR20_DIMENSION, dtype=bool)
    reasons = np.full(CAISR20_DIMENSION, InvalidReason.MISSING_INPUT.value, dtype="U64")
    return values, valid, reasons


def _set_feature(values: np.ndarray, valid: np.ndarray, reasons: np.ndarray, index: int, value: float, reason: str | None = None) -> None:
    if reason is None and np.isfinite(value):
        values[index] = np.float32(value)
        valid[index] = True
        reasons[index] = ""
    else:
        values[index] = np.nan
        valid[index] = False
        reasons[index] = reason or InvalidReason.INVALID_VALUE.value


def _compute_features(
    stage: Any,
    stage_probability: Any,
    stage_probability_valid: Any,
    arousal: Any,
    respiratory: Any,
    limb: Any,
    *,
    availability: Mapping[str, bool] | None = None,
    epoch_duration_sec: float = EPOCH_SEC,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """对已读入的 canonical annotation 数组计算 CAISR20。"""
    available = {"stage": True, "stage_probability": True, "arousal": True, "respiratory": True, "limb": True}
    if availability is not None:
        available.update({key: bool(value) for key, value in availability.items()})
    stage_array = _as_1d(stage, np.int64)
    probabilities = np.asarray(stage_probability, dtype=np.float32)
    if probabilities.size == 0:
        probabilities = np.empty((0, 5), dtype=np.float32)
    probability_valid = _as_1d(stage_probability_valid, bool)
    values, valid, reasons = _empty_outputs()
    diagnostics: dict[str, Any] = {"tst_sec": float("nan"), "sleep_bout_lengths_epochs": []}

    stage_valid = available["stage"] & np.isin(stage_array, _STAGE_CODES)
    valid_stage_count = int(np.count_nonzero(stage_valid))
    sleep_mask = stage_valid & np.isin(stage_array, _SLEEP_CODES)
    sleep_count = int(np.count_nonzero(sleep_mask))
    record_duration_sec = float(len(stage_array) * epoch_duration_sec)
    diagnostics.update({"valid_stage_epochs": valid_stage_count, "sleep_epochs": sleep_count, "record_duration_sec": record_duration_sec})

    if available["stage"] and sleep_count > 0:
        tst_sec = sleep_count * epoch_duration_sec
        first_sleep = int(np.flatnonzero(sleep_mask)[0])
        rem_indices = np.flatnonzero(stage_valid & (stage_array == 4))
        rem_after = rem_indices[rem_indices >= first_sleep]
        waso_sec = float(np.count_nonzero(stage_valid[first_sleep:] & (stage_array[first_sleep:] == 5)) * epoch_duration_sec)
        transition_count = int(np.count_nonzero((stage_array[1:] != stage_array[:-1]) & stage_valid[1:] & stage_valid[:-1])) if len(stage_array) > 1 else 0
        bouts = _sleep_bouts(np.where(stage_valid, stage_array, -1))
        bout_lengths = [end - start for start, end in bouts]
        diagnostics["sleep_bout_lengths_epochs"] = bout_lengths
        diagnostics["tst_sec"] = float(tst_sec)
        _set_feature(values, valid, reasons, 0, tst_sec)
        _set_feature(values, valid, reasons, 1, tst_sec / record_duration_sec if record_duration_sec > 0 else float("nan"))
        _set_feature(values, valid, reasons, 2, first_sleep * epoch_duration_sec)
        _set_feature(values, valid, reasons, 3, (int(rem_after[0]) - first_sleep) * epoch_duration_sec if len(rem_after) else float("nan"), None if len(rem_after) else InvalidReason.COVERAGE_INSUFFICIENT.value)
        _set_feature(values, valid, reasons, 4, waso_sec)
        denominator = float(tst_sec)
        for index, code in zip((5, 6, 7, 8), (3, 2, 1, 4)):
            _set_feature(values, valid, reasons, index, float(np.count_nonzero(stage_valid & (stage_array == code))) * epoch_duration_sec / denominator)
        _set_feature(values, valid, reasons, 9, transition_count / (tst_sec / 3600.0))
        _set_feature(values, valid, reasons, 10, float(np.mean(np.asarray(bout_lengths) < 3)) if bout_lengths else float("nan"), None if bout_lengths else InvalidReason.COVERAGE_INSUFFICIENT.value)
    else:
        reason = InvalidReason.CAISR_UNAVAILABLE.value if not available["stage"] else InvalidReason.COVERAGE_INSUFFICIENT.value
        for index in range(11):
            _set_feature(values, valid, reasons, index, float("nan"), reason)

    entropy, rows_used = _stage_probability_entropy(probabilities, probability_valid, available["stage_probability"])
    diagnostics["stage_probability_rows_used"] = rows_used
    _set_feature(values, valid, reasons, 11, entropy, None if np.isfinite(entropy) else (InvalidReason.CAISR_UNAVAILABLE.value if not available["stage_probability"] else InvalidReason.COVERAGE_INSUFFICIENT.value))

    tst_value = diagnostics["tst_sec"]
    tst_hours = float(tst_value) / 3600.0 if np.isfinite(tst_value) and tst_value > 0 else 0.0
    arousal_segments = _segment_binary(np.asarray(arousal).reshape(-1)) if available["arousal"] else []
    respiratory_segments = _segment_multiclass(np.asarray(respiratory).reshape(-1), (1, 2, 3, 4, 5)) if available["respiratory"] else []
    limb_segments = _segment_multiclass(np.asarray(limb).reshape(-1), (1, 2)) if available["limb"] else []
    diagnostics.update({
        "arousal_segment_count": len(arousal_segments),
        "respiratory_segment_count": len(respiratory_segments),
        "limb_periodic_segment_count": sum(code == 2 for _, _, code in limb_segments),
    })

    def _event_metrics(indexes: tuple[int, int, int], segments: list[Any], fs: float, source_available: bool) -> None:
        index, burden, mean_duration = indexes
        if not source_available:
            for feature_index in indexes:
                _set_feature(values, valid, reasons, feature_index, float("nan"), InvalidReason.CAISR_UNAVAILABLE.value)
            return
        if tst_hours <= 0:
            for feature_index in indexes:
                _set_feature(values, valid, reasons, feature_index, float("nan"), InvalidReason.COVERAGE_INSUFFICIENT.value)
            return
        durations = [(segment[1] - segment[0]) / fs for segment in segments]
        _set_feature(values, valid, reasons, index, len(segments) / tst_hours)
        _set_feature(values, valid, reasons, burden, sum(durations) / float(tst_value))
        _set_feature(values, valid, reasons, mean_duration, float(np.mean(durations)) if durations else float("nan"), None if durations else InvalidReason.COVERAGE_INSUFFICIENT.value)

    _event_metrics((12, 13, 14), arousal_segments, AROUSAL_FS, available["arousal"])
    _event_metrics((15, 16, 17), respiratory_segments, EVENT_FS, available["respiratory"])
    if not available["limb"]:
        _set_feature(values, valid, reasons, 18, float("nan"), InvalidReason.CAISR_UNAVAILABLE.value)
        _set_feature(values, valid, reasons, 19, float("nan"), InvalidReason.CAISR_UNAVAILABLE.value)
    elif tst_hours <= 0:
        _set_feature(values, valid, reasons, 18, float("nan"), InvalidReason.COVERAGE_INSUFFICIENT.value)
        _set_feature(values, valid, reasons, 19, float("nan"), InvalidReason.COVERAGE_INSUFFICIENT.value)
    else:
        _set_feature(values, valid, reasons, 18, len(limb_segments) / tst_hours)
        _set_feature(values, valid, reasons, 19, sum(code == 2 for _, _, code in limb_segments) / tst_hours)

    return values.astype(np.float32, copy=False), valid, reasons, diagnostics


def _source_kind(context: Any) -> str:
    record_path = str(getattr(getattr(context, "record", None), "caisr_edf_path", "")).lower()
    if "human_annotations" in record_path or "expert" in record_path or "manual" in record_path:
        return "manual"
    if "algorithmic_annotations" in record_path or "caisr" in record_path:
        return "auto"
    try:
        source = _text(np.asarray(context.metadata("caisr", "annotation_source")).reshape(-1)[0]).lower()
    except (KeyError, IndexError, TypeError, ValueError):
        source = ""
    return "manual" if source == "manual" else ("auto" if source in {"auto", "caisr", "algorithmic"} else "unavailable")


def _source_availability(context: Any) -> dict[str, bool]:
    """依据 annotation provenance 与 channel_names 返回每类标注是否可用。"""
    source = _source_kind(context)
    try:
        labels = {name.lower() for name in context.channel_names("caisr")}
    except (KeyError, OSError, RuntimeError):
        labels = set()
    if source == "unavailable":
        return {key: False for key in ("stage", "stage_probability", "arousal", "respiratory", "limb")}
    result = {key: bool({alias.lower() for alias in aliases} & labels) for key, aliases in _CHANNEL_ALIASES.items()}
    result["stage_probability"] = source != "manual" and _PROBABILITY_LABELS.issubset(labels)
    return result


def extract(context: Any, config: Mapping[str, Any] | None = None) -> BranchResult:
    """从一条已打开的 SampleContext 提取 CAISR20；只读取 annotations 与 timebase。"""
    del config
    arrays = {name: np.asarray(context.caisr_dataset(name)[...]) for name in ("stage", "stage_probability", "stage_probability_valid", "arousal", "respiratory", "limb")}
    availability = _source_availability(context)
    values, valid, reasons, diagnostics = _compute_features(**arrays, availability=availability)
    timebase = context.timebase()
    diagnostics.update({
        "record_id": context.record.record_id,
        "site": context.record.site,
        "annotation_source": _source_kind(context),
        "source_path": context.record.caisr_edf_path,
        "epoch_count": int(context.epoch_count),
        "timebase_start_sec": float(timebase["epoch_start_sec"][0]) if len(timebase["epoch_start_sec"]) else 0.0,
        "timebase_end_sec": float(timebase["epoch_end_sec"][-1]) if len(timebase["epoch_end_sec"]) else 0.0,
        "availability": dict(availability),
    })
    block = FeatureBlock(
        branch_name=BRANCH_NAME,
        feature_set_name="CAISR20",
        time_scale=TimeScale.NIGHT,
        values=values,
        feature_names=CAISR20_FEATURE_NAMES,
        valid_mask=valid,
        invalid_reason=reasons,
        config={"epoch_duration_sec": EPOCH_SEC, "arousal_fs_hz": AROUSAL_FS, "event_fs_hz": EVENT_FS},
        version=FEATURE_VERSION,
    )
    return BranchResult(branch_name=BRANCH_NAME, blocks=(block,), metadata=diagnostics).validate()


def quality_gate(context: Any, config: Mapping[str, Any] | None = None):
    raise BranchNotImplementedError("CAISR-specific quality gating is not implemented")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--record-id")
    parser.add_argument("--list-features", action="store_true")
    parser.add_argument("--show-diagnostics", action="store_true")
    return parser


def _print_feature_list() -> None:
    print("index\tfeature_name\tunit")
    for index, (name, unit) in enumerate(zip(CAISR20_FEATURE_NAMES, CAISR20_FEATURE_UNITS)):
        print(f"{index}\t{name}\t{unit}")


@lru_cache(maxsize=4)
def _load_manifest(path: str) -> Manifest:
    return Manifest.from_tsv(Path(path))


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.list_features:
        _print_feature_list()
        return 0
    if args.manifest is None or not args.record_id:
        _parser().error("record extraction requires --manifest and --record-id")
    try:
        manifest = _load_manifest(str(args.manifest.resolve()))
        record = manifest.get(args.record_id)
        with SampleContext(record) as context:
            result = extract(context)
    except Exception as exc:
        print(f"EXTRACTION_ERROR: {type(exc).__name__}: {exc}")
        return 2
    block = result.blocks[0]
    print(f"record_id: {record.record_id}")
    print(f"site: {record.site}")
    print(f"annotation_source: {result.metadata.get('annotation_source')}")
    print("index\tfeature_name\tvalue\tvalid\tinvalid_reason")
    for index, name in enumerate(block.feature_names):
        value = "nan" if np.isnan(block.values[index]) else f"{float(block.values[index]):.8g}"
        print(f"{index}\t{name}\t{value}\t{bool(block.valid_mask[index])}\t{block.invalid_reason[index]}")
    print(f"valid_count: {int(block.valid_mask.sum())}")
    print(f"invalid_count: {int((~block.valid_mask).sum())}")
    if args.show_diagnostics:
        print("diagnostics:")
        for key, value in sorted(result.metadata.items()):
            print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# PYTHONNOUSERSITE=1 \
# /database/home/gaohaojie/.conda/envs/sleepfm_env/bin/python \
# -m feature_extraction.branches.caisr \
# --manifest cache_manifest.tsv \
# --record-id sub-I0002150026011_ses-2 \
# --show-diagnostics