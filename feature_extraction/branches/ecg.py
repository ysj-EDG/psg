"""ECG/HRV 特征分支。

正式 ``extract()`` 实现冻结的 Stage-HRV-S2 ECG-exact 纯阶段窗口，返回
S01/S02 两个整夜特征。文件内保留 A14 低层数学实现和显式兼容包装，供历史
A14 回放使用；兼容层不属于正式 S2 输出。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.signal import welch

from ..feature_interfaces import BranchResult
from ..feature_schema import FeatureBlock, TimeScale

BRANCH_NAME = "ecg"
FEATURE_VERSION = "hrv_night_candidate_a14_p1"
S2_FEATURE_VERSION = "stage_hrv_s2_v1"
S2_FEATURE_NAMES = ("S01_N2_HF_relative", "S02_N3_HF_absolute_ms2")
S2_FEATURE_SET_NAME = "stage_hrv_s2"
PLANNED_TIME_SCALES = (TimeScale.WINDOW.value, TimeScale.NIGHT.value)
ECG_FS = 200.0
WINDOW_SECONDS = 300.0
WINDOW_SAMPLES = int(ECG_FS * WINDOW_SECONDS)
SUBWINDOW_SECONDS = 5.0
A14_FEATURE_NAMES = (
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
A14_UNITS = ("ms", "ratio", "ratio", "ratio", "%", "ms^2", "ms^2", "ratio", "ratio", "ratio", "ratio", "ms", "ms", "ms")

# 这些是本轮明确冻结的工程初值，不是临床标准。
DEFAULT_P1_POLICY: dict[str, Any] = {
    "version": "hrv_a14_p1_policy_v1",
    "parameter_status": "engineering_pilot_setting_not_clinical_standard",
    "parameter_basis": "local canonical schema, prior P0R audit, current NeuroKit/scipy interfaces; no label or model-score tuning",
    "canonical_sampling_rate_hz": 200.0,
    "analysis_window_sec": 300.0,
    "analysis_step_sec": 300.0,
    "detection": {
        "method": "neurokit", "clean_method": "neurokit",
        "correction_method": "Kubios", "iterative": True,
        "correct_artifacts": False,
    },
    "continuity": {
        "mask": "finite AND NOT invalid_5s AND NOT flat_5s AND NOT extreme_5s",
        "min_continuous_block_sec": 30.0,
        "hard_gap_breaks_continuity": True,
        "window_context_sec": 0.0,
        "analysis_segment_rule": "longest_continuous_block_only",
        "analysis_segment_tie_break": "earliest_start",
    },
    "nn": {
        "min_rr_ms": 300.0, "max_rr_ms": 3000.0,
        "min_nn_count": 30, "min_adjacent_pairs": 29,
        "min_triples": 28, "allow_uncorrected_after_correction_failure": False,
    },
    "psd": {
        "interpolation_rate_hz": 4.0, "interpolation": "linear",
        "grid_endpoint": "include_first_and_last_nn_endpoint",
        "detrend": "constant", "window": "hann",
        "nperseg": 256, "noverlap": 128, "nfft": 512,
        "scaling": "density", "average": "mean", "onesided": True,
        "lf_hz": [0.04, 0.15], "hf_hz": [0.15, 0.40],
        "min_continuous_support_sec": 120.0,
    },
    "night": {
        "min_valid_windows": 2, "aggregation": "equal_mean_of_valid_nonoverlap_300s_windows",
    },
}

@dataclass(frozen=True)
class NNInterval:
    start_peak: int
    end_peak: int
    value_ms: float
    block_id: int
    ordinal: int

@dataclass
class HRVComputationResult:
    values: np.ndarray
    valid_mask: np.ndarray
    primary_reasons: np.ndarray
    all_failed_conditions: tuple[str, ...]
    diagnostics: dict[str, Any]

@dataclass
class ECGWindowHRVResult:
    values: np.ndarray
    valid_mask: np.ndarray
    primary_reasons: np.ndarray
    all_failed_conditions: tuple[str, ...]
    diagnostics: dict[str, Any]
    provenance: list[dict[str, Any]]

def _policy(policy: Mapping[str, Any] | None) -> dict[str, Any]:
    if policy is None:
        return DEFAULT_P1_POLICY
    result = dict(DEFAULT_P1_POLICY)
    for key, value in policy.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            nested = dict(result[key]); nested.update(value); result[key] = nested
        else:
            result[key] = value
    return result

def _unit_to_mV(unit: Any) -> tuple[str, float | None, str]:
    if isinstance(unit, (bytes, np.bytes_)):
        unit = bytes(unit).decode("utf-8", errors="replace")
    normalized = str(unit).strip().replace("μ", "µ").lower()
    if normalized == "v":
        return "V", 1000.0, "recognized"
    if normalized == "mv":
        return "mV", 1.0, "recognized"
    if normalized in {"uv", "µv"}:
        return "µV", 0.001, "recognized"
    return str(unit), None, "unknown"


def normalize_ecg_input(values: Sequence[float], stored_unit: Any) -> tuple[np.ndarray, dict[str, Any]]:
    """按 physical_dimension 明确转换为 NeuroKit 使用的 mV，且只转换一次。"""
    arr = np.asarray(values, dtype=np.float64)
    if isinstance(stored_unit, (bytes, np.bytes_)):
        stored_unit = bytes(stored_unit).decode("utf-8", errors="replace")
    label, scale, status = _unit_to_mV(stored_unit)
    if scale is None:
        return arr.copy(), {
            "stored_unit": str(stored_unit), "detector_input_unit": "unknown",
            "scale_factor": "NA", "conversion_applied": False,
            "unit_status": status, "unit_evidence": "metadata/ecg/physical_dimension",
        }
    return arr * scale, {
        "stored_unit": str(stored_unit), "detector_input_unit": "mV",
        "scale_factor": scale, "conversion_applied": bool(scale != 1.0),
        "unit_status": status, "unit_evidence": "metadata/ecg/physical_dimension",
        "canonical_unit_label": label,
    }


def audit_peak_integrity(peaks: Sequence[Any], sample_count: int) -> dict[str, Any]:
    raw = np.asarray(peaks).reshape(-1)
    numeric = np.asarray(raw, dtype=float) if raw.size else np.empty(0, dtype=float)
    finite = numeric[np.isfinite(numeric)]
    integer = finite[np.isclose(finite, np.rint(finite), rtol=0.0, atol=1e-6)]
    duplicate_count = int(finite.size - np.unique(finite).size)
    out_of_range_count = int(np.count_nonzero((finite < 0) | (finite >= int(sample_count))))
    normalized = np.rint(integer).astype(np.int64) if integer.size else np.empty(0, dtype=np.int64)
    normalized = np.unique(normalized)
    normalized = normalized[(normalized >= 0) & (normalized < int(sample_count))]
    return {
        "raw_values": numeric.tolist(),
        "nonfinite_count": int(numeric.size - finite.size),
        "noninteger_count": int(finite.size - integer.size),
        "duplicate_count": duplicate_count,
        "out_of_range_count": out_of_range_count,
        "ordered": bool(np.all(np.diff(finite) >= 0)) if finite.size > 1 else True,
        "raw_count": int(numeric.size),
        "normalized": normalized,
    }

def _runs(mask: np.ndarray, min_length: int = 1) -> list[tuple[int, int]]:
    x = np.asarray(mask, dtype=bool).reshape(-1)
    if x.size == 0:
        return []
    padded = np.concatenate(([False], x, [False]))
    starts = np.flatnonzero(padded[1:] & ~padded[:-1])
    ends = np.flatnonzero(~padded[1:] & padded[:-1])
    return [(int(a), int(b)) for a, b in zip(starts, ends) if b - a >= min_length]

def build_candidate_nn_audited(peaks: Sequence[int], *, fs: float, window_start_sample: int,
                               window_end_sample: int, block_id: Sequence[int] | int = 0,
                               policy: Mapping[str, Any] | None = None) -> tuple[list[NNInterval], dict[str, int]]:
    cfg = _policy(policy)["nn"]
    p = np.asarray(peaks, dtype=np.int64).reshape(-1)
    blocks = np.full(p.shape, int(block_id), dtype=int) if np.isscalar(block_id) else np.asarray(block_id, dtype=int).reshape(-1)
    if blocks.size != p.size:
        raise ValueError("block_id must have one value per peak")
    result: list[NNInterval] = []
    rejected: dict[str, int] = {}
    ordinal = 0
    def reject(reason: str) -> None:
        rejected[reason] = rejected.get(reason, 0) + 1
    for i in range(max(0, p.size - 1)):
        a, b = int(p[i]), int(p[i + 1])
        if blocks[i] != blocks[i + 1]:
            reject("block_boundary")
            continue
        if a < window_start_sample or b >= window_end_sample:
            reject("outside_window")
            continue
        if b <= a:
            reject("nonincreasing_peak")
            continue
        value_ms = (b - a) / float(fs) * 1000.0
        if not np.isfinite(value_ms):
            reject("rr_nonfinite")
            continue
        if value_ms < float(cfg["min_rr_ms"]):
            reject("rr_below_min")
            continue
        if value_ms > float(cfg["max_rr_ms"]):
            reject("rr_above_max")
            continue
        result.append(NNInterval(a, b, float(value_ms), int(blocks[i]), ordinal))
        ordinal += 1
    return result, rejected


def build_candidate_nn(peaks: Sequence[int], *, fs: float, window_start_sample: int,
                       window_end_sample: int, block_id: Sequence[int] | int = 0,
                       policy: Mapping[str, Any] | None = None) -> list[NNInterval]:
    return build_candidate_nn_audited(
        peaks, fs=fs, window_start_sample=window_start_sample,
        window_end_sample=window_end_sample, block_id=block_id, policy=policy,
    )[0]


def split_nn_chains(intervals: Sequence[NNInterval]) -> list[list[NNInterval]]:
    """按真实峰邻接关系拆链；被删除的RR不允许被隐式跨越。"""
    chains: list[list[NNInterval]] = []
    current: list[NNInterval] = []
    for interval in intervals:
        if not current:
            current = [interval]
            continue
        previous = current[-1]
        if interval.block_id == previous.block_id and interval.start_peak == previous.end_peak:
            current.append(interval)
        else:
            chains.append(current)
            current = [interval]
    if current:
        chains.append(current)
    return chains


def chain_support_sec(chain: Sequence[NNInterval], fs: float = ECG_FS) -> float:
    if not chain:
        return 0.0
    return float(chain[-1].end_peak - chain[0].start_peak) / float(fs)

def build_candidate_triples(intervals: Sequence[NNInterval]) -> list[tuple[NNInterval, NNInterval, NNInterval]]:
    triples = []
    for a, b, c in zip(intervals[:-2], intervals[1:-1], intervals[2:]):
        if a.block_id == b.block_id == c.block_id and a.end_peak == b.start_peak and b.end_peak == c.start_peak:
            triples.append((a, b, c))
    return triples

def _integrate_band(freq: np.ndarray, psd: np.ndarray, low: float, high: float) -> float:
    if low < freq[0] or high > freq[-1] or low >= high:
        return float("nan")
    interior = (freq > low) & (freq < high)
    f = np.concatenate(([low], freq[interior], [high]))
    p = np.interp(f, freq, psd)
    integrate = getattr(np, "trapezoid", np.trapz)
    return float(integrate(p, f))

def frequency_power_from_nn(intervals: Sequence[NNInterval], policy: Mapping[str, Any] | None = None) -> tuple[float, float, dict[str, Any]]:
    cfg = _policy(policy)["psd"]
    if len(intervals) < 3:
        return float("nan"), float("nan"), {"reason": "spectral_support_insufficient", "interpolation_points": 0}
    first = intervals[0]
    for previous, current in zip(intervals[:-1], intervals[1:]):
        if current.block_id != first.block_id or previous.block_id != current.block_id:
            return float("nan"), float("nan"), {"reason": "nn_continuity_broken", "interpolation_points": 0}
        if previous.end_peak != current.start_peak:
            return float("nan"), float("nan"), {"reason": "nn_continuity_broken", "interpolation_points": 0}
        if current.start_peak <= previous.start_peak or current.end_peak <= current.start_peak:
            return float("nan"), float("nan"), {"reason": "nn_continuity_broken", "interpolation_points": 0}
        observed_ms = (current.end_peak - current.start_peak) / ECG_FS * 1000.0
        if not np.isclose(observed_ms, current.value_ms, rtol=0.0, atol=1e-6):
            return float("nan"), float("nan"), {"reason": "nn_continuity_broken", "interpolation_points": 0}
    if first.end_peak <= first.start_peak:
        return float("nan"), float("nan"), {"reason": "nn_continuity_broken", "interpolation_points": 0}
    observed_first_ms = (first.end_peak - first.start_peak) / ECG_FS * 1000.0
    if not np.isclose(observed_first_ms, first.value_ms, rtol=0.0, atol=1e-6):
        return float("nan"), float("nan"), {"reason": "nn_continuity_broken", "interpolation_points": 0}
    times = np.asarray([x.end_peak / ECG_FS for x in intervals], dtype=float)
    values = np.asarray([x.value_ms for x in intervals], dtype=float)
    support_duration = chain_support_sec(intervals)
    tachogram_span = float(times[-1] - times[0])
    rate = float(cfg["interpolation_rate_hz"]); step = 1.0 / rate
    if support_duration < float(cfg["min_continuous_support_sec"]):
        return float("nan"), float("nan"), {"reason": "spectral_support_insufficient", "continuous_support_sec": support_duration, "tachogram_span_sec": tachogram_span, "interpolation_points": 0}
    grid = np.arange(times[0], times[-1] + step * 0.5, step, dtype=float)
    if grid.size < 8 or grid[-1] > times[-1] + 1e-9:
        grid = grid[grid <= times[-1] + 1e-9]
    if grid.size < 8:
        return float("nan"), float("nan"), {"reason": "spectral_support_insufficient", "continuous_support_sec": support_duration, "tachogram_span_sec": tachogram_span, "interpolation_points": int(grid.size)}
    tachogram = np.interp(grid, times, values)
    nperseg = min(int(cfg["nperseg"]), int(grid.size)); noverlap = min(int(cfg["noverlap"]), max(0, nperseg - 1)); nfft = max(int(cfg["nfft"]), nperseg)
    freq, psd = welch(tachogram, fs=rate, window=cfg["window"], nperseg=nperseg, noverlap=noverlap, nfft=nfft, detrend=cfg["detrend"], return_onesided=bool(cfg["onesided"]), scaling=cfg["scaling"], average=cfg["average"])
    lf = _integrate_band(freq, psd, *cfg["lf_hz"]); hf = _integrate_band(freq, psd, *cfg["hf_hz"])
    return lf, hf, {"continuous_support_sec": support_duration, "tachogram_span_sec": tachogram_span, "interpolation_points": int(grid.size), "interpolation_rate_hz": rate, "nperseg": nperseg, "noverlap": noverlap, "nfft": nfft, "psd_unit": "ms^2/Hz", "power_unit": "ms^2"}

def _reason_array(values: Sequence[Any]) -> np.ndarray:
    texts = [str(value) for value in values]
    width = max(1, max((len(value) for value in texts), default=0))
    return np.asarray(texts, dtype=f"<U{width}")


def _math_result(values: np.ndarray, valid: np.ndarray, reasons: list[str], failed: list[str], diagnostics: dict[str, Any]) -> HRVComputationResult:
    return HRVComputationResult(np.asarray(values, dtype=np.float64), np.asarray(valid, dtype=bool), _reason_array(reasons), tuple(dict.fromkeys(failed)), diagnostics)

def compute_nn_hrv_metrics(nn_values: Sequence[float], *, policy: Mapping[str, Any] | None = None, enforce_support: bool = False) -> HRVComputationResult:
    cfg = _policy(policy); x = np.asarray(nn_values, dtype=np.float64).reshape(-1); values = np.full(14, np.nan, dtype=float); valid = np.zeros(14, dtype=bool); reasons = ["" for _ in range(14)]; failed: list[str] = []
    if not np.all(np.isfinite(x)):
        return _empty_computation("nonfinite_nn")
    finite = x.copy(); pairs = np.diff(finite) if finite.size >= 2 else np.empty(0, dtype=float); triples = [(finite[i], finite[i+1], finite[i+2]) for i in range(max(0, finite.size-2))]
    if enforce_support and finite.size < int(cfg["nn"]["min_nn_count"]): failed.append("nn_support_insufficient")
    if enforce_support and pairs.size < int(cfg["nn"]["min_adjacent_pairs"]): failed.append("adjacent_pair_support_insufficient")
    if enforce_support and len(triples) < int(cfg["nn"]["min_triples"]): failed.append("triple_support_insufficient")
    def set_value(index: int, value: float, needed: int, reason: str = "math_insufficient"):
        if finite.size < needed:
            reasons[index] = reason; failed.append(reasons[index]); return
        if not np.isfinite(value):
            if index in (3, 4, 13) and pairs.size < (int(cfg["nn"]["min_adjacent_pairs"]) if enforce_support else 1):
                reasons[index] = "adjacent_pair_support_insufficient"
            elif index == 8 and pairs.size < 3:
                reasons[index] = "adjacent_pair_support_insufficient"
            elif index in (9, 10) and len(triples) < (int(cfg["nn"]["min_triples"]) if enforce_support else 1):
                reasons[index] = "triple_support_insufficient"
            else:
                reasons[index] = "invalid_value"
            failed.append(reasons[index]); return
        if enforce_support and (index in (0,1,2,11,12,13) and finite.size < int(cfg["nn"]["min_nn_count"])):
            reasons[index] = "nn_support_insufficient"; failed.append(reasons[index]); return
        if enforce_support and index in (3,4,8,13) and pairs.size < int(cfg["nn"]["min_adjacent_pairs"]):
            reasons[index] = "adjacent_pair_support_insufficient"; failed.append(reasons[index]); return
        if enforce_support and index in (9,10) and len(triples) < int(cfg["nn"]["min_triples"]):
            reasons[index] = "triple_support_insufficient"; failed.append(reasons[index]); return
        values[index] = float(value); valid[index] = True
    med = float(np.median(finite)) if finite.size else float("nan")
    set_value(0, med, 1, "nn_support_insufficient")
    mad = 1.4826 * float(np.median(np.abs(finite - med))) / med if finite.size and med != 0 else float("nan")
    set_value(1, mad, 2, "math_insufficient")
    set_value(2, float(np.std(finite, ddof=1) / np.mean(finite)) if finite.size >= 2 and np.mean(finite) != 0 else float("nan"), 2, "math_insufficient")
    rmssd = float(np.sqrt(np.mean(pairs ** 2))) if pairs.size else float("nan")
    set_value(3, rmssd / np.mean(finite) if finite.size and np.mean(finite) != 0 else float("nan"), 1, "adjacent_pair_support_insufficient")
    set_value(4, 100.0 * float(np.count_nonzero(np.abs(pairs) > 20.0)) / pairs.size if pairs.size else float("nan"), 1, "adjacent_pair_support_insufficient")
    # 频域项需要真实峰时间；仅有NN数值的公式入口不能伪造时间轴。
    lf, hf, psd_diag = float("nan"), float("nan"), {"reason": "spectral_support_requires_peak_times"}
    if np.isfinite(lf): values[5] = lf; valid[5] = True
    else: reasons[5] = "spectral_support_insufficient"; failed.append(reasons[5])
    if np.isfinite(hf): values[6] = hf; valid[6] = True
    else: reasons[6] = "spectral_support_insufficient"; failed.append(reasons[6])
    if np.isfinite(lf) and np.isfinite(hf) and lf + hf > 0:
        values[7] = hf / (lf + hf); valid[7] = True
    else: reasons[7] = "zero_denominator" if np.isfinite(lf) and np.isfinite(hf) else "spectral_support_insufficient"; failed.append(reasons[7])
    if pairs.size >= 3:
        adjacent_a = finite[:-1]
        adjacent_b = finite[1:]
        u = (adjacent_a - adjacent_b) / np.sqrt(2.0)
        v = (adjacent_a + adjacent_b) / np.sqrt(2.0)
        sd2 = float(np.std(v, ddof=1))
        ratio = float(np.std(u, ddof=1) / sd2) if sd2 != 0 else float("nan")
    else:
        ratio = float("nan")
    set_value(8, ratio, 2, "adjacent_pair_support_insufficient")
    if len(triples):
        q=np.percentile(finite,[0,25,50,75,100],method="linear"); symbols=np.digitize(finite,q[1:-1],right=False); zero=sum(a==b==c for a,b,c in zip(symbols[:-2],symbols[1:-1],symbols[2:])); uv=sum((b-a)*(c-b)<0 for a,b,c in zip(symbols[:-2],symbols[1:-1],symbols[2:])); set_value(9,float(zero/len(triples)),1,"triple_support_insufficient"); set_value(10,float(uv/len(triples)),1,"triple_support_insufficient")
    else:
        reasons[9]=reasons[10]="triple_support_insufficient"; failed.extend([reasons[9],reasons[10]])
    set_value(11,float(np.mean(finite)) if finite.size else float("nan"),1,"nn_support_insufficient")
    set_value(12,float(np.std(finite,ddof=1)) if finite.size>=2 else float("nan"),2,"math_insufficient")
    set_value(13,rmssd,1,"adjacent_pair_support_insufficient")
    if enforce_support:
        for i in range(14):
            if not valid[i] and not reasons[i]: reasons[i] = failed[0] if failed else "invalid_value"
    return _math_result(values,valid,reasons,failed,{"nn_count":int(finite.size),"adjacent_pair_count":int(pairs.size),"triple_count":len(triples),"lf_power_ms2":lf,"hf_power_ms2":hf,"psd":psd_diag})

def _empty_computation(reason: str) -> HRVComputationResult:
    return _math_result(np.full(14,np.nan),np.zeros(14,bool),[reason]*14,[reason],{"nn_count":0,"adjacent_pair_count":0,"triple_count":0})

def _detect_block(block_signal: np.ndarray, fs: float, method: str, policy: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    try:
        import neurokit2 as nk
        import warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cleaned = nk.ecg_clean(block_signal, sampling_rate=fs, method=policy["detection"]["clean_method"]) if method == "cleaned" else np.asarray(block_signal, dtype=float)
            _, info = nk.ecg_peaks(cleaned, sampling_rate=fs, method="neurokit", correct_artifacts=False, show=False)
        raw_audit = audit_peak_integrity(info.get("ECG_R_Peaks", []), block_signal.size)
        if raw_audit["nonfinite_count"] or raw_audit["noninteger_count"] or raw_audit["duplicate_count"] or raw_audit["out_of_range_count"] or not raw_audit["ordered"]:
            return np.empty(0,dtype=int), np.empty(0,dtype=int), {"status":"raw_peak_integrity_failed","raw_audit":raw_audit,"warnings":[str(x.message) for x in caught],"processed_signal":cleaned}
        raw = raw_audit["normalized"]
        if raw.size < 2:
            return raw, np.empty(0,dtype=int), {"status":"insufficient_peaks","raw_audit":raw_audit,"warnings":[str(x.message) for x in caught],"processed_signal":cleaned}
        try:
            with warnings.catch_warnings(record=True) as correction_warnings:
                warnings.simplefilter("always")
                _, corrected_raw = nk.signal_fixpeaks(raw, sampling_rate=fs, method=policy["detection"]["correction_method"], iterative=True, show=False)
            corrected_audit = audit_peak_integrity(corrected_raw, block_signal.size)
            if corrected_audit["nonfinite_count"] or corrected_audit["noninteger_count"] or corrected_audit["duplicate_count"] or corrected_audit["out_of_range_count"] or not corrected_audit["ordered"]:
                return raw, np.empty(0,dtype=int), {"status":"corrected_peak_integrity_failed","raw_audit":raw_audit,"corrected_audit":corrected_audit,"warnings":[str(x.message) for x in caught]+[str(x.message) for x in correction_warnings],"processed_signal":cleaned}
            corrected=corrected_audit["normalized"]
            return raw, corrected, {"status":"ok","raw_audit":raw_audit,"corrected_audit":corrected_audit,"warnings":[str(x.message) for x in caught]+[str(x.message) for x in correction_warnings],"processed_signal":cleaned}
        except Exception as exc:
            return raw, np.empty(0,dtype=int), {"status":"correction_failed","raw_audit":raw_audit,"warnings":[str(x.message) for x in caught],"error":f"{type(exc).__name__}: {exc}","processed_signal":cleaned}
    except Exception as exc:
        return np.empty(0,dtype=int), np.empty(0,dtype=int), {"status":"detection_failed","warnings":[],"error":f"{type(exc).__name__}: {exc}"}

def _apply_psd_metrics(comp: HRVComputationResult, intervals: Sequence[NNInterval], policy: Mapping[str, Any]) -> HRVComputationResult:
    """用真实NN峰时间计算H06-H08，避免从NN数值伪造时间轴。"""
    values = comp.values.copy(); valid = comp.valid_mask.copy(); reasons = comp.primary_reasons.copy()
    failed = [reason for reason in comp.all_failed_conditions if reason not in {"spectral_support_insufficient", "spectral_support_requires_peak_times"}]
    lf, hf, diag = frequency_power_from_nn(intervals, policy)
    values[5:8] = np.nan; valid[5:8] = False
    if np.isfinite(lf):
        values[5] = lf; valid[5] = True; reasons[5] = ""
    else:
        reasons[5] = str(diag.get("reason", "spectral_support_insufficient")); failed.append(reasons[5])
    if np.isfinite(hf):
        values[6] = hf; valid[6] = True; reasons[6] = ""
    else:
        reasons[6] = str(diag.get("reason", "spectral_support_insufficient")); failed.append(reasons[6])
    if np.isfinite(lf) and np.isfinite(hf) and lf + hf > 0:
        values[7] = hf / (lf + hf); valid[7] = True; reasons[7] = ""
    else:
        reasons[7] = "zero_denominator" if np.isfinite(lf) and np.isfinite(hf) else str(diag.get("reason", "spectral_support_insufficient")); failed.append(reasons[7])
    support_sec = float(diag.get("continuous_support_sec", chain_support_sec(intervals))) if intervals else 0.0
    diagnostics = dict(comp.diagnostics); diagnostics.update({"continuous_support_sec": support_sec, "lf_power_ms2": lf, "hf_power_ms2": hf, "psd": diag})
    return HRVComputationResult(values, valid, reasons, tuple(dict.fromkeys(failed)), diagnostics)


def compute_ecg_window_hrv(signal: np.ndarray, qc_mask: np.ndarray, *, fs: float = ECG_FS, method: str = "cleaned", policy: Mapping[str, Any] | None = None, window_start_sample: int = 0, window_start_sec: float = 0.0) -> ECGWindowHRVResult:
    cfg=_policy(policy); values=np.full(14,np.nan); valid=np.zeros(14,bool); reasons=_reason_array(["no_continuous_block"] * 14); provenance=[]
    signal=np.asarray(signal,dtype=float).reshape(-1); qc=np.asarray(qc_mask,dtype=bool).reshape(-1)
    if signal.size != qc.size or not np.isclose(float(fs),ECG_FS,rtol=0,atol=1e-9): return ECGWindowHRVResult(values,valid,reasons,("schema_or_fs_error",),{"nn_count":0},provenance)
    allowed=np.isfinite(signal)&qc; min_samples=int(round(float(cfg["continuity"]["min_continuous_block_sec"])*fs)); blocks=_runs(allowed,min_samples); block_chains=[]
    for bid,(a,b) in enumerate(blocks):
        raw,corrected,diag=_detect_block(signal[a:b],fs,method,cfg); provenance.append({"block_id":bid,"start_sample":a+window_start_sample,"end_sample":b+window_start_sample,"status":diag.get("status"),"raw_peak_integrity":diag.get("raw_audit",{}),"corrected_peak_integrity":diag.get("corrected_audit",{}),"raw_peak_sample_indices":diag.get("raw_audit",{}).get("normalized",np.empty(0,dtype=np.int64)),"corrected_peak_sample_indices":diag.get("corrected_audit",{}).get("normalized",np.empty(0,dtype=np.int64)),"warnings":diag.get("warnings",[])})
        if diag.get("status") != "ok" or corrected.size < 2: continue
        global_peaks=corrected+a+window_start_sample
        intervals, rejected = build_candidate_nn_audited(global_peaks,fs=fs,window_start_sample=window_start_sample,window_end_sample=window_start_sample+signal.size,block_id=bid,policy=cfg)
        chains=split_nn_chains(intervals)
        block_chains.extend(chains)
        provenance[-1]["candidate_nn_ms"] = [float(x.value_ms) for x in intervals]
        provenance[-1]["candidate_nn_start_samples"] = [int(x.start_peak) for x in intervals]
        provenance[-1]["candidate_nn_end_samples"] = [int(x.end_peak) for x in intervals]
        provenance[-1]["candidate_nn_ordinals"] = [int(x.ordinal) for x in intervals]
        provenance[-1]["candidate_nn_rejected"] = rejected
        provenance[-1]["nn_chain_count"] = len(chains)
        provenance[-1]["nn_chain_lengths"] = [len(x) for x in chains]
        provenance[-1]["nn_chain_support_sec"] = [chain_support_sec(x, fs) for x in chains]
    # 过滤RR后再次拆链；不同链不得拼接，最长真实支持相同时取最早链。
    candidates = [x for x in block_chains if x]
    if candidates:
        chosen = sorted(candidates, key=lambda chain: (-chain_support_sec(chain, fs), chain[0].start_peak))[0]
        comp = _apply_psd_metrics(compute_nn_hrv_metrics([n.value_ms for n in chosen], policy=cfg, enforce_support=True), chosen, cfg)
        comp.diagnostics.update({
            "block_count": len(blocks), "chain_count": len(candidates),
            "selected_block_id": int(chosen[0].block_id), "selected_chain_index": int(candidates.index(chosen)),
            "selected_chain_nn_count": len(chosen), "selected_chain_support_sec": chain_support_sec(chosen, fs),
            "aggregation": "longest_true_nn_chain_only", "all_chain_nn_counts": [len(x) for x in candidates],
            "all_chain_support_sec": [chain_support_sec(x, fs) for x in candidates],
        })
    else:
        comp = _empty_computation("no_continuous_block")
        comp.diagnostics.update({"block_count": len(blocks), "chain_count": 0, "all_chain_nn_counts": [], "all_chain_support_sec": []})
    reasons=_reason_array(["" if x else (comp.primary_reasons[i] or "invalid_value") for i,x in enumerate(comp.valid_mask)])
    return ECGWindowHRVResult(comp.values,comp.valid_mask,reasons,comp.all_failed_conditions,comp.diagnostics,provenance)

def aggregate_ecg_window_metrics(values: np.ndarray, valid_mask: np.ndarray, *, min_valid_windows: int = 2) -> HRVComputationResult:
    x=np.asarray(values,dtype=float); m=np.asarray(valid_mask,dtype=bool)
    if x.ndim != 2 or x.shape[1] != 14 or m.shape != x.shape: raise ValueError("A14 window arrays must be (n,14)")
    out=np.full(14,np.nan); valid=np.zeros(14,bool); reasons=["no_valid_window"]*14; counts=np.sum(m,axis=0)
    for i in range(14):
        if counts[i] >= int(min_valid_windows): out[i]=float(np.mean(x[m[:,i],i])); valid[i]=True; reasons[i]=""
        elif counts[i] > 0: reasons[i]="night_support_insufficient"
    return _math_result(out,valid,reasons,[r for r in reasons if r],{"valid_window_counts":counts.astype(int).tolist(),"window_count":int(x.shape[0]),"min_valid_windows":int(min_valid_windows)})

def empty_night_result(reason: str) -> BranchResult:
    values=np.full(14,np.nan,dtype=np.float32); valid=np.zeros(14,bool); reasons=_reason_array([reason] * 14)
    block=FeatureBlock(branch_name=BRANCH_NAME,feature_set_name="hrv_night_a14",time_scale=TimeScale.NIGHT,values=values,feature_names=A14_FEATURE_NAMES,valid_mask=valid,invalid_reason=reasons,config=DEFAULT_P1_POLICY,version=FEATURE_VERSION)
    return BranchResult(branch_name=BRANCH_NAME,blocks=(block,),metadata={"feature_version":FEATURE_VERSION,"scope":"full_night","error":reason})

def hrv_a14_model_transform(feature_name: str, value: float) -> float:
    if feature_name in {A14_FEATURE_NAMES[5],A14_FEATURE_NAMES[6]}: return float(np.log1p(value))
    return float(value)

def extract_a14(context: Any, config: Mapping[str, Any] | None = None) -> BranchResult:
    cfg=_policy(config); present=np.asarray(context.channel_present("ecg"),dtype=bool).reshape(-1)
    if present.size == 0 or not bool(present[0]): return empty_night_result("channel_missing")
    try:
        dimensions = context.metadata("ecg", "physical_dimension")
        stored_unit = np.asarray(dimensions).reshape(-1)[0]
    except Exception:
        return empty_night_result("unit_unknown")
    ds=context.dataset("signals/ecg"); shape=tuple(int(x) for x in ds.shape)
    if len(shape)!=3 or shape[1]!=1 or shape[2]!=6000: return empty_night_result("processing_error")
    signal, unit_meta = normalize_ecg_input(np.asarray(ds[:,0,:],dtype=float).reshape(-1), stored_unit)
    if unit_meta.get("unit_status") == "unknown":
        return empty_night_result("unit_unknown")
    inv=np.asarray(context.quality_dataset("channel_invalid_5s","ecg")[...],bool).reshape(-1); flat=np.asarray(context.quality_dataset("channel_flat_5s","ecg")[...],bool).reshape(-1); ext=np.asarray(context.quality_dataset("channel_extreme_5s","ecg")[...],bool).reshape(-1); qc=~(inv|flat|ext); qc=np.repeat(qc, int(round(ECG_FS*SUBWINDOW_SECONDS))); qc=qc[:signal.size]&np.isfinite(signal); window_values=[]; window_valid=[]; starts=[]
    for start in range(0,signal.size-WINDOW_SAMPLES+1,WINDOW_SAMPLES):
        out=extract_window_a14(signal[start:start+WINDOW_SAMPLES],qc[start:start+WINDOW_SAMPLES],fs=ECG_FS,method=cfg["detection"]["method"] if cfg["detection"]["method"] in {"cleaned","direct"} else "cleaned",policy=cfg,window_start_sample=start,window_start_sec=start/ECG_FS); window_values.append(out.values); window_valid.append(out.valid_mask); starts.append(start/ECG_FS)
    if not window_values: return empty_night_result("coverage_insufficient")
    agg=aggregate_ecg_window_metrics(np.stack(window_values),np.stack(window_valid),min_valid_windows=int(cfg["night"]["min_valid_windows"]))
    reasons=_reason_array(["" if x else agg.primary_reasons[i] for i,x in enumerate(agg.valid_mask)])
    block=FeatureBlock(branch_name=BRANCH_NAME,feature_set_name="hrv_night_a14",time_scale=TimeScale.NIGHT,values=agg.values.astype(np.float32),feature_names=A14_FEATURE_NAMES,valid_mask=agg.valid_mask,invalid_reason=reasons,config=cfg,version=FEATURE_VERSION)
    return BranchResult(branch_name=BRANCH_NAME,blocks=(block,),metadata={"feature_version":FEATURE_VERSION,"scope":"full_night","window_starts_sec":starts,"valid_window_counts":agg.diagnostics["valid_window_counts"],"candidate_values":agg.values.tolist(),"ecg_unit_meta":unit_meta})

# 旧 A14 名称仅作为历史回放兼容层；正式 S2 代码使用上面的职责命名。
A14Computation = HRVComputationResult
A14WindowOutput = ECGWindowHRVResult


def compute_a14_from_nn(*args: Any, **kwargs: Any) -> HRVComputationResult:
    return compute_nn_hrv_metrics(*args, **kwargs)


def extract_window_a14(*args: Any, **kwargs: Any) -> ECGWindowHRVResult:
    return compute_ecg_window_hrv(*args, **kwargs)


def aggregate_a14_windows(*args: Any, **kwargs: Any) -> HRVComputationResult:
    return aggregate_ecg_window_metrics(*args, **kwargs)


# ---------------------------------------------------------------------------
# Stage-HRV-S2 正式两维入口
# ---------------------------------------------------------------------------

S2_STAGE_NAMES = {1: "N3", 2: "N2", 3: "N1", 4: "REM"}
S2_TARGET_CODES = frozenset(S2_STAGE_NAMES)
S2_NREM_CODES = frozenset({1, 2, 3})
S2_WINDOW_EPOCHS = 10
S2_EPOCH_SECONDS = 30.0
S2_SUBWINDOW_SECONDS = 5.0
S2_TIME_TOLERANCE = 1e-6


@dataclass(frozen=True)
class ECGExactWindow:
    key: str
    start_pos: int
    end_pos: int
    stage_code: int
    stage_name: str
    stage_run_id: int
    start_sec: float
    end_sec: float


@dataclass(frozen=True)
class ECGExactWindowQC:
    channel_present: bool
    mask_available: bool
    mask_shape_valid: bool
    valid_5s: int
    valid_sec: float
    longest_valid_sec: float
    ge120: bool
    ge240: bool
    all_300: bool


def _s2_time_valid(epoch_index: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    n = len(epoch_index)
    valid = np.isfinite(starts) & np.isfinite(ends) & (ends > starts)
    valid &= np.isclose(ends - starts, S2_EPOCH_SECONDS, atol=S2_TIME_TOLERANCE, rtol=0.0)
    if n > 1:
        valid[1:] &= np.isclose(epoch_index[1:], epoch_index[:-1] + 1, atol=0, rtol=0)
        valid[1:] &= np.isclose(starts[1:], ends[:-1], atol=S2_TIME_TOLERANCE, rtol=0.0)
    return valid


def _s2_stage_runs(stage: np.ndarray, time_valid: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    start: int | None = None
    code: int | None = None
    for index in range(len(stage)):
        current = int(stage[index])
        accepted = bool(time_valid[index]) and current in S2_TARGET_CODES
        continuation = accepted and start is not None and index > 0 and bool(time_valid[index - 1]) and current == code
        if accepted and start is None:
            start, code = index, current
        elif accepted and not continuation:
            if start is not None:
                runs.append({"start": start, "end": index, "code": int(code), "duration_sec": float(ends[index - 1] - starts[start])})
            start, code = index, current
        elif not accepted and start is not None:
            runs.append({"start": start, "end": index, "code": int(code), "duration_sec": float(ends[index - 1] - starts[start])})
            start, code = None, None
    if start is not None:
        runs.append({"start": start, "end": len(stage), "code": int(code), "duration_sec": float(ends[-1] - starts[start])})
    return runs


def build_ecg_exact_windows(stage: np.ndarray, epoch_index: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> tuple[list[ECGExactWindow], bool, int]:
    time_valid = _s2_time_valid(epoch_index, starts, ends)
    runs = _s2_stage_runs(stage, time_valid, starts, ends)
    windows: list[ECGExactWindow] = []
    for run_id, run in enumerate(runs):
        code = int(run["code"])
        start, end = int(run["start"]), int(run["end"])
        for offset in range(0, max(0, end - start - S2_WINDOW_EPOCHS + 1), S2_WINDOW_EPOCHS):
            a, b = start + offset, start + offset + S2_WINDOW_EPOCHS
            windows.append(ECGExactWindow(
                key=f"{a}:{b}:{code}", start_pos=a, end_pos=b, stage_code=code,
                stage_name=S2_STAGE_NAMES[code], stage_run_id=run_id,
                start_sec=float(starts[a]), end_sec=float(ends[b - 1]),
            ))
    valid_stage = int(np.isin(stage, list(S2_STAGE_NAMES)).sum())
    return windows, bool(time_valid.all()), valid_stage


def _s2_longest_true(mask: np.ndarray) -> int:
    best = current = 0
    for flag in np.asarray(mask, dtype=bool).reshape(-1):
        current = current + 1 if bool(flag) else 0
        best = max(best, current)
    return best


def _ecg_exact_window_qc(present: bool, invalid: np.ndarray | None, flat: np.ndarray | None, extreme: np.ndarray | None, start_pos: int, end_pos: int) -> ECGExactWindowQC:
    if not present or invalid is None or flat is None or extreme is None:
        return ECGExactWindowQC(present, False, False, 0, 0.0, 0.0, False, False, False)
    arrays = [np.asarray(item) for item in (invalid, flat, extreme)]
    shape_valid = all(item.ndim == 3 and item.shape[1] == 1 for item in arrays) and arrays[0].shape == arrays[1].shape == arrays[2].shape
    if not shape_valid or start_pos < 0 or end_pos > arrays[0].shape[0]:
        return ECGExactWindowQC(present, True, False, 0, 0.0, 0.0, False, False, False)
    mask5 = ~(arrays[0][start_pos:end_pos, 0, :] | arrays[1][start_pos:end_pos, 0, :] | arrays[2][start_pos:end_pos, 0, :])
    mask5 = mask5.reshape(-1)
    valid_5s = int(mask5.sum())
    longest_sec = _s2_longest_true(mask5) * S2_SUBWINDOW_SECONDS
    return ECGExactWindowQC(True, True, True, valid_5s, valid_5s * S2_SUBWINDOW_SECONDS, longest_sec, longest_sec >= 120.0, longest_sec >= 240.0, bool(mask5.all()))


def _ecg_exact_selection_order(windows: Sequence[ECGExactWindow]) -> list[ECGExactWindow]:
    ordered = sorted(list(windows), key=lambda item: (item.start_sec, item.end_sec, item.key))
    if len(ordered) <= 3:
        return ordered
    starts = np.asarray([float(item.start_sec) for item in ordered])
    targets = [float(np.quantile(starts, quantile, method="linear")) for quantile in (0.25, 0.50, 0.75)]
    result: list[ECGExactWindow] = []
    for target in targets:
        candidates = sorted(ordered, key=lambda item: (abs(float(item.start_sec) - target), float(item.start_sec), item.key))
        chosen = next((item for item in candidates if item not in result), None)
        if chosen is not None:
            result.append(chosen)
    remaining = sorted(ordered, key=lambda item: (min(abs(float(item.start_sec) - target) for target in targets), float(item.start_sec), item.key))
    result.extend(item for item in remaining if item not in result)
    return result


def _ecg_exact_select_representative(windows: Sequence[ECGExactWindow], valid_keys: set[str], max_windows: int = 3) -> list[ECGExactWindow]:
    return [item for item in _ecg_exact_selection_order(windows) if item.key in valid_keys][:max_windows]


def _s2_annotation_source(context: Any) -> str:
    path = str(getattr(getattr(context, "record", None), "caisr_edf_path", "")).lower()
    if "algorithmic_annotations" in path or "caisr" in path:
        return "auto"
    if "human_annotations" in path or "expert_annotations" in path:
        return "manual"
    return "unavailable"


def _s2_empty_result(reason: str, *, metadata: Mapping[str, Any] | None = None) -> BranchResult:
    values = np.full(2, np.nan, dtype=np.float64)
    valid = np.zeros(2, dtype=bool)
    reasons = _reason_array([reason, reason])
    block = FeatureBlock(
        branch_name=BRANCH_NAME, feature_set_name=S2_FEATURE_SET_NAME, time_scale=TimeScale.NIGHT,
        values=values, feature_names=S2_FEATURE_NAMES, valid_mask=valid, invalid_reason=reasons,
        config=DEFAULT_P1_POLICY, version=S2_FEATURE_VERSION,
    )
    payload = {"feature_version": S2_FEATURE_VERSION, "feature_set_name": S2_FEATURE_SET_NAME, "scope": "full_night", "error": reason}
    if metadata:
        payload.update(dict(metadata))
    return BranchResult(branch_name=BRANCH_NAME, blocks=(block,), metadata=payload)


def _s2_reason_for_output(output: ECGWindowHRVResult, metric_index: int) -> str:
    if bool(output.valid_mask[metric_index]) and np.isfinite(output.values[metric_index]):
        return ""
    reason = str(output.primary_reasons[metric_index]) if str(output.primary_reasons[metric_index]) else "invalid_value"
    failed = [str(value) for value in output.all_failed_conditions if value]
    return ";".join(dict.fromkeys([reason] + failed))


def extract_stage_hrv_s2(context: Any, config: Mapping[str, Any] | None = None) -> BranchResult:
    """从 canonical HDF5 原样执行冻结 Stage-HRV-S2 两维计算。"""

    cfg = _policy(config)
    present = np.asarray(context.channel_present("ecg"), dtype=bool).reshape(-1)
    base_meta = {"annotation_source": _s2_annotation_source(context), "ecg_channel_present": bool(present.size and present[0])}
    if present.size == 0 or not bool(present[0]):
        return _s2_empty_result("channel_missing", metadata=base_meta)
    try:
        stored_unit = np.asarray(context.metadata("ecg", "physical_dimension")).reshape(-1)[0]
        stage = context.read("annotations/caisr/stage").astype(np.int16, copy=False).reshape(-1)
        timebase = context.timebase()
        epoch_index = np.asarray(timebase["epoch_index"]).reshape(-1)
        starts = np.asarray(timebase["epoch_start_sec"], dtype=float).reshape(-1)
        ends = np.asarray(timebase["epoch_end_sec"], dtype=float).reshape(-1)
        if not (len(stage) == len(epoch_index) == len(starts) == len(ends) == context.epoch_count):
            return _s2_empty_result("ecg_exact_timebase_length_mismatch", metadata=base_meta)
        windows, timebase_valid, valid_stage = build_ecg_exact_windows(stage, epoch_index, starts, ends)
        invalid = np.asarray(context.quality_dataset("channel_invalid_5s", "ecg")[...], dtype=bool)
        flat = np.asarray(context.quality_dataset("channel_flat_5s", "ecg")[...], dtype=bool)
        extreme = np.asarray(context.quality_dataset("channel_extreme_5s", "ecg")[...], dtype=bool)
        qcs = {window.key: _ecg_exact_window_qc(True, invalid, flat, extreme, window.start_pos, window.end_pos) for window in windows}
        dataset = context.dataset("signals/ecg")
        if len(dataset.shape) != 3 or tuple(dataset.shape[1:]) != (1, 6000):
            return _s2_empty_result("processing_error", metadata=base_meta)
    except Exception as exc:
        return _s2_empty_result("record_processing_error", metadata={**base_meta, "error_detail": f"{type(exc).__name__}:{exc}"})

    unit = bytes(stored_unit).decode("utf-8", errors="replace") if isinstance(stored_unit, (bytes, np.bytes_)) else str(stored_unit)
    unit_signal_meta: dict[str, Any] = {}
    by_stage = {"N2": [item for item in windows if item.stage_code == 2], "N3": [item for item in windows if item.stage_code == 1]}
    metric_index = {"N2": 7, "N3": 6}
    valid_keys = {"N2": set(), "N3": set()}
    attempted_reasons = {"N2": [], "N3": []}
    values_by_key: dict[tuple[str, str], float] = {}
    diagnostics: list[dict[str, Any]] = []
    for stage_name in ("N2", "N3"):
        for window in _ecg_exact_selection_order(by_stage[stage_name]):
            qc = qcs[window.key]
            feature_name = "S01_N2_HF_relative" if stage_name == "N2" else "S02_N3_HF_absolute_ms2"
            if not qc.ge120:
                attempted_reasons[stage_name].append("qc_proxy_below_120")
                diagnostics.append({"feature": feature_name, "window_start_sec": window.start_sec, "window_end_sec": window.end_sec, "stage_run_id": window.stage_run_id, "valid": False, "reason": "qc_proxy_below_120", "selected": False})
                continue
            start = int(round(window.start_sec * ECG_FS))
            end = int(round(window.end_sec * ECG_FS))
            raw = np.asarray(dataset[window.start_pos:window.end_pos, 0, :], dtype=float).reshape(-1)
            signal, unit_signal_meta = normalize_ecg_input(raw, unit)
            mask5 = ~(invalid[window.start_pos:window.end_pos, 0, :] | flat[window.start_pos:window.end_pos, 0, :] | extreme[window.start_pos:window.end_pos, 0, :])
            qc_mask = np.repeat(mask5.reshape(-1), int(round(ECG_FS * S2_SUBWINDOW_SECONDS)))
            if raw.size != end - start or raw.size != WINDOW_SAMPLES:
                reason = "window_sample_count_mismatch"
                attempted_reasons[stage_name].append(reason)
                diagnostics.append({"feature": feature_name, "window_start_sec": window.start_sec, "window_end_sec": window.end_sec, "stage_run_id": window.stage_run_id, "valid": False, "reason": reason, "selected": False})
                continue
            output = compute_ecg_window_hrv(signal, qc_mask, fs=ECG_FS, method="cleaned", policy=cfg, window_start_sample=start, window_start_sec=window.start_sec)
            index = metric_index[stage_name]
            reason = _s2_reason_for_output(output, index)
            valid = bool(output.valid_mask[index]) and not reason and np.isfinite(output.values[index])
            if valid:
                valid_keys[stage_name].add(window.key)
                values_by_key[(stage_name, window.key)] = float(output.values[index])
            else:
                attempted_reasons[stage_name].extend(value for value in reason.split(";") if value)
            diagnostics.append({
                "feature": feature_name, "window_start_sec": window.start_sec, "window_end_sec": window.end_sec,
                "stage_run_id": window.stage_run_id, "valid": valid, "reason": reason,
                "value": float(output.values[index]) if np.isfinite(output.values[index]) else None,
                "nn_count": int(output.diagnostics.get("selected_chain_nn_count", output.diagnostics.get("nn_count", 0)) or 0),
                "selected_chain_support_sec": float(output.diagnostics.get("selected_chain_support_sec", output.diagnostics.get("continuous_support_sec", 0.0)) or 0.0),
                "selected": False,
            })
            if len(valid_keys[stage_name]) >= 3:
                break

    result_values = np.full(2, np.nan, dtype=np.float64)
    result_valid = np.zeros(2, dtype=bool)
    result_reasons = ["", ""]
    selected_meta: dict[str, list[dict[str, Any]]] = {"N2": [], "N3": []}
    valid_counts: dict[str, int] = {}
    selected_counts: dict[str, int] = {}
    for stage_name, index in (("N2", 0), ("N3", 1)):
        selected = _ecg_exact_select_representative(by_stage[stage_name], valid_keys[stage_name], 3)
        selected_counts[stage_name] = len(selected)
        valid_counts[stage_name] = len(valid_keys[stage_name])
        selected_keys = {item.key for item in selected}
        selected_values = [values_by_key[(stage_name, item.key)] for item in selected if (stage_name, item.key) in values_by_key]
        for row in diagnostics:
            if row["feature"] == S2_FEATURE_NAMES[index] and row["window_start_sec"] is not None:
                key = next((item.key for item in by_stage[stage_name] if np.isclose(item.start_sec, row["window_start_sec"]) and np.isclose(item.end_sec, row["window_end_sec"])), None)
                if key in selected_keys:
                    row["selected"] = True
        selected_meta[stage_name] = [
            {"start_sec": item.start_sec, "end_sec": item.end_sec, "stage_run_id": item.stage_run_id, "key": item.key, "value": values_by_key.get((stage_name, item.key))}
            for item in selected
        ]
        if len(selected_values) >= 2:
            result_values[index] = float(np.median(np.asarray(selected_values, dtype=np.float64)))
            result_valid[index] = True
            result_reasons[index] = ""
        elif not by_stage[stage_name]:
            result_reasons[index] = "no_ecg_exact_stage_window"
        else:
            reason_values = list(dict.fromkeys(value for value in attempted_reasons[stage_name] if value))
            reason_values.append("hrv_valid_window_count_lt2")
            result_reasons[index] = ";".join(dict.fromkeys(reason_values))

    result_status = "both_valid" if bool(result_valid.all()) else ("partial_valid" if bool(result_valid.any()) else "both_invalid")
    metadata = {
        "feature_version": S2_FEATURE_VERSION, "feature_set_name": S2_FEATURE_SET_NAME, "scope": "full_night",
        **base_meta, "stage_timebase_valid": bool(timebase_valid), "valid_stage_epoch_count": int(valid_stage),
        "ecg_exact_n2_windows": int(len(by_stage["N2"])), "ecg_exact_n3_windows": int(len(by_stage["N3"])),
        "ecg_exact_n2_qc_ge120_windows": int(sum(qcs[item.key].ge120 for item in by_stage["N2"])),
        "ecg_exact_n3_qc_ge120_windows": int(sum(qcs[item.key].ge120 for item in by_stage["N3"])),
        "ecg_exact_n2_hrv_valid_windows": int(valid_counts["N2"]), "ecg_exact_n3_hrv_valid_windows": int(valid_counts["N3"]),
        "ecg_exact_selected_n2_windows": int(selected_counts["N2"]), "ecg_exact_selected_n3_windows": int(selected_counts["N3"]),
        "selection_n2_windows": selected_meta["N2"], "selection_n3_windows": selected_meta["N3"],
        "s01_invalid_reason": result_reasons[0], "s02_invalid_reason": result_reasons[1],
        "record_status": result_status, "ecg_unit_meta": unit_signal_meta,
        "window_diagnostics": diagnostics,
    }
    block = FeatureBlock(
        branch_name=BRANCH_NAME, feature_set_name=S2_FEATURE_SET_NAME, time_scale=TimeScale.NIGHT,
        values=result_values, feature_names=S2_FEATURE_NAMES, valid_mask=result_valid,
        invalid_reason=_reason_array(result_reasons), config=cfg, version=S2_FEATURE_VERSION,
    )
    return BranchResult(branch_name=BRANCH_NAME, blocks=(block,), metadata=metadata)


def extract(context: Any, config: Mapping[str, Any] | None = None) -> BranchResult:
    """正式 ECG 分支入口：返回 Stage-HRV-S2 的 S01/S02 两维结果。"""

    return extract_stage_hrv_s2(context, config)


def quality_gate(context: Any, config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    present=np.asarray(context.channel_present("ecg"),dtype=bool).reshape(-1)
    return {"channel_present":bool(present.size and present[0]),"policy_version":_policy(config).get("version",DEFAULT_P1_POLICY["version"]),"scope":"diagnostic_only"}
