"""PSG 单 record canonical waveform 原始/预处理检查器。

本脚本面向单个固定 record，复用 canonical source plan 和正式 waveform 主路径，
把 START、MIDDLE、END 三个 30 秒窗口的 canonical raw waveform 与同一 PSG-relative
时间段的 processed waveform 并排绘图，并把 source provenance 与数值统计写入 TXT。
它是可重复的检查工具，不是新的预处理器，也不写入正式 cache。
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np

from channel_sources import CANONICAL_BY_MODALITY, SourcePlan, build_source_plans
from waveform_cache import (
    CACHE_SCHEMA_VERSION,
    EPOCH_DURATION_SEC,
    EdfHeader,
    PREPROCESSING_VERSION,
    read_edf_header,
)
from waveform_preprocessing import (
    BANDPASS_RANGES_HZ,
    TARGET_SAMPLING_RATES,
    WaveformPreprocessResult,
    derive_canonical_raw,
    preprocess_selected_sources,
)


RECORD_ID = "sub-I0002150026011_ses-2"
SITE = "I0002"
PSG_PATH = Path(
    "/database2/physionet2026_kaggle/data2/physiological_data/I0002/"
    "sub-I0002150026011_ses-2.edf"
)
CAISR_PATH = Path(
    "/database2/physionet2026_kaggle/data2/algorithmic_annotations/I0002/"
    "sub-I0002150026011_ses-2_caisr_annotations.edf"
)
OUTPUT_ROOT = Path("/database/home/gaohaojie/workspace/psg/waveform_record_output")
INSPECTION_CHANNELS = ("EEG_F3", "ECG", "NASAL_PRESSURE", "SPO2")


def inspection_epochs(n_epochs: int) -> tuple[tuple[str, int], ...]:
    """返回固定的 START/MIDDLE/END epoch 索引。"""

    if n_epochs <= 0:
        raise ValueError("at least one complete epoch is required")
    return (
        ("START", 0),
        ("MIDDLE", n_epochs // 2),
        ("END", n_epochs - 1),
    )


def epoch_bounds(epoch_index: int, sampling_rate: float) -> tuple[int, int]:
    """把 PSG-relative epoch 映射为 native/target sample slice。"""

    if not math.isfinite(float(sampling_rate)) or sampling_rate <= 0:
        raise ValueError(f"invalid sampling rate: {sampling_rate!r}")
    start = int(round(epoch_index * EPOCH_DURATION_SEC * sampling_rate))
    end = int(round((epoch_index + 1) * EPOCH_DURATION_SEC * sampling_rate))
    return start, end


def expected_output_names(record_id: str) -> tuple[str, ...]:
    """返回当前工具允许生成的 6 个 PNG 和 1 个 TXT 文件名。"""

    return (
        "eeg.png",
        "eog.png",
        "ecg.png",
        "emg.png",
        "resp.png",
        "spo2.png",
        f"{record_id}.txt",
    )


def _percentile(values: np.ndarray, percentile: float) -> float:
    finite = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    return float(np.percentile(finite, percentile))


def _stats(values: np.ndarray | None) -> dict[str, float | int]:
    if values is None:
        values = np.array([], dtype=np.float64)
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = np.isfinite(values)
    size = int(values.size)
    finite_count = int(finite.sum())
    zero_count = int(np.count_nonzero(values == 0.0))
    return {
        "finite_fraction": float(finite_count / size) if size else 0.0,
        "finite_count": finite_count,
        "nan_count": int(np.isnan(values).sum()),
        "posinf_count": int(np.isposinf(values).sum()),
        "neginf_count": int(np.isneginf(values).sum()),
        "zero_fraction": float(zero_count / size) if size else 0.0,
        "min": _percentile(values, 0.0),
        "p1": _percentile(values, 1.0),
        "p5": _percentile(values, 5.0),
        "median": _percentile(values, 50.0),
        "p95": _percentile(values, 95.0),
        "p99": _percentile(values, 99.0),
        "max": _percentile(values, 100.0),
    }


def _fmt(value: object) -> str:
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            return "nan"
        return f"{float(value):.8g}"
    return str(value)


def _build_canonical_raws(
    plans: Mapping[str, SourcePlan],
    captured: Mapping[int, np.ndarray],
) -> dict[str, np.ndarray | None]:
    """从已读取的 selected raw source 构造 canonical raw（不读取其他 candidate）。"""

    output: dict[str, np.ndarray | None] = {}
    for canonical, plan in plans.items():
        if not plan.channel_present or not plan.selected_source_indices:
            output[canonical] = None
            continue
        sources = [
            captured[index]
            for index in plan.selected_source_indices
            if index in captured
        ]
        if len(sources) != len(plan.selected_source_indices):
            output[canonical] = None
            continue
        try:
            output[canonical] = derive_canonical_raw(sources)
        except ValueError:
            output[canonical] = None
    return output


def _processed_slot_values(
    result: WaveformPreprocessResult,
    modality: str,
    canonical: str,
) -> np.ndarray:
    slot_index = CANONICAL_BY_MODALITY[modality].index(canonical)
    return result.signals[modality][:, slot_index, :].reshape(-1)


def _window_values(
    values: np.ndarray | None,
    epoch_index: int,
    sampling_rate: float,
) -> np.ndarray:
    if values is None:
        return np.array([], dtype=np.float64)
    start, end = epoch_bounds(epoch_index, sampling_rate)
    return np.asarray(values, dtype=np.float64).reshape(-1)[start:end]


def _printable_points(values: np.ndarray | None, count: int = 20) -> str:
    if values is None:
        return "[nan] * 20"
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return "[nan] * 20"
    shown = np.full(count, np.nan, dtype=np.float64)
    shown[: min(count, values.size)] = values[:count]
    return np.array2string(shown, precision=8, separator=", ")


def _source_plan_rows(
    plans: Mapping[str, SourcePlan],
    result: WaveformPreprocessResult,
    raw_waveforms: Mapping[str, np.ndarray | None],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for modality, canonical_names in CANONICAL_BY_MODALITY.items():
        for slot_index, canonical in enumerate(canonical_names):
            plan = plans[canonical]
            processed_values = result.signals[modality][:, slot_index, :]
            rows.append(
                {
                    "modality": modality,
                    "canonical": canonical,
                    "present": bool(plan.channel_present),
                    "selected_source": plan.selected_source_labels,
                    "derivation": plan.derivation_method,
                    "native_fs": plan.native_sampling_rate_hz,
                    "target_fs": TARGET_SAMPLING_RATES[modality],
                    "physical_dimension": plan.physical_dimension,
                    "raw": _stats(raw_waveforms.get(canonical)),
                    "processed": _stats(processed_values),
                }
            )
    return rows


def _summary_lines(
    *,
    psg_header: EdfHeader,
    caisr_header: EdfHeader,
    plans: Mapping[str, SourcePlan],
    result: WaveformPreprocessResult,
    raw_waveforms: Mapping[str, np.ndarray | None],
    epochs: tuple[tuple[str, int], ...],
) -> list[str]:
    lines = [
        f"record_id: {RECORD_ID}",
        f"site: {SITE}",
        f"PSG path: {PSG_PATH}",
        f"CAISR path: {CAISR_PATH}",
        f"PSG start datetime: {psg_header.start_datetime}",
        f"PSG duration: {psg_header.duration_sec:.6f} sec",
        f"CAISR start datetime: {caisr_header.start_datetime}",
        f"CAISR duration: {caisr_header.duration_sec:.6f} sec",
        f"complete epoch count: {result.n_epochs}",
        f"preprocessing_version: {PREPROCESSING_VERSION}",
        f"cache_schema_version: {CACHE_SCHEMA_VERSION}",
        "CAISR alignment: metadata only; no header datetime offset applied",
        "",
        "20-slot summary:",
        (
            "modality | canonical | present | selected_source | derivation | "
            "native_fs | target_fs | physical_dimension | raw_finite_fraction | "
            "raw_zero_fraction | raw_min | raw_median | raw_max | "
            "processed_finite_fraction | processed_min | processed_median | "
            "processed_max"
        ),
    ]
    rows = _source_plan_rows(plans, result, raw_waveforms)
    for row in rows:
        raw_stats = row["raw"]
        processed_stats = row["processed"]
        lines.append(
            f"{row['modality']} | {row['canonical']} | {int(row['present'])} | "
            f"{json.dumps(row['selected_source'], ensure_ascii=False)} | "
            f"{row['derivation']} | {_fmt(row['native_fs'])} | {row['target_fs']} | "
            f"{row['physical_dimension']} | {_fmt(raw_stats['finite_fraction'])} | "
            f"{_fmt(raw_stats['zero_fraction'])} | {_fmt(raw_stats['min'])} | "
            f"{_fmt(raw_stats['median'])} | {_fmt(raw_stats['max'])} | "
            f"{_fmt(processed_stats['finite_fraction'])} | "
            f"{_fmt(processed_stats['min'])} | {_fmt(processed_stats['median'])} | "
            f"{_fmt(processed_stats['max'])}"
        )

    lines.extend(["", "full source provenance and statistics:"])
    for row in rows:
        plan = plans[row["canonical"]]
        raw_stats = row["raw"]
        processed_stats = row["processed"]
        lines.extend(
            [
                f"[{row['modality']}/{row['canonical']}]",
                f"channel_present: {bool(row['present'])}",
                f"candidate_source_labels: {json.dumps(plan.candidate_source_labels, ensure_ascii=False)}",
                f"selected_source_labels: {json.dumps(plan.selected_source_labels, ensure_ascii=False)}",
                f"channel_source: {plan.channel_source}",
                f"derivation_method: {plan.derivation_method}",
                f"reference_system: {plan.reference_system}",
                f"native_fs: {_fmt(plan.native_sampling_rate_hz)}",
                f"target_fs: {row['target_fs']}",
                f"native_sample_count: {plan.native_sample_count}",
                f"physical_dimension: {plan.physical_dimension}",
                f"filter: {BANDPASS_RANGES_HZ.get(row['modality'], 'NOT APPLIED')}",
                (
                    "raw "
                    f"finite_fraction={_fmt(raw_stats['finite_fraction'])} "
                    f"nan={raw_stats['nan_count']} +inf={raw_stats['posinf_count']} "
                    f"-inf={raw_stats['neginf_count']} "
                    f"zero_fraction={_fmt(raw_stats['zero_fraction'])} "
                    f"min={_fmt(raw_stats['min'])} P1={_fmt(raw_stats['p1'])} "
                    f"P5={_fmt(raw_stats['p5'])} median={_fmt(raw_stats['median'])} "
                    f"P95={_fmt(raw_stats['p95'])} P99={_fmt(raw_stats['p99'])} "
                    f"max={_fmt(raw_stats['max'])}"
                ),
                (
                    "processed "
                    f"finite_fraction={_fmt(processed_stats['finite_fraction'])} "
                    f"nan={processed_stats['nan_count']} "
                    f"min={_fmt(processed_stats['min'])} P1={_fmt(processed_stats['p1'])} "
                    f"P5={_fmt(processed_stats['p5'])} "
                    f"median={_fmt(processed_stats['median'])} "
                    f"P95={_fmt(processed_stats['p95'])} "
                    f"P99={_fmt(processed_stats['p99'])} "
                    f"max={_fmt(processed_stats['max'])}"
                ),
            ]
        )

    lines.extend(["", "inspection epochs:"])
    for label, epoch in epochs:
        lines.append(
            f"{label}: epoch={epoch}, PSG-relative="
            f"{epoch * EPOCH_DURATION_SEC:.1f}-{(epoch + 1) * EPOCH_DURATION_SEC:.1f} sec"
        )
    lines.extend(["", "selected waveform first 20 points:"])
    for canonical in INSPECTION_CHANNELS:
        modality = next(
            m for m, names in CANONICAL_BY_MODALITY.items() if canonical in names
        )
        lines.append(f"{canonical} raw: {_printable_points(raw_waveforms.get(canonical))}")
        lines.append(
            f"{canonical} processed: "
            f"{_printable_points(_processed_slot_values(result, modality, canonical))}"
        )

    spo2_stats = _stats(raw_waveforms.get("SPO2"))
    spo2_valid = result.quality.get("spo2_second_valid")
    spo2_valid_fraction = (
        float(np.asarray(spo2_valid, dtype=bool).mean())
        if spo2_valid is not None and np.asarray(spo2_valid).size
        else 0.0
    )
    spo2_valid_start = (
        np.asarray(spo2_valid, dtype=bool)[0].astype(int).tolist()
        if spo2_valid is not None and np.asarray(spo2_valid).shape[0]
        else []
    )
    spo2_valid_end = (
        np.asarray(spo2_valid, dtype=bool)[-1].astype(int).tolist()
        if spo2_valid is not None and np.asarray(spo2_valid).shape[0]
        else []
    )
    lines.extend(
        [
            "",
            "SpO2:",
            f"EDF physical_dimension: {plans['SPO2'].physical_dimension}",
            "raw numeric scale: shown exactly as EDF values",
            (
                "raw P1/P5/P50/P95/P99: "
                f"{_fmt(spo2_stats['p1'])}/{_fmt(spo2_stats['p5'])}/"
                f"{_fmt(spo2_stats['median'])}/{_fmt(spo2_stats['p95'])}/"
                f"{_fmt(spo2_stats['p99'])}"
            ),
            f"canonical_unit: {result.canonical_units.get('spo2', '%')}",
            f"scale_factor: {result.scale_factors.get('spo2', 1)}",
            "per-second aggregation: nanmedian; target=1 Hz; interpolation=linear",
            "invalid values: NaN before interpolation; second_valid retained pre-interpolation",
            f"second_valid_fraction: {_fmt(spo2_valid_fraction)}",
            f"second_valid_start_epoch: {spo2_valid_start}",
            f"second_valid_end_epoch: {spo2_valid_end}",
            "",
            "missing slots:",
        ]
    )
    missing = [canonical for canonical, plan in plans.items() if not plan.channel_present]
    lines.extend(
        [f"{canonical}: channel_present=False; stored waveform=NaN" for canonical in missing]
        or ["none"]
    )
    lines.extend(
        [
            "",
            "inspection contract: SpO2 uses only its per-second linear interpolation; "
            "other modalities have no gap interpolation, normalization, quantization, "
            "clipping, zero-fill, notch, QC, CAISR annotation write, or feature extraction.",
        ]
    )
    return lines


def _plot_window_axis(
    ax,
    values: np.ndarray,
    title: str,
    *,
    ylim: tuple[float, float] | None = None,
) -> None:
    if values.size:
        x = np.linspace(0.0, EPOCH_DURATION_SEC, values.size, endpoint=False)
        ax.plot(x, values, color="black", linewidth=0.45, rasterized=True)
    ax.set_xlim(0.0, EPOCH_DURATION_SEC)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=7)
    ax.grid(True, linewidth=0.25, alpha=0.25)
    ax.tick_params(labelsize=6)


def plot_width_ratios(channel_names: tuple[str, ...]) -> tuple[float, float, float]:
    """按最长 channel name 分配标签列，并等宽放大 raw/processed 列。"""

    if not channel_names:
        raise ValueError("channel_names must not be empty")
    longest_name = max(len(str(name)) for name in channel_names)
    name_width = max(2.0, longest_name / 4.5)
    signal_width = 6.0
    return name_width, signal_width, signal_width


def render_modality_plot(
    *,
    modality: str,
    record_id: str,
    plans: Mapping[str, SourcePlan],
    result: WaveformPreprocessResult,
    raw_waveforms: Mapping[str, np.ndarray | None],
    epochs: tuple[tuple[str, int], ...],
    output_path: Path,
) -> None:
    """绘制一个 modality 的三窗口 raw/processed 对照图。"""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle

    channel_names = CANONICAL_BY_MODALITY[modality]
    rows = len(channel_names) * len(epochs)
    width_ratios = plot_width_ratios(channel_names)
    fig, axes = plt.subplots(
        rows,
        3,
        figsize=(24, max(7.0, rows * 1.65)),
        gridspec_kw={"width_ratios": width_ratios},
        squeeze=False,
    )
    fig.suptitle(f"{modality.upper()} | {record_id}", fontsize=13, y=0.995)
    colors = {"START": "#e8f5e9", "MIDDLE": "#ffebee", "END": "#fafafa"}
    target_rate = TARGET_SAMPLING_RATES[modality]

    for window_index, (label, epoch) in enumerate(epochs):
        for slot_index, canonical in enumerate(channel_names):
            row_index = window_index * len(channel_names) + slot_index
            text_ax, raw_ax, processed_ax = axes[row_index]
            plan = plans[canonical]
            processed_values = _processed_slot_values(result, modality, canonical)
            raw_values = raw_waveforms.get(canonical)
            if not plan.channel_present or raw_values is None:
                text_ax.axis("off")
                text_ax.text(
                    0.96, 0.54, canonical, ha="right", va="center",
                    fontsize=10, weight="bold",
                )
                text_ax.text(
                    0.5, 0.36, "STRUCTURALLY MISSING\nchannel_present=False",
                    ha="center", va="center", fontsize=7,
                )
                raw_ax.axis("off")
                raw_ax.text(
                    0.5, 0.5, "STRUCTURALLY MISSING\nchannel_present=False",
                    ha="center", va="center", fontsize=8,
                )
                processed_ax.axis("off")
                processed_ax.text(
                    0.5, 0.5, "STRUCTURALLY MISSING\nchannel_present=False",
                    ha="center", va="center", fontsize=8,
                )
            else:
                raw_segment = _window_values(
                    raw_values, epoch, plan.native_sampling_rate_hz
                )
                processed_segment = _window_values(
                    processed_values, epoch, target_rate
                )
                finite = np.concatenate(
                    (
                        raw_segment[np.isfinite(raw_segment)],
                        processed_segment[np.isfinite(processed_segment)],
                    )
                )
                ylim = None
                if finite.size:
                    low, high = float(np.min(finite)), float(np.max(finite))
                    margin = max((high - low) * 0.05, 1e-9)
                    if high == low:
                        margin = max(abs(high) * 0.05, 1e-6)
                    ylim = (low - margin, high + margin)
                text_ax.axis("off")
                text_ax.text(
                    0.96, 0.53, canonical, ha="right", va="center",
                    fontsize=10, weight="bold",
                )
                text_ax.text(
                    0.5, 0.34, f"{label}\nepoch={epoch}",
                    ha="center", va="center", fontsize=7,
                )
                _plot_window_axis(raw_ax, raw_segment, "RAW", ylim=ylim)
                _plot_window_axis(
                    processed_ax, processed_segment, "PREPROCESSED", ylim=ylim
                )
            for axis in (text_ax, raw_ax, processed_ax):
                axis.add_patch(
                    Rectangle(
                        (0, 0), 1, 0.035, transform=axis.transAxes,
                        facecolor=colors[label],
                        edgecolor="#888888" if label == "END" else "none",
                        linewidth=0.6,
                        clip_on=False,
                    )
                )

    fig.supxlabel("PSG-relative time within window (sec)", fontsize=9)
    fig.legend(
        handles=[
            Patch(
                facecolor=colors["START"], edgecolor="none",
                label=f"START epoch={epochs[0][1]} time={epochs[0][1] * 30}-{(epochs[0][1] + 1) * 30} sec",
            ),
            Patch(
                facecolor=colors["MIDDLE"], edgecolor="none",
                label=f"MIDDLE epoch={epochs[1][1]} time={epochs[1][1] * 30}-{(epochs[1][1] + 1) * 30} sec",
            ),
            Patch(
                facecolor=colors["END"], edgecolor="#888888",
                label=f"END epoch={epochs[2][1]} time={epochs[2][1] * 30}-{(epochs[2][1] + 1) * 30} sec",
            ),
        ],
        loc="lower center",
        ncol=3,
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, 0.0),
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.98), h_pad=0.65, w_pad=0.6)
    fig.savefig(output_path, dpi=150, format="png")
    plt.close(fig)


def _run_record() -> tuple[Path, ...]:
    """执行固定 record 的 header/source-plan、正式预处理和三窗口检查。"""

    psg_header = read_edf_header(PSG_PATH)
    caisr_header = read_edf_header(CAISR_PATH)
    plans = build_source_plans(
        psg_header.channel_names,
        psg_header.native_sampling_rates_hz,
        psg_header.native_sample_counts,
        psg_header.physical_dimensions,
    )
    n_epochs = int(math.floor(psg_header.duration_sec / EPOCH_DURATION_SEC))
    captured: dict[int, np.ndarray] = {}

    import pyedflib

    reader = pyedflib.EdfReader(str(PSG_PATH))
    try:
        def read_selected(index: int) -> np.ndarray:
            if index not in captured:
                captured[index] = np.asarray(
                    reader.readSignal(index), dtype=np.float64
                ).reshape(-1)
            return captured[index]

        result = preprocess_selected_sources(
            plans,
            psg_header,
            read_signal=read_selected,
            n_epochs=n_epochs,
        )
    finally:
        reader.close()

    raw_waveforms = _build_canonical_raws(plans, captured)
    epochs = inspection_epochs(n_epochs)
    output_dir = OUTPUT_ROOT / RECORD_ID
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = _summary_lines(
        psg_header=psg_header,
        caisr_header=caisr_header,
        plans=plans,
        result=result,
        raw_waveforms=raw_waveforms,
        epochs=epochs,
    )
    for line in lines:
        print(line)
    report_path = output_dir / f"{RECORD_ID}.txt"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    output_paths: list[Path] = [report_path]
    for modality in CANONICAL_BY_MODALITY:
        plot_path = output_dir / f"{modality}.png"
        render_modality_plot(
            modality=modality,
            record_id=RECORD_ID,
            plans=plans,
            result=result,
            raw_waveforms=raw_waveforms,
            epochs=epochs,
            output_path=plot_path,
        )
        output_paths.append(plot_path)
    print("output files:")
    for path in sorted(output_paths):
        print(f"  {path}")
    return tuple(sorted(output_paths))


def main() -> None:
    """运行固定 record 的 canonical waveform inspection。"""

    _run_record()


if __name__ == "__main__":
    main()

# cd /database/home/gaohaojie/workspace/psg
# PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
# /database/home/gaohaojie/.conda/envs/sleepfm_env/bin/python waveform_record.py