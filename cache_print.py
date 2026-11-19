#!/usr/bin/env python3
"""只读检查 canonical HDF5 cache 的结构、统计摘要和关键样本。"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np


SMALL_DATASET_LIMIT = 5000
BLOCK_ELEMENTS = 1_000_000
STAGE_PROBABILITY_NAMES = ("N3", "N2", "N1", "REM", "Wake")
QC_MASK_NAMES = (
    "channel_invalid_5s",
    "channel_flat_5s",
    "channel_extreme_5s",
)
MODALITY_NAMES = ("eeg", "eog", "ecg", "emg", "resp", "spo2")


def _format_value(value: Any) -> str:
    if isinstance(value, bytes):
        return repr(value.decode("utf-8", errors="replace"))
    if isinstance(value, np.generic):
        return _format_value(value.item())
    if isinstance(value, np.ndarray):
        if value.size <= 5000:
            return np.array2string(value, threshold=5000, max_line_width=160)
        return f"<array shape={value.shape} dtype={value.dtype} size={value.size}>"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_format_value(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{key!r}: {_format_value(item)}" for key, item in value.items()) + "}"
    return repr(value)


def _append_attrs(lines: list[str], attrs: h5py.AttributeManager, indent: str = "  ") -> None:
    if not attrs:
        lines.append(f"{indent}{{}}")
        return
    for key in attrs:
        lines.append(f"{indent}{key}: {_format_value(attrs[key])}")


def _dataset_slices(dataset: h5py.Dataset) -> Iterable[tuple[slice, ...] | tuple[()]]:
    if dataset.ndim == 0:
        yield ()
        return
    if dataset.shape[0] == 0:
        return
    trailing = int(np.prod(dataset.shape[1:], dtype=np.int64)) if dataset.ndim > 1 else 1
    rows = max(1, BLOCK_ELEMENTS // max(1, trailing))
    for start in range(0, int(dataset.shape[0]), rows):
        end = min(int(dataset.shape[0]), start + rows)
        yield (slice(start, end),) + (slice(None),) * (dataset.ndim - 1)


def _read_slice(dataset: h5py.Dataset, selection: tuple[slice, ...] | tuple[()]) -> np.ndarray:
    return np.asarray(dataset[selection]) if selection else np.asarray(dataset[()])


def _numeric_stats(dataset: h5py.Dataset) -> dict[str, Any] | None:
    if dataset.dtype.kind not in "biuf":
        return None
    total = int(dataset.size)
    finite_count = nan_count = posinf_count = neginf_count = 0
    total_sum = 0.0
    total_sumsq = 0.0
    finite_min = np.inf
    finite_max = -np.inf
    unique: Counter[Any] = Counter()
    categorical = dataset.dtype.kind in "biu"

    for selection in _dataset_slices(dataset):
        values = _read_slice(dataset, selection)
        flat = values.reshape(-1)
        if categorical:
            raw_values, raw_counts = np.unique(flat, return_counts=True)
            for value, count in zip(raw_values.tolist(), raw_counts.tolist()):
                unique[value] += int(count)
        if dataset.dtype.kind == "f":
            finite = np.isfinite(flat)
            nan_count += int(np.isnan(flat).sum())
            posinf_count += int(np.isposinf(flat).sum())
            neginf_count += int(np.isneginf(flat).sum())
        else:
            finite = np.ones(flat.shape, dtype=bool)
        finite_values = flat[finite].astype(np.float64, copy=False)
        if finite_values.size:
            finite_count += int(finite_values.size)
            total_sum += float(finite_values.sum(dtype=np.float64))
            total_sumsq += float(np.square(finite_values, dtype=np.float64).sum(dtype=np.float64))
            finite_min = min(finite_min, float(finite_values.min()))
            finite_max = max(finite_max, float(finite_values.max()))

    result: dict[str, Any] = {
        "total_elements": total,
        "finite_count": finite_count,
        "nan_count": nan_count,
        "inf_count": posinf_count + neginf_count,
        "finite_fraction": (finite_count / total) if total else 0.0,
    }
    if finite_count:
        mean = total_sum / finite_count
        variance = max(0.0, total_sumsq / finite_count - mean * mean)
        result.update({"min": finite_min, "max": finite_max, "mean": mean, "std": float(np.sqrt(variance))})
    else:
        result.update({"min": "nan", "max": "nan", "mean": "nan", "std": "nan"})
    if dataset.dtype.kind == "f":
        result["posinf_count"] = posinf_count
        result["neginf_count"] = neginf_count
    if categorical:
        result["unique_counts"] = unique
    return result


def _append_stats(lines: list[str], stats: dict[str, Any], indent: str = "  ") -> None:
    for key in ("total_elements", "finite_count", "nan_count", "inf_count", "posinf_count", "neginf_count", "finite_fraction", "min", "max", "mean", "std"):
        if key in stats:
            lines.append(f"{indent}{key}: {_format_value(stats[key])}")
    unique = stats.get("unique_counts")
    if unique is not None:
        lines.append(f"{indent}unique values + counts:")
        items = sorted(unique.items(), key=lambda item: repr(item[0]))
        limit = 100
        for value, count in items[:limit]:
            lines.append(f"{indent}  {_format_value(value)}: {count}")
        if len(items) > limit:
            lines.append(f"{indent}  ... {len(items) - limit} more unique values")


def _channel_names(handle: h5py.File, modality: str, count: int) -> list[str]:
    path = f"metadata/{modality}/channel_names"
    if path in handle:
        try:
            values = handle[path].asstr()[:].tolist()
            return [str(value) for value in values]
        except Exception:
            pass
    return [f"channel_{index}" for index in range(count)]


def _waveform_details(lines: list[str], handle: h5py.File, path: str, dataset: h5py.Dataset) -> None:
    if dataset.ndim != 3 or dataset.shape[0] == 0 or dataset.shape[1] == 0 or dataset.shape[2] == 0:
        lines.append("  waveform_sample: unavailable (expected non-empty 3D epoch/channel/sample dataset)")
        return
    epochs, channels, samples = (int(value) for value in dataset.shape)
    modality = path.strip("/").split("/")[-1]
    names = _channel_names(handle, modality, channels)
    first_count = min(20, samples)
    first_first = np.asarray(dataset[0, 0, :first_count])
    first_last = np.asarray(dataset[0, channels - 1, :first_count])
    last_start = max(0, samples - 20)
    last_first = np.asarray(dataset[epochs - 1, 0, last_start:samples])
    lines.append("  waveform samples:")
    lines.append(f"    first_epoch/first_channel ({names[0]}): {_format_value(first_first)}")
    lines.append(f"    first_epoch/last_channel ({names[-1]}): {_format_value(first_last)}")
    lines.append(f"    last_epoch/first_channel ({names[0]}): {_format_value(last_first)}")

    accumulators = [dict(total=0, finite=0, nan=0, inf=0, sum=0.0, sumsq=0.0, min=np.inf, max=-np.inf) for _ in range(channels)]
    rows = max(1, min(8, BLOCK_ELEMENTS // max(1, channels * samples)))
    for start in range(0, epochs, rows):
        end = min(epochs, start + rows)
        values = np.asarray(dataset[start:end, :, :])
        for channel in range(channels):
            flat = values[:, channel, :].reshape(-1)
            acc = accumulators[channel]
            acc["total"] += int(flat.size)
            if dataset.dtype.kind == "f":
                finite = np.isfinite(flat)
                acc["nan"] += int(np.isnan(flat).sum())
                acc["inf"] += int(np.isinf(flat).sum())
            else:
                finite = np.ones(flat.shape, dtype=bool)
            selected = flat[finite].astype(np.float64, copy=False)
            if selected.size:
                acc["finite"] += int(selected.size)
                acc["sum"] += float(selected.sum(dtype=np.float64))
                acc["sumsq"] += float(np.square(selected, dtype=np.float64).sum(dtype=np.float64))
                acc["min"] = min(acc["min"], float(selected.min()))
                acc["max"] = max(acc["max"], float(selected.max()))
    lines.append("  per-channel:")
    for index, acc in enumerate(accumulators):
        finite_count = acc["finite"]
        total = acc["total"]
        mean = acc["sum"] / finite_count if finite_count else np.nan
        variance = max(0.0, acc["sumsq"] / finite_count - mean * mean) if finite_count else np.nan
        lines.append(f"    {names[index]}:")
        lines.append(f"      finite_fraction: {(finite_count / total) if total else 0.0}")
        lines.append(f"      nan_fraction: {(acc['nan'] / total) if total else 0.0}")
        lines.append(f"      min: {_format_value(acc['min'] if finite_count else np.nan)}")
        lines.append(f"      max: {_format_value(acc['max'] if finite_count else np.nan)}")
        lines.append(f"      mean: {_format_value(mean)}")
        lines.append(f"      std: {_format_value(np.sqrt(variance))}")


def _quality_details(lines: list[str], handle: h5py.File, path: str, dataset: h5py.Dataset) -> None:
    stats = _numeric_stats(dataset)
    if dataset.dtype.kind == "b" and stats is not None:
        true_count = int(np.count_nonzero(_read_slice(dataset, next(iter(_dataset_slices(dataset)), ())))) if dataset.size <= SMALL_DATASET_LIMIT else None
        if true_count is not None:
            lines.append(f"  quality_true_count: {true_count}")
            lines.append(f"  quality_false_count: {int(dataset.size) - true_count}")
        else:
            lines.append("  quality_true/false counts: included in unique values + counts")
    parts = path.strip("/").split("/")
    if len(parts) == 3 and parts[1] in {
        "channel_invalid_5s", "channel_flat_5s", "channel_extreme_5s", "channel_valid_fraction_30s"
    } and dataset.ndim >= 2:
        modality = parts[2]
        channels = int(dataset.shape[1])
        names = _channel_names(handle, modality, channels)
        lines.append("  per-channel quality summary:")
        rows = max(1, min(8, BLOCK_ELEMENTS // max(1, int(np.prod(dataset.shape[2:], dtype=np.int64)) * channels)))
        sums = np.zeros(channels, dtype=np.float64)
        finite_counts = np.zeros(channels, dtype=np.int64)
        total_counts = np.zeros(channels, dtype=np.int64)
        for start in range(0, int(dataset.shape[0]), rows):
            end = min(int(dataset.shape[0]), start + rows)
            values = np.asarray(dataset[start:end, ...])
            for channel in range(channels):
                flat = values[:, channel, ...].reshape(-1)
                total_counts[channel] += flat.size
                if dataset.dtype.kind == "f":
                    finite = np.isfinite(flat)
                    finite_counts[channel] += int(finite.sum())
                    sums[channel] += float(flat[finite].astype(np.float64).sum()) if finite.any() else 0.0
                else:
                    finite_counts[channel] += flat.size
                    sums[channel] += float(np.count_nonzero(flat)) if dataset.dtype.kind == "b" else float(flat.astype(np.float64).sum())
        for channel, name in enumerate(names):
            if parts[1] == "channel_valid_fraction_30s":
                value = sums[channel] / finite_counts[channel] if finite_counts[channel] else np.nan
                lines.append(f"    {name}: mean={_format_value(value)} finite_fraction={(finite_counts[channel] / total_counts[channel]) if total_counts[channel] else 0.0}")
            elif dataset.dtype.kind == "b":
                lines.append(f"    {name}: true_fraction={(sums[channel] / total_counts[channel]) if total_counts[channel] else 0.0}")
            else:
                lines.append(f"    {name}: mean={_format_value(sums[channel] / finite_counts[channel] if finite_counts[channel] else np.nan)}")


def _append_qc_mask_details(lines: list[str], handle: h5py.File) -> None:
    """输出单个 HDF5 record 的 invalid/flat/extreme 实际掩码内容。

    三类 QC 数据集通常是 ``(epoch, channel, subwindow)``，其中每个
    subwindow 对应一个 5 秒窗口。这里不读取 waveform，只逐 dataset、逐
    chunk 扫描 QC 掩码，并把每个通道为 True 的 ``(epoch, subwindow)``
    位置完整列出；没有列出的窗口值即为 False。
    """

    lines.append("QC MASK CONTENTS")
    lines.append("说明: true_positions 列出值为 True 的(epoch, subwindow)；subwindow=0..5对应每个epoch内的5秒窗口。")
    for modality in MODALITY_NAMES:
        lines.append(f"modality: {modality}")
        channel_count = 0
        try:
            present_path = f"quality/channel_present/{modality}"
            if present_path in handle:
                channel_count = int(handle[present_path].shape[0])
            else:
                signal_path = f"signals/{modality}"
                if signal_path in handle and handle[signal_path].ndim >= 2:
                    channel_count = int(handle[signal_path].shape[1])
            names = _channel_names(handle, modality, channel_count)
        except Exception as exc:
            lines.append(f"  ERROR channel metadata: {type(exc).__name__}: {exc}")
            names = []

        for qc_name in QC_MASK_NAMES:
            path = f"quality/{qc_name}/{modality}"
            if path not in handle:
                lines.append(f"  {path}: MISSING")
                continue
            dataset = handle[path]
            lines.append(f"  {path}:")
            lines.append(f"    shape: {dataset.shape}")
            lines.append(f"    dtype: {dataset.dtype}")
            if dataset.ndim == 0:
                value = bool(dataset[()])
                lines.append(f"    value: {value}")
                continue

            channels = int(dataset.shape[1]) if dataset.ndim >= 2 else 1
            if not names or len(names) != channels:
                candidate_names = _channel_names(handle, modality, channels)
                names = candidate_names if len(candidate_names) == channels else [
                    f"channel_{index}" for index in range(channels)
                ]
            true_counts = np.zeros(channels, dtype=np.int64)
            total_counts = np.zeros(channels, dtype=np.int64)
            true_positions: list[list[tuple[int, ...]]] = [[] for _ in range(channels)]

            # 沿第一个轴分块读取，避免把异常大的 QC dataset 一次性载入内存。
            for selection in _dataset_slices(dataset):
                values = np.asarray(_read_slice(dataset, selection), dtype=bool)
                if dataset.ndim == 1:
                    values = values.reshape(-1, 1)
                for channel in range(channels):
                    channel_values = values[:, channel, ...] if dataset.ndim >= 2 else values[:, 0]
                    flat = channel_values.reshape(-1)
                    true_counts[channel] += int(np.count_nonzero(flat))
                    total_counts[channel] += int(flat.size)
                    if dataset.ndim == 3:
                        for row, subwindows in enumerate(channel_values):
                            for subwindow in np.flatnonzero(subwindows):
                                epoch = int(selection[0].start or 0) + row
                                true_positions[channel].append((epoch, int(subwindow)))
                    elif dataset.ndim == 2:
                        for row in np.flatnonzero(channel_values):
                            epoch = int(selection[0].start or 0) + int(row)
                            true_positions[channel].append((epoch,))
                    else:
                        for offset in np.flatnonzero(flat):
                            true_positions[channel].append((int(selection[0].start or 0) + int(offset),))

            for channel, name in enumerate(names):
                fraction = (true_counts[channel] / total_counts[channel]) if total_counts[channel] else 0.0
                if dataset.ndim == 3:
                    positions = ", ".join(
                        f"(epoch={epoch}, subwindow={subwindow})"
                        for epoch, subwindow in true_positions[channel]
                    )
                else:
                    positions = ", ".join(str(position) for position in true_positions[channel])
                lines.append(f"    {name}:")
                lines.append(f"      true_count: {int(true_counts[channel])}")
                lines.append(f"      total_count: {int(total_counts[channel])}")
                lines.append(f"      true_fraction: {fraction}")
                lines.append(f"      true_positions: [{positions}]")
        lines.append("")


def _caisr_details(lines: list[str], dataset_path: str, dataset: h5py.Dataset) -> None:
    name = dataset_path.rsplit("/", 1)[-1]
    if not dataset_path.startswith("/annotations/caisr/"):
        return
    if name in {"stage", "respiratory", "limb", "arousal"} and dataset.dtype.kind in "biu":
        values = _numeric_stats(dataset)
        if values and "unique_counts" in values:
            lines.append("  CAISR code counts:")
            for code in ([-1, 0, 1, 2, 3, 4, 5] if name == "respiratory" else [-1, 0, 1, 2] if name == "limb" else [-1, 0, 1] if name == "arousal" else [-1, 1, 2, 3, 4, 5]):
                lines.append(f"    {code}: {values['unique_counts'].get(code, 0)}")
    if name == "stage_probability":
        try:
            values = np.asarray(dataset[:], dtype=np.float64)
            lines.append("  CAISR stage probability columns:")
            for index, column_name in enumerate(STAGE_PROBABILITY_NAMES[: values.shape[1]]):
                column = values[:, index]
                finite = np.isfinite(column)
                lines.append(f"    {column_name}: finite_min={_format_value(column[finite].min() if finite.any() else np.nan)} finite_max={_format_value(column[finite].max() if finite.any() else np.nan)} mean={_format_value(column[finite].mean() if finite.any() else np.nan)} nan_count={int(np.isnan(column).sum())}")
        except Exception as exc:
            lines.append(f"  ERROR CAISR stage_probability details: {type(exc).__name__}: {exc}")


def _inspect_dataset(lines: list[str], handle: h5py.File, path: str, dataset: h5py.Dataset) -> None:
    path = "/" + path.lstrip("/")
    lines.append(f"DATASET: {path}")
    lines.append(f"shape: {dataset.shape}")
    lines.append(f"dtype: {dataset.dtype}")
    lines.append(f"chunks: {dataset.chunks}")
    lines.append(f"compression: {dataset.compression}")
    lines.append(f"compression_opts: {dataset.compression_opts}")
    lines.append(f"fillvalue: {_format_value(dataset.fillvalue)}")
    lines.append(f"storage_size: {dataset.id.get_storage_size()}")
    lines.append("attrs:")
    _append_attrs(lines, dataset.attrs)
    try:
        if int(dataset.size) <= SMALL_DATASET_LIMIT:
            lines.append(f"data: {_format_value(_read_slice(dataset, () if dataset.ndim == 0 else (slice(None),) * dataset.ndim))}")
        stats = _numeric_stats(dataset)
        if stats is not None:
            lines.append("statistics:")
            _append_stats(lines, stats)
        elif int(dataset.size) > SMALL_DATASET_LIMIT:
            lines.append("statistics: unavailable for non-numeric large dataset")
        if path.startswith("/signals/"):
            _waveform_details(lines, handle, path, dataset)
        if path.startswith("/quality/"):
            _quality_details(lines, handle, path, dataset)
        _caisr_details(lines, path, dataset)
    except Exception as exc:
        lines.append(f"ERROR inspecting data: {type(exc).__name__}: {exc}")
    lines.append("")


def inspect_cache(input_path: Path, output_path: Path, *, include_qc: bool = False) -> list[str]:
    resolved_input = input_path.expanduser().resolve()
    resolved_output = output_path.expanduser().resolve()
    if resolved_input == resolved_output:
        raise ValueError("--output must not overwrite the HDF5 input")
    lines: list[str] = []
    lines.append("CACHE INSPECTION")
    lines.append(f"absolute_path: {resolved_input}")
    lines.append(f"file_size: {resolved_input.stat().st_size}")
    lines.append("")
    with h5py.File(resolved_input, "r") as handle:
        lines.append(f"root_keys: {_format_value(list(handle.keys()))}")
        lines.append("root_attrs:")
        _append_attrs(lines, handle.attrs)
        lines.append("")
        lines.append("TREE")
        lines.append("GROUP: /")
        lines.append("attrs:")
        _append_attrs(lines, handle.attrs)
        lines.append("")

        def visitor(name: str, item: h5py.Group | h5py.Dataset) -> None:
            try:
                if isinstance(item, h5py.Group):
                    lines.append(f"GROUP: /{name}")
                    lines.append("attrs:")
                    _append_attrs(lines, item.attrs)
                    lines.append("")
                else:
                    _inspect_dataset(lines, handle, name, item)
            except Exception as exc:
                lines.append(f"ERROR visiting /{name}: {type(exc).__name__}: {exc}")
                lines.append("")

        handle.visititems(visitor)
        if include_qc:
            _append_qc_mask_details(lines, handle)
    lines.append("Inspection complete")
    lines.append(f"Input: {resolved_input}")
    lines.append(f"Report: {resolved_output}")
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    resolved_output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hdf5_path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--print-qc",
        "--qc",
        action="store_true",
        help="在终端和报告中打印 invalid/flat/extreme 三类逐5秒窗口掩码",
    )
    args = parser.parse_args()
    lines = inspect_cache(args.hdf5_path, args.output, include_qc=args.print_qc)
    if args.print_qc:
        try:
            start = lines.index("QC MASK CONTENTS")
            end = lines.index("Inspection complete", start)
            print("\n".join(lines[start:end]).rstrip())
        except ValueError:
            print("QC MASK CONTENTS unavailable")
    print("Inspection complete")
    print(f"Input: {args.hdf5_path.expanduser().resolve()}")
    print(f"Report: {args.output.expanduser().resolve()}")


if __name__ == "__main__":
    main()


# cd /database/home/gaohaojie/workspace/psg

# PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
# /database/home/gaohaojie/.conda/envs/sleepfm_env/bin/python cache_print.py \
#   waveform_cache/I0002/sub-I0002150002662_ses-1.h5 \
#   --output cache_print.txt


# cd /database/home/gaohaojie/workspace/psg

# PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
# /database/home/gaohaojie/.conda/envs/sleepfm_env/bin/python \
# cache_print.py \
# waveform_cache/I0002/sub-I0002150002662_ses-1.h5 \
# --output cache_print.txt \
# --print-qc