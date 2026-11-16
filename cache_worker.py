"""批处理与本地落盘共用的单 record 处理逻辑。

所有 preprocessing、schema、quality、CAISR 语义仍由 ``waveform_cache`` /
``waveform_preprocessing`` 提供；本模块只负责：

- 单 record 的 build + 正式 validator + raw nonfinite 检查 + 原子输出；
- 失败时删除临时 HDF5 并重新抛出异常。
"""

from __future__ import annotations

import os
from pathlib import Path

import h5py
import numpy as np

from waveform_cache import (
    CANONICAL_CHANNEL_NAMES,
    MODALITY_ORDER,
    build_waveform_cache,
    validate_caisr_annotations,
)


def validate_waveform_tree(path: Path | str, record_id: str, site: str) -> None:
    """对当前 writer 的 signals/quality 结果做最小结构和 finite 检查。"""

    expected_rates = {
        "eeg": 128,
        "eog": 128,
        "ecg": 200,
        "emg": 200,
        "resp": 25,
        "spo2": 1,
    }
    with h5py.File(path, "r") as handle:
        epochs = int(handle.attrs["complete_epoch_count"])
        if handle.attrs["record_id"] != record_id:
            raise ValueError("record_id does not match the requested record")
        if handle.attrs["site"] != site:
            raise ValueError("site does not match the requested record")
        for modality in MODALITY_ORDER:
            dataset = handle[f"signals/{modality}"]
            channels = len(CANONICAL_CHANNEL_NAMES[modality])
            expected_shape = (epochs, channels, 30 * expected_rates[modality])
            if dataset.shape != expected_shape:
                raise ValueError(
                    f"{modality} shape mismatch: {dataset.shape} != {expected_shape}"
                )
            if dataset.dtype != np.dtype("float32"):
                raise ValueError(f"{modality} dtype is {dataset.dtype}, not float32")
            if np.isinf(dataset[:]).any():
                raise ValueError(f"{modality} contains +/-Inf")
            present = np.asarray(handle[f"quality/channel_present/{modality}"][:], dtype=bool)
            values = dataset[:]
            for channel_index, is_present in enumerate(present):
                finite = np.isfinite(values[:, channel_index, :])
                if is_present and not finite.all():
                    raise ValueError(
                        f"present {modality}[{channel_index}] contains nonfinite output"
                    )
                if not is_present and finite.any():
                    raise ValueError(
                        f"structurally missing {modality}[{channel_index}] is not all NaN"
                    )


def check_raw_nonfinite(result: object) -> None:
    """普通模态 selected raw source 一旦含 NaN/Inf 就失败（SPO2 除外）。"""

    inspections = getattr(result, "source_inspections", ())
    failures = []
    for inspection in inspections:
        if inspection.canonical_name == "SPO2":
            continue
        if inspection.has_nonfinite:
            failures.append(
                {
                    "canonical": inspection.canonical_name,
                    "raw_label": inspection.raw_label,
                    "source_index": inspection.source_index,
                    "nan_count": inspection.nan_count,
                    "posinf_count": inspection.posinf_count,
                    "neginf_count": inspection.neginf_count,
                }
            )
    if failures:
        raise RuntimeError(f"selected ordinary raw source has NaN/Inf: {failures}")


def process_one_record(
    psg_path: Path | str,
    caisr_path: Path | str | None,
    record_id: str,
    site: str,
    tmp_path: Path | str,
    final_path: Path | str,
) -> None:
    """处理单条 record：tmp 写入 → 正式 validator → 原子 rename 为 final。

    任何异常都删除 tmp 并重新抛出；绝不留下半成品 HDF5 冒充成功文件。
    """

    psg_path = Path(psg_path)
    caisr_path = Path(caisr_path) if caisr_path else None
    tmp_path = Path(tmp_path)
    final_path = Path(final_path)

    if final_path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {final_path}")
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    if tmp_path.exists():
        tmp_path.unlink()
    try:
        output, result = build_waveform_cache(
            psg_path,
            caisr_path,
            tmp_path,
            record_id=record_id,
            site=site,
        )
        check_raw_nonfinite(result)
        validate_caisr_annotations(output)
        validate_waveform_tree(output, record_id, site)
        if final_path.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {final_path}")
        os.replace(tmp_path, final_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
