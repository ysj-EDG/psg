"""6600 条 PSG record 的 SpO2 原始数值、单位和采样率审计器。

本脚本只读取每条 record 的 EDF header，并按照已经冻结的 canonical
source-plan 定位 SPO2 selected raw channel。对 present record 只调用一次
selected SpO2 的 readSignal，直接在 EDF 原始数值上计算统计；不执行
scale conversion、resample、interpolation、clipping、normalization、QC
或任何正式 cache 写入。
"""

from __future__ import annotations

import argparse
import csv
import math
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from channel_sources import SourcePlan, build_source_plans
from waveform_cache import DEFAULT_PSG_ROOT, EdfHeader, iter_psg_records, read_edf_header


EXPECTED_RECORDS = 6600
DEFAULT_SAMPLE_PER_SITE = 3
AUDIT_SITES = ("I0002", "I0006", "S0001")
MAX_WORKERS = 8
OUTPUT_DIR = Path("/database/home/gaohaojie/workspace/psg/spo2_audit_output")
CSV_PATH = OUTPUT_DIR / "spo2_audit.csv"
SUMMARY_PATH = OUTPUT_DIR / "spo2_audit_summary.txt"
SPO2_CANONICAL = "SPO2"
EXTREME_MIN = -1.0
EXTREME_MAX = 110.0
COMMON_PHYSICAL_DIMENSIONS = {"%", "percent", "percentage"}

CSV_FIELDS = (
    "record_id",
    "site",
    "channel_present",
    "selected_source_label",
    "physical_dimension",
    "native_fs",
    "native_sample_count",
    "finite_fraction",
    "nan_count",
    "posinf_count",
    "neginf_count",
    "zero_fraction",
    "min",
    "p001",
    "p01",
    "p05",
    "p50",
    "p95",
    "p99",
    "p999",
    "max",
    "fraction_in_0_1_5",
    "fraction_in_50_100_5",
    "scale_hint",
    "actual_sample_count",
    "error",
)


def _percentile(values: np.ndarray, percentile: float) -> float:
    finite = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    return float(np.percentile(finite, percentile))


def summarize_raw_spo2(values: Sequence[float] | np.ndarray) -> dict[str, float | int]:
    """在不改变输入数据的前提下计算 SpO2 原始数值统计。"""

    values = np.asarray(values, dtype=np.float64).reshape(-1)
    finite_mask = np.isfinite(values)
    finite = values[finite_mask]
    size = int(values.size)
    finite_count = int(finite_mask.sum())
    zero_count = int(np.count_nonzero(values == 0.0))
    denominator = float(finite_count) if finite_count else 0.0
    in_fraction = (
        float(np.count_nonzero((finite >= 0.0) & (finite <= 1.5)) / denominator)
        if denominator
        else 0.0
    )
    in_percent = (
        float(np.count_nonzero((finite >= 50.0) & (finite <= 100.5)) / denominator)
        if denominator
        else 0.0
    )
    return {
        "finite_fraction": float(finite_count / size) if size else 0.0,
        "nan_count": int(np.isnan(values).sum()),
        "posinf_count": int(np.isposinf(values).sum()),
        "neginf_count": int(np.isneginf(values).sum()),
        "zero_fraction": float(zero_count / size) if size else 0.0,
        "min": _percentile(values, 0.0),
        "p001": _percentile(values, 0.1),
        "p01": _percentile(values, 1.0),
        "p05": _percentile(values, 5.0),
        "p50": _percentile(values, 50.0),
        "p95": _percentile(values, 95.0),
        "p99": _percentile(values, 99.0),
        "p999": _percentile(values, 99.9),
        "max": _percentile(values, 100.0),
        "fraction_in_0_1_5": in_fraction,
        "fraction_in_50_100_5": in_percent,
    }


def classify_scale_hint(finite_median: float) -> str:
    """按暂定 audit 规则分类；分类不参与任何数据变换。"""

    if math.isfinite(float(finite_median)):
        if 0.5 <= finite_median <= 1.5:
            return "fraction_like"
        if 50.0 <= finite_median <= 100.5:
            return "percent_like"
    return "other"


def is_integer_like(value: float, *, atol: float = 1e-6) -> bool:
    """判断 native sampling rate 是否等价于整数采样率。"""

    return math.isfinite(float(value)) and math.isclose(
        float(value), round(float(value)), abs_tol=atol, rel_tol=0.0
    )


def is_unusual_physical_dimension(value: str) -> bool:
    """仅用于报告：把非空且非常见百分比单位列为 unusual。"""

    normalized = str(value).strip().casefold()
    return normalized not in COMMON_PHYSICAL_DIMENSIONS


def _empty_stats() -> dict[str, float | int]:
    return summarize_raw_spo2(np.array([], dtype=np.float64))


def _base_row(record_id: str, site: str) -> dict[str, object]:
    row: dict[str, object] = {
        "record_id": record_id,
        "site": site,
        "channel_present": False,
        "selected_source_label": "",
        "physical_dimension": "",
        "native_fs": float("nan"),
        "native_sample_count": -1,
        "scale_hint": "missing",
        "actual_sample_count": 0,
        "error": "",
    }
    row.update(_empty_stats())
    row["_raw_stats"] = dict(row)
    return row


def _read_selected_spo2(
    *,
    psg_path: Path,
    plan: SourcePlan,
) -> np.ndarray:
    """只读取 SPO2 selected source index，不遍历其它 modality。"""

    if len(plan.selected_source_indices) != 1:
        raise ValueError(
            f"SPO2 selected source count must be one: {plan.selected_source_indices}"
        )
    import pyedflib

    reader = pyedflib.EdfReader(str(psg_path))
    try:
        return np.asarray(
            reader.readSignal(int(plan.selected_source_indices[0])),
            dtype=np.float64,
        ).reshape(-1)
    finally:
        reader.close()


def audit_record(
    *,
    record_id: str,
    site: str,
    psg_path: Path,
    header: EdfHeader,
) -> dict[str, object]:
    """审计单条 record 的 SPO2 source 和原始数组。"""

    plans = build_source_plans(
        header.channel_names,
        header.native_sampling_rates_hz,
        header.native_sample_counts,
        header.physical_dimensions,
    )
    plan = plans[SPO2_CANONICAL]
    row = _base_row(record_id, site)
    if not plan.channel_present:
        row["_raw_stats"] = dict(row)
        return row

    values = _read_selected_spo2(psg_path=psg_path, plan=plan)
    stats = summarize_raw_spo2(values)
    row.update(
        {
            "channel_present": True,
            "selected_source_label": "|".join(plan.selected_source_labels),
            "physical_dimension": plan.physical_dimension,
            "native_fs": plan.native_sampling_rate_hz,
            "native_sample_count": plan.native_sample_count,
            "scale_hint": classify_scale_hint(float(stats["p50"])),
            "actual_sample_count": int(values.size),
        }
    )
    row.update(stats)
    row["_raw_stats"] = dict(stats)
    return row


def _error_row(record_id: str, site: str, error: Exception) -> dict[str, object]:
    row = _base_row(record_id, site)
    row["scale_hint"] = "error"
    row["error"] = f"{type(error).__name__}: {error}"
    row["_raw_stats"] = dict(row)
    return row


def _format_number(value: object) -> str:
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            return "nan"
        return f"{float(value):.10g}"
    return str(value)


def _csv_row(row: Mapping[str, object]) -> dict[str, object]:
    return {field: _format_number(row.get(field, "")) for field in CSV_FIELDS}


def write_csv(rows: Sequence[Mapping[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(_csv_row(row) for row in rows)


def _present_rows(rows: Iterable[Mapping[str, object]]) -> list[Mapping[str, object]]:
    return [
        row
        for row in rows
        if bool(row.get("channel_present")) and not row.get("error")
    ]


def _site_counts(rows: Sequence[Mapping[str, object]], site: str) -> tuple[int, int]:
    site_rows = [row for row in rows if row.get("site") == site]
    present = sum(bool(row.get("channel_present")) for row in site_rows)
    return present, len(site_rows) - present


def _site_distribution(rows: Sequence[Mapping[str, object]]) -> str:
    counts = Counter(str(row.get("site", "")) for row in rows)
    return ", ".join(f"{site}:{counts[site]}" for site in sorted(counts)) or "none"


def _example_lines(
    title: str,
    rows: Sequence[Mapping[str, object]],
    *,
    limit: int = 20,
) -> list[str]:
    lines = [title]
    if not rows:
        lines.append("  none")
        return lines
    for row in rows[:limit]:
        stats = row.get("_raw_stats", {})
        lines.append(
            "  "
            f"{row.get('record_id')} | site={row.get('site')} | "
            f"unit={row.get('physical_dimension') or '<empty>'} | "
            f"fs={_format_number(row.get('native_fs'))} | "
            f"scale_hint={row.get('scale_hint')} | "
            f"median={_format_number(stats.get('p50'))} | "
            f"min={_format_number(stats.get('min'))} | "
            f"max={_format_number(stats.get('max'))} | "
            f"nan={stats.get('nan_count', 0)} | "
            f"+inf={stats.get('posinf_count', 0)} | "
            f"-inf={stats.get('neginf_count', 0)}"
        )
    if len(rows) > limit:
        lines.append(f"  ... {len(rows) - limit} more")
    return lines


def build_summary(
    rows: Sequence[Mapping[str, object]],
    *,
    expected_records: int,
    scope_description: str = "all discovered records",
) -> str:
    """构造完整的 raw SpO2 audit 汇总文本。"""

    present_rows = _present_rows(rows)
    missing_count = sum(not bool(row.get("channel_present")) for row in rows)
    error_rows = [row for row in rows if row.get("error")]
    scale_counts = Counter(str(row.get("scale_hint")) for row in present_rows)
    lines = [
        "SpO2 raw scale / unit / sampling-rate audit",
        f"scope: {scope_description}",
        "",
        "Overall",
        f"records scanned: {len(rows)}",
        f"expected records: {expected_records}",
        f"record_count_matches_expected: {len(rows) == expected_records}",
        f"scan_successful: {not error_rows}",
        f"records_with_errors: {len(error_rows)}",
        f"SpO2 present: {len(present_rows)}",
        f"SpO2 missing: {missing_count}",
        f"fraction_like count: {scale_counts.get('fraction_like', 0)}",
        f"percent_like count: {scale_counts.get('percent_like', 0)}",
        f"other count: {scale_counts.get('other', 0)}",
        "",
        "By site",
    ]
    sites = sorted(set(["I0002", "I0006", "S0001"]) | {str(row.get("site")) for row in rows})
    for site in sites:
        present, missing = _site_counts(rows, site)
        site_scale = Counter(
            str(row.get("scale_hint"))
            for row in present_rows
            if row.get("site") == site
        )
        lines.append(
            f"{site}: present={present} missing={missing} "
            f"fraction_like={site_scale.get('fraction_like', 0)} "
            f"percent_like={site_scale.get('percent_like', 0)} "
            f"other={site_scale.get('other', 0)}"
        )

    lines.extend(["", "physical_dimension"])
    dimension_counts = Counter(str(row.get("physical_dimension", "")) for row in present_rows)
    dimension_sites: dict[str, list[Mapping[str, object]]] = {}
    for row in present_rows:
        dimension_sites.setdefault(str(row.get("physical_dimension", "")), []).append(row)
    for dimension in sorted(dimension_counts):
        lines.append(
            f"{dimension or '<empty>'}: record_count={dimension_counts[dimension]} "
            f"sites={_site_distribution(dimension_sites[dimension])}"
        )

    lines.extend(["", "native sampling rate"])
    fs_rows: dict[str, list[Mapping[str, object]]] = {}
    for row in present_rows:
        key = _format_number(row.get("native_fs"))
        fs_rows.setdefault(key, []).append(row)
    for fs in sorted(fs_rows, key=lambda value: float(value) if value != "nan" else float("inf")):
        lines.append(
            f"{fs} Hz: record_count={len(fs_rows[fs])} "
            f"sites={_site_distribution(fs_rows[fs])}"
        )

    lines.extend(["", "unit × numeric scale"])
    unit_scale = Counter(
        (
            str(row.get("site")),
            str(row.get("physical_dimension", "")),
            str(row.get("scale_hint")),
        )
        for row in present_rows
    )
    lines.append("site | physical_dimension | scale_hint | record_count")
    for (site, dimension, scale), count in sorted(unit_scale.items()):
        lines.append(f"{site} | {dimension or '<empty>'} | {scale} | {count}")

    non_integer_fs = [
        row for row in present_rows if not is_integer_like(float(row.get("native_fs", float("nan"))))
    ]
    lines.extend(["", "Matcha-compatible 1 Hz downsampling applicability"])
    if not non_integer_fs:
        lines.append("all present SpO2 native_fs are integer-like")
        lines.append("per-second reshape + nanmedian is structurally applicable")
    else:
        lines.append("per-second reshape + nanmedian is not uniformly applicable")
        fs_counter = Counter(_format_number(row.get("native_fs")) for row in non_integer_fs)
        for fs, count in sorted(fs_counter.items()):
            examples = ", ".join(
                str(row.get("record_id"))
                for row in non_integer_fs
                if _format_number(row.get("native_fs")) == fs
            )
            lines.append(f"non-integer fs={fs}: count={count} examples={examples}")

    unusual_unit_rows = [
        row for row in present_rows
        if is_unusual_physical_dimension(str(row.get("physical_dimension", "")))
    ]
    nonfinite_rows = [
        row for row in present_rows
        if any(int(row.get(name, 0)) > 0 for name in ("nan_count", "posinf_count", "neginf_count"))
    ]
    other_rows = [row for row in present_rows if row.get("scale_hint") == "other"]
    extreme_rows = [
        row for row in present_rows
        if float(row.get("min", float("nan"))) < EXTREME_MIN
        or float(row.get("max", float("nan"))) > EXTREME_MAX
    ]
    lines.extend(["", "Unusual examples (maximum 20 each)"])
    lines.extend(_example_lines(f"scale_hint=other (count={len(other_rows)})", other_rows))
    lines.extend(
        _example_lines(
            f"unusual physical_dimension (count={len(unusual_unit_rows)})",
            unusual_unit_rows,
        )
    )
    lines.extend(_example_lines(f"NaN/Inf exists (count={len(nonfinite_rows)})", nonfinite_rows))
    lines.extend(
        _example_lines(
            f"median not in either normal class (count={len(other_rows)})",
            other_rows,
        )
    )
    lines.extend(
        _example_lines(
            f"extreme min/max outside [{EXTREME_MIN}, {EXTREME_MAX}] (count={len(extreme_rows)})",
            extreme_rows,
        )
    )
    lines.extend(["", "Audit semantics"])
    lines.append("all statistics use EDF raw numeric values; no scale conversion was applied")
    lines.append("no resampling, interpolation, clipping, normalization, QC, or cache write was performed")
    if error_rows:
        lines.extend(_example_lines("record errors", error_rows))
    return "\n".join(lines) + "\n"


def _audit_record_entry(entry: tuple[str, str, Path]) -> tuple[dict[str, object], str]:
    """在线程 worker 中完成一条 record 的 header/source/raw SpO2 审计。"""

    site, record_id, psg_path = entry
    try:
        header = read_edf_header(psg_path)
        row = audit_record(
            record_id=record_id,
            site=site,
            psg_path=psg_path,
            header=header,
        )
        status = "present" if row["channel_present"] else "missing"
    except Exception as exc:
        row = _error_row(record_id, site, exc)
        status = f"ERROR {row[error]}"
    return row, status


def select_records(
    records: Sequence[tuple[str, str, Path]],
    *,
    sites: Sequence[str] = AUDIT_SITES,
    per_site: int = DEFAULT_SAMPLE_PER_SITE,
) -> tuple[tuple[str, str, Path], ...]:
    """按稳定文件顺序为每个站点选择固定数量的 record。"""

    if per_site <= 0:
        raise ValueError("per_site must be positive")
    selected: list[tuple[str, str, Path]] = []
    for site in sites:
        site_records = [record for record in records if record[0] == site]
        selected.extend(site_records[:per_site])
    return tuple(selected)


def run_audit(
    *,
    psg_root: Path = DEFAULT_PSG_ROOT,
    output_dir: Path = OUTPUT_DIR,
    expected_records: int = EXPECTED_RECORDS,
    max_workers: int = MAX_WORKERS,
    record_subset: Sequence[tuple[str, str, Path]] | None = None,
    scope_description: str = "all discovered records",
) -> tuple[Path, Path, list[dict[str, object]]]:
    """扫描指定 record 集合并写出 CSV 与 summary。"""

    records = tuple(record_subset) if record_subset is not None else iter_psg_records(psg_root)
    rows: list[dict[str, object]] = []
    print(
        f"SpO2 audit: selected {len(records)} PSG records; expected {expected_records}; "
        f"workers={max_workers}; scope={scope_description}"
    )
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        outcomes = executor.map(_audit_record_entry, records)
        for position, ((site, record_id, _), (row, status)) in enumerate(
            zip(records, outcomes), start=1
        ):
            rows.append(row)
            if len(records) <= 20 or position % 100 == 0 or status.startswith("ERROR"):
                print(f"[{position}/{len(records)}] {record_id} {status}", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "spo2_audit.csv"
    summary_path = output_dir / "spo2_audit_summary.txt"
    write_csv(rows, csv_path)
    summary_path.write_text(
        build_summary(
            rows,
            expected_records=expected_records,
            scope_description=scope_description,
        ),
        encoding="utf-8",
    )
    print(f"wrote {csv_path}")
    print(f"wrote {summary_path}")
    return csv_path, summary_path, rows

def main() -> None:
    """默认对三个站点各抽取 3 条 record；--full 才运行全量集合。"""

    parser = argparse.ArgumentParser(description="SpO2 raw scale/unit audit")
    parser.add_argument(
        "--full",
        action="store_true",
        help="scan all discovered records instead of the default 3 records per site",
    )
    parser.add_argument(
        "--sample-per-site",
        type=int,
        default=DEFAULT_SAMPLE_PER_SITE,
        help="number of stable records selected from each audit site (default: 3)",
    )
    args = parser.parse_args()
    records = iter_psg_records(DEFAULT_PSG_ROOT)
    if args.full:
        run_audit(
            record_subset=records,
            expected_records=EXPECTED_RECORDS,
            scope_description="full discovered record set",
        )
        return
    selected = select_records(
        records, sites=AUDIT_SITES, per_site=args.sample_per_site
    )
    run_audit(
        record_subset=selected,
        expected_records=len(AUDIT_SITES) * args.sample_per_site,
        scope_description=(
            f"stable first {args.sample_per_site} records per site: "
            + ", ".join(AUDIT_SITES)
        ),
    )


if __name__ == "__main__":
    main()
