"""直接读取 canonical HDF5 的 CAISR20 有效率审计。

本脚本只访问 manifest 指向的 canonical HDF5，并复用 ``caisr.py`` 的
``extract`` 计算逻辑；不会打开原始 CAISR EDF、人工标注 EDF 或 waveform。
逐条 record 的特征结果只在内存中累计，不保存中间结果。
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np

if __package__ in (None, ""):
    ROOT = Path(__file__).resolve().parents[3]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from feature_extraction.branches.caisr import CAISR20_FEATURE_NAMES, extract
    from feature_extraction.sample_loader import Manifest, SampleContext
else:
    from ...branches.caisr import CAISR20_FEATURE_NAMES, extract
    from ...sample_loader import Manifest, SampleContext


SITES = ("I0002", "I0006", "S0001")


def _new_feature_stats() -> dict[str, Any]:
    return {
        "valid": 0,
        "invalid": 0,
        "reasons": Counter(),
        "examples": [],
        "by_site": {site: {"valid": 0, "invalid": 0, "reasons": Counter()} for site in SITES},
    }


def _new_summary() -> dict[str, Any]:
    return {
        "records_scanned": 0,
        "records_extracted": 0,
        "errors": [],
        "sources": Counter(),
        "sites": Counter(),
        "features": {name: _new_feature_stats() for name in CAISR20_FEATURE_NAMES},
    }


def _record_result(
    summary: dict[str, Any],
    record_id: str,
    site: str,
    source: str,
    values: np.ndarray,
    valid: np.ndarray,
    reasons: np.ndarray,
) -> None:
    """将一条 ``caisr.extract`` 结果折叠到总体计数，不保留逐条中间结果。"""
    del values  # 审计只统计 valid/reason；数值本身不需要保存。
    summary["records_extracted"] += 1
    summary["sources"][source] += 1
    summary["sites"][site] += 1
    valid = np.asarray(valid, dtype=bool).reshape(-1)
    reasons = np.asarray(reasons).reshape(-1)
    if valid.size != len(CAISR20_FEATURE_NAMES) or reasons.size != len(CAISR20_FEATURE_NAMES):
        raise ValueError("CAISR20 result must contain exactly 20 validity/reason entries")
    for index, name in enumerate(CAISR20_FEATURE_NAMES):
        stats = summary["features"][name]
        site_stats = stats["by_site"].setdefault(site, {"valid": 0, "invalid": 0, "reasons": Counter()})
        if bool(valid[index]):
            stats["valid"] += 1
            site_stats["valid"] += 1
            continue
        reason = str(reasons[index]) or "unknown"
        stats["invalid"] += 1
        stats["reasons"][reason] += 1
        site_stats["invalid"] += 1
        site_stats["reasons"][reason] += 1
        if len(stats["examples"]) < 20:
            stats["examples"].append(record_id)


def summarize_records(
    records: Iterable[tuple[str, str, str, np.ndarray, np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    """汇总 ``(record_id, site, source, values, valid, reasons)``，便于单元测试。"""
    summary = _new_summary()
    for record_id, site, source, values, valid, reasons in records:
        summary["records_scanned"] += 1
        _record_result(summary, record_id, site, source, values, valid, reasons)
    return summary


def _pct(value: int, denominator: int) -> str:
    return f"{100.0 * value / denominator:.3f}%" if denominator else "NA"


def render_summary(summary: dict[str, Any], manifest_path: Path, output_path: Path) -> str:
    lines = [
        "CAISR20 特征有效率审计",
        "=" * 72,
        f"manifest: {manifest_path.resolve()}",
        f"output: {output_path.resolve()}",
        "读取方式: 直接打开 manifest 指向的 canonical HDF5，复用 caisr.py 的 extract；不读取原始 CAISR/human EDF。",
        "中间结果: 不保存逐条 feature 输出，只累计有效/无效及原因计数。",
        "",
        "总体",
        "----",
        f"records_scanned: {summary['records_scanned']}",
        f"records_extracted: {summary['records_extracted']}",
        f"record_errors: {len(summary['errors'])}",
        "annotation_source: " + ", ".join(f"{key}={summary['sources'][key]}" for key in sorted(summary["sources"])) if summary["sources"] else "annotation_source: none",
        "",
        "20 个特征总体有效率",
        "--------------------",
        "feature_name | valid | invalid | valid_fraction | invalid_reasons",
    ]
    denominator = int(summary["records_extracted"])
    for name in CAISR20_FEATURE_NAMES:
        stats = summary["features"][name]
        reasons = ", ".join(f"{key}:{value}" for key, value in sorted(stats["reasons"].items())) or "-"
        lines.append(f"{name} | {stats['valid']} | {stats['invalid']} | {_pct(stats['valid'], denominator)} | {reasons}")

    lines.extend(["", "按站点无效特征数量", "--------------------", "site | feature_name | invalid | valid | invalid_fraction | invalid_reasons"])
    for site in SITES:
        site_total = int(summary["sites"].get(site, 0))
        for name in CAISR20_FEATURE_NAMES:
            stats = summary["features"][name]["by_site"].get(site, {"valid": 0, "invalid": 0, "reasons": Counter()})
            reasons = ", ".join(f"{key}:{value}" for key, value in sorted(stats["reasons"].items())) or "-"
            lines.append(f"{site} | {name} | {stats['invalid']} | {stats['valid']} | {_pct(stats['invalid'], site_total)} | {reasons}")

    lines.extend(["", "无效特征示例 record_id（每个特征最多 20 条）", "----------------------------------------"])
    for name in CAISR20_FEATURE_NAMES:
        examples = summary["features"][name]["examples"]
        lines.append(f"{name}: {', '.join(examples) if examples else '-'}")

    lines.extend(["", "读取/提取错误记录", "------------------"])
    if summary["errors"]:
        lines.extend(f"{record_id}: {error}" for record_id, error in summary["errors"])
    else:
        lines.append("none")
    return "\n".join(lines) + "\n"


def run_audit(manifest_path: Path, output_path: Path) -> dict[str, Any]:
    """逐条打开 canonical HDF5，调用 CAISR20 提取并立即折叠计数。"""
    manifest = Manifest.from_tsv(manifest_path)
    summary = _new_summary()
    summary["records_scanned"] = len(manifest.records)
    for record_id, record in manifest.records.items():
        try:
            with SampleContext(record) as context:
                result = extract(context)
            block = result.blocks[0]
            _record_result(
                summary,
                record_id,
                record.site,
                str(result.metadata.get("annotation_source", "unknown")),
                np.asarray(block.values),
                np.asarray(block.valid_mask),
                np.asarray(block.invalid_reason),
            )
        except Exception as exc:  # 一条坏缓存不能阻断其余只读审计
            summary["errors"].append((record_id, f"{type(exc).__name__}: {exc}"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_summary(summary, manifest_path, output_path), encoding="utf-8")
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[3]
    parser.add_argument("--manifest", type=Path, default=root / "cache_manifest.tsv")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("caisr.txt"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary = run_audit(args.manifest.resolve(), args.output.resolve())
    print(f"records_scanned: {summary['records_scanned']}")
    print(f"records_extracted: {summary['records_extracted']}")
    print(f"record_errors: {len(summary['errors'])}")
    print(f"report: {args.output.resolve()}")
    return 0 if not summary["errors"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
