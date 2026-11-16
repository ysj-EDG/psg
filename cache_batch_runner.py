"""本地 batch runner：多个 spawn worker 并行提取，record 级失败隔离。

只负责一个 batch manifest：

- 所有路径锚定 ``Path(__file__).resolve().parent``（当前工作区根）
- BLAS 线程数在 import numpy 之前固定为 1，避免 12 进程各自再开大量线程
- 单 record 异常只记录失败并继续，batch 正常结束 exit 0
- 只有基础设施级错误（manifest 缺失、输入系统性缺失、进程池崩溃）才非0退出
- 每条 record 仍是 tmp → validators → os.replace 原子输出
"""

from __future__ import annotations

import os

# 必须在 import numpy/waveform_cache 之前设置，spawn 的 worker 也会在 re-import
# 本模块时同样先执行这里。
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import csv
import json
import multiprocessing
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from cache_worker import process_one_record


SCRIPT_DIR = Path(__file__).resolve().parent


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="batch manifest TSV (workspace 相对)")
    parser.add_argument("--output-root", default="result_bundle", help="输出根（workspace 相对）")
    parser.add_argument("--workers", type=int, default=12)
    return parser.parse_args(argv)


def _load_batch_manifest(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            rows.append(
                {
                    "index": row["index"],
                    "site": row["site"],
                    "record_id": row["record_id"],
                    "psg_path": row["psg_path"],
                    "caisr_path": row["caisr_path"],
                }
            )
    if not rows:
        raise RuntimeError(f"batch manifest is empty: {path}")
    return rows


def _run_record(job: dict) -> dict:
    """单个 record 的隔离入口；任何异常都只转为 FAILED 结果。"""

    started = time.monotonic()
    base = {
        "index": job["index"],
        "record_id": job["record_id"],
        "site": job["site"],
    }
    try:
        process_one_record(
            job["psg"],
            job["caisr"],
            job["record_id"],
            job["site"],
            job["tmp"],
            job["final"],
        )
        output_bytes = Path(job["final"]).stat().st_size
        return {
            **base,
            "status": "SUCCESS",
            "output_bytes": output_bytes,
            "elapsed_sec": round(time.monotonic() - started, 2),
        }
    except Exception as exc:
        return {
            **base,
            "status": "FAILED",
            "exception_type": type(exc).__name__,
            "exception_message": str(exc)[:4000],
            "traceback": traceback.format_exc(),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "output_bytes": "",
            "elapsed_sec": round(time.monotonic() - started, 2),
        }


def _write_tsv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})
    os.replace(tmp, path)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(sys.argv[1:] if argv is None else argv))

    # 尽早创建 output root，确保批处理结果始终有明确落点。
    if os.path.isabs(args.output_root):
        print(f"output_root must be workspace-relative: {args.output_root}", flush=True)
        return 1
    output_root = SCRIPT_DIR / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"workers: {args.workers}", flush=True)

    manifest_path = SCRIPT_DIR / args.manifest
    try:
        rows = _load_batch_manifest(manifest_path)
    except Exception as exc:
        print(f"batch manifest unreadable: {exc}", flush=True)
        traceback.print_exc()
        return 1

    jobs = []
    for row in rows:
        final = output_root / row["site"] / f"{row['record_id']}.h5"
        tmp = output_root / row["site"] / f".{row['record_id']}.h5.tmp"
        jobs.append(
            {
                **row,
                "psg": str(SCRIPT_DIR / row["psg_path"]),
                "caisr": str(SCRIPT_DIR / row["caisr_path"]) if row["caisr_path"] else None,
                "tmp": str(tmp),
                "final": str(final),
            }
        )

    # 系统性输入缺失属于基础设施错误，必须在进程池之前整体失败，
    # 不能伪装成大批 record-level failure。
    missing = [
        job
        for job in jobs
        if not Path(job["psg"]).is_file()
        or (job["caisr"] is not None and not Path(job["caisr"]).is_file())
    ]
    if missing:
        for job in missing:
            print(
                f"[INFRA] missing input: {job['record_id']} psg={job['psg']} "
                f"caisr={job['caisr']}",
                flush=True,
            )
        print(f"[INFRA] {len(missing)}/{len(jobs)} inputs missing, aborting batch", flush=True)
        return 1

    skipped = [job for job in jobs if Path(job["final"]).is_file()]
    to_run = [job for job in jobs if job not in skipped]
    if skipped:
        print(f"skipped (final exists): {len(skipped)}", flush=True)
    print(f"to_run: {len(to_run)}", flush=True)

    batch_started = time.monotonic()
    results: dict[str, dict] = {}
    ctx = multiprocessing.get_context("spawn")
    try:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
            futures = {pool.submit(_run_record, job): job for job in to_run}
            for future in as_completed(futures):
                job = futures[future]
                result = future.result()
                results[job["index"]] = result
                print(
                    f"[{result['status']}] {job['record_id']} "
                    f"elapsed={result['elapsed_sec']}s",
                    flush=True,
                )
    except Exception as exc:
        # BrokenProcessPool / worker 崩溃属于基础设施错误。
        print(f"[INFRA] process pool failure: {exc}", flush=True)
        traceback.print_exc()
        return 1

    for job in skipped:
        results[job["index"]] = {
            "index": job["index"],
            "record_id": job["record_id"],
            "site": job["site"],
            "status": "SKIPPED",
            "output_bytes": Path(job["final"]).stat().st_size,
            "elapsed_sec": 0.0,
        }

    ordered = [results[row["index"]] for row in rows]
    success = sum(1 for r in ordered if r["status"] == "SUCCESS")
    failed = sum(1 for r in ordered if r["status"] == "FAILED")
    skipped_count = sum(1 for r in ordered if r["status"] == "SKIPPED")

    _write_tsv(
        output_root / "batch_status.tsv",
        ["index", "record_id", "site", "status", "output_bytes", "elapsed_sec"],
        ordered,
    )
    _write_tsv(
        output_root / "batch_failures.tsv",
        [
            "index",
            "record_id",
            "site",
            "exception_type",
            "exception_message",
            "timestamp",
            "traceback",
        ],
        [r for r in ordered if r["status"] == "FAILED"],
    )
    summary = {
        "batch_id": Path(args.manifest).stem,
        "total": len(ordered),
        "success": success,
        "failed": failed,
        "skipped": skipped_count,
        "elapsed_sec": round(time.monotonic() - batch_started, 2),
        "workers": args.workers,
    }
    summary_tmp = output_root / "batch_summary.json.tmp"
    with open(summary_tmp, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    os.replace(summary_tmp, output_root / "batch_summary.json")
    print(f"batch_summary: {json.dumps(summary)}", flush=True)

    # 普通 record failure 不影响 batch 的完成状态；失败明细已写入 TSV。
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
