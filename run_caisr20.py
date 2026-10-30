"""Fresh-run CLI for CAISR20 historical-semantics aggregate scoring."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from per_epoch_features.extensions.caisr20_core_protocol_runner import load_data2_inputs, run_route
from score import score_site


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--summary-cache", type=Path, required=True)
    parser.add_argument("--rules", type=Path, required=True)
    parser.add_argument("--pool-root", type=Path, required=True)
    parser.add_argument("--sample-root", type=Path, required=True)
    parser.add_argument("--seed42-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = run_route(
        manifest_path=args.manifest,
        summary_cache=args.summary_cache,
        rules_path=args.rules,
        pool_root=args.pool_root,
        sample_root=args.sample_root,
        output_root=args.output_root,
        seed42_root=args.seed42_root,
        strict_source_bridge=True,
    )
    data = load_data2_inputs(args.manifest, args.summary_cache)
    manifest = data["manifest"]
    rows = []
    for site in ("I0002", "I0006", "S0001"):
        indices, fused = result["fused"][site]
        expected = np.flatnonzero(data["site"] == site).astype(np.int64)
        if not np.array_equal(indices, expected):
            raise RuntimeError(f"holdout identity mismatch: {site}")
        scored = score_site(manifest.loc[indices, "label"].to_numpy(np.int8), manifest.loc[indices, "age"].to_numpy(np.float64), fused)
        rows.append({"route": "CAISR20_HISTORICAL_SEMANTICS_CORE_PROTOCOL", "held_out_site": site, **scored})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "loso_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    ac = [float(row["ac_auroc"]) for row in rows]
    summary = {
        "route": "CAISR20_HISTORICAL_SEMANTICS_CORE_PROTOCOL",
        "small_site_ac_auroc": (ac[0] + ac[1]) / 2,
        "macro_ac_auroc": sum(ac) / 3,
        "macro_auroc": sum(float(row["auroc"]) for row in rows) / 3,
        "macro_auprc": sum(float(row["auprc"]) for row in rows) / 3,
    }
    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary))
        writer.writeheader(); writer.writerow(summary)
    print({"models_built": result["models_built"], "predictions_built": result["predictions_built"], **summary})


if __name__ == "__main__":
    main()
