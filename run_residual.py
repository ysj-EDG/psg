#!/usr/bin/env python3
"""Secondary stage: recovered M2 V5 date residual, fitted within each LOSO fold."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from residual_date import _fit_date_rule, _date_adjustment
from run_loso import COUNTS, aggregate, compare_targets
from train_sequence_ci_loso import evaluate_scores

ROUTE = "MATCHA_M2_SECONDARY_V5_DATE_RESIDUAL"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--primary-private-output", type=Path, required=True)
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, default=Path("results/residual_date"))
    args = parser.parse_args()
    sites, summary, predictions = aggregate(args.primary_private_output)
    if any(abs(row["delta"]) > 1e-12 for row in compare_targets(sites, summary)):
        raise RuntimeError("Primary M2 parity must pass before the secondary stage")
    manifest = pd.read_parquet(args.manifest)
    manifest["record_id"] = manifest.record_id.astype(str)
    predictions["record_id"] = predictions.record_id.astype(str)
    if manifest.groupby("SiteID").size().to_dict() != COUNTS:
        raise ValueError("Expected the same Large 6600 cohort")
    joined = predictions.merge(manifest[["record_id", "CreationTime"]], on="record_id", validate="one_to_one")
    if len(joined) != 6600:
        raise ValueError("Primary predictions and residual metadata must align exactly")
    args.private_output.mkdir(parents=True, exist_ok=False)
    if args.results_dir.exists() and any(args.results_dir.iterdir()):
        raise ValueError("Existing secondary results will not be overwritten")
    rows = []
    for site in COUNTS:
        train = manifest.loc[manifest.SiteID.ne(site)]
        test = joined.loc[joined.SiteID.eq(site)].copy()
        rule = _fit_date_rule(train)
        (args.private_output / f"date_rule_{site}.json").write_text(json.dumps(rule, indent=2) + "\n")
        adjustments = np.asarray([_date_adjustment(rule, row) for row in test.to_dict("records")])
        # Exact upstream V5 addition; no weight search, refit of M2, or reranking.
        test["secondary_logit"] = test.m2_logit_ensemble.to_numpy(float) + adjustments
        metrics = evaluate_scores(test, test.secondary_logit.to_numpy(float))
        rows.append({"route": ROUTE, "held_out_site": site,
            "n_records": metrics["records"], "n_positive": metrics["positives"],
            "ac_pair_count": metrics["age_pair_count"], "ac_auroc": metrics["age_conditioned_auroc"],
            "auroc": metrics["auroc"], "auprc": metrics["auprc"]})
        test.to_parquet(args.private_output / f"held_{site}.parquet", index=False)
    sites = pd.DataFrame(rows)
    summary = pd.DataFrame([{"route": ROUTE,
        "small_site_ac_auroc": sites.loc[sites.held_out_site.isin(["I0002", "I0006"]), "ac_auroc"].mean(),
        "macro_ac_auroc": sites.ac_auroc.mean(), "macro_auroc": sites.auroc.mean(),
        "macro_auprc": sites.auprc.mean()}])
    args.results_dir.mkdir(parents=True, exist_ok=True)
    sites.to_csv(args.results_dir / "loso_metrics.csv", index=False)
    summary.to_csv(args.results_dir / "summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
