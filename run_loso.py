#!/usr/bin/env python3
"""Run the recovered M2 protocol once; publish aggregates only after target parity."""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

from train_sequence_ci_loso import evaluate_scores

ROUTE = "MATCHA_M2_HISTORICAL_INNER1TO4"
SEEDS = (20260806, 20260807, 20260808)
COUNTS = {"I0002": 319, "I0006": 1142, "S0001": 5139}
TARGETS = {"I0002": 0.7481442205726405, "I0006": 0.7393224315186642,
           "S0001": 0.6680414632808643}
E1_SHA = "e9f135c973c0540970f34acbde89096c028eb3a95a0a601d85f21e8f83d36ac0"
KEYS = ["record_id", "patient_id", "SiteID", "_label", "_age"]


def validate_inputs(manifest: Path, embedding_manifest: Path) -> None:
    frame = pd.read_parquet(manifest)
    embeddings = pd.read_parquet(embedding_manifest)
    if frame["record_id"].duplicated().any() or embeddings["record_id"].duplicated().any():
        raise ValueError("Duplicate input record identifiers")
    if frame.groupby("SiteID").size().to_dict() != COUNTS:
        raise ValueError("Training manifest must contain the exact Large 6600 site counts")
    if len(embeddings) != 6600 or set(frame.record_id.astype(str)) != set(embeddings.record_id.astype(str)):
        raise ValueError("Embedding records do not match the complete training cohort")
    for index, rows in embeddings.groupby("shard_index"):
        path = embedding_manifest.parent / "shards" / f"shard_{int(index):04d}.h5"
        with h5py.File(path, "r") as handle:
            if handle.attrs.get("checkpoint_sha256") != E1_SHA:
                raise ValueError("Embedding shard does not identify the historical E1 checkpoint")
            for row in rows.itertuples(index=False):
                group = handle[str(row.group_key)]
                if str(group.attrs["record_id"]) != str(row.record_id):
                    raise ValueError("Embedding group/record alignment mismatch")
                if group["embedding"].shape != (int(row.embedding_count), 192):
                    raise ValueError("Embedding dimensions/count differ from the manifest")
    print(json.dumps({"input_preflight": "passed", "site_counts": COUNTS,
                      "e1_sha256": E1_SHA}, sort_keys=True), flush=True)


def aggregate(private_output: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    merged = None
    columns = []
    for seed in SEEDS:
        column = f"m2_logit_{seed}"
        current = pd.read_parquet(private_output / f"seed_{seed}" / "loso_predictions.parquet")
        current = current[KEYS + ["raw_sequence"]].rename(columns={"raw_sequence": column})
        if len(current) != 6600 or current.record_id.duplicated().any():
            raise ValueError("Each seed must contain exactly one prediction per record")
        merged = current if merged is None else merged.merge(current, on=KEYS, validate="one_to_one")
        columns.append(column)
    if len(merged) != 6600 or merged.groupby("SiteID").size().to_dict() != COUNTS:
        raise ValueError("Seed prediction keys or site counts disagree")
    if not np.isfinite(merged[columns].to_numpy(float)).all():
        raise ValueError("Non-finite prediction; incomplete LOSO is not accepted")
    merged["m2_logit_ensemble"] = merged[columns].mean(axis=1)
    rows = []
    for site in COUNTS:
        held = merged.loc[merged.SiteID.eq(site)]
        metrics = evaluate_scores(held, held.m2_logit_ensemble.to_numpy(float))
        rows.append({"route": ROUTE, "held_out_site": site,
                     "n_records": metrics["records"], "n_positive": metrics["positives"],
                     "ac_pair_count": metrics["age_pair_count"],
                     "ac_auroc": metrics["age_conditioned_auroc"],
                     "auroc": metrics["auroc"], "auprc": metrics["auprc"]})
    sites = pd.DataFrame(rows)
    summary = pd.DataFrame([{"route": ROUTE,
        "small_site_ac_auroc": sites.loc[sites.held_out_site.isin(["I0002", "I0006"]), "ac_auroc"].mean(),
        "macro_ac_auroc": sites.ac_auroc.mean(), "macro_auroc": sites.auroc.mean(),
        "macro_auprc": sites.auprc.mean()}])
    return sites, summary, merged


def compare_targets(sites: pd.DataFrame, summary: pd.DataFrame) -> list[dict]:
    comparisons = []
    for row in sites.itertuples(index=False):
        target = TARGETS[row.held_out_site]
        comparisons.append({"metric": row.held_out_site, "actual": row.ac_auroc,
                            "target": target, "delta": row.ac_auroc - target})
    for key, target in (("small_site_ac_auroc", 0.7437333260456523),
                        ("macro_ac_auroc", 0.7185027051240563)):
        actual = float(summary.iloc[0][key])
        comparisons.append({"metric": key, "actual": actual, "target": target,
                            "delta": actual - target})
    return comparisons


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--embedding-manifest", type=Path, required=True)
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--check-inputs", action="store_true")
    args = parser.parse_args()
    validate_inputs(args.manifest, args.embedding_manifest)
    if args.check_inputs:
        return 0
    # Do not permit historical predictions or a previous run to satisfy this rerun.
    args.private_output.mkdir(parents=True, exist_ok=False)
    if args.results_dir.exists() and any(args.results_dir.iterdir()):
        raise ValueError("Choose an empty results directory; existing results will not be overwritten")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; do not silently switch the reproduction to CPU")
    context = {"python": platform.python_version(), "torch": torch.__version__,
               "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
               "numpy": np.__version__, "pandas": pd.__version__, "seeds": SEEDS,
               "commands": []}
    entry = Path(__file__).resolve().with_name("train_sequence_ci_loso.py")
    for seed in SEEDS:
        command = [sys.executable, str(entry), "--manifest", str(args.manifest.resolve()),
            "--embedding-manifest", str(args.embedding_manifest.resolve()), "--seed", str(seed),
            "--batch-size", "16", "--num-workers", "4", "--learning-rate", "1e-4",
            "--weight-decay", "1e-4", "--max-epochs", "4", "--minimum-epochs", "4",
            "--patience", "99", "--pairwise-weight", "0.15", "--pair-scope", "same_site",
            "--device", "cuda", "--models", "raw_sequence", "--output-dir",
            str((args.private_output / f"seed_{seed}").resolve())]
        context["commands"].append(command)
        (args.private_output / "run_context.json").write_text(json.dumps(context, indent=2) + "\n")
        print(json.dumps({"exec": command}), flush=True)
        subprocess.run(command, check=True)
    sites, summary, predictions = aggregate(args.private_output)
    sites.to_csv(args.private_output / "loso_metrics.csv", index=False)
    summary.to_csv(args.private_output / "summary.csv", index=False)
    predictions.to_parquet(args.private_output / "ensemble_predictions.parquet", index=False)
    comparison = compare_targets(sites, summary)
    (args.private_output / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    print(json.dumps({"comparison": comparison}, indent=2), flush=True)
    # 1e-12 covers floating point arithmetic only, not a changed concordant pair.
    if any(not np.isfinite(row["delta"]) or abs(row["delta"]) > 1e-12 for row in comparison):
        print("STOP: reproduction differs from target; no public results exported.", flush=True)
        return 2
    args.results_dir.mkdir(parents=True, exist_ok=True)
    sites.to_csv(args.results_dir / "loso_metrics.csv", index=False)
    summary.to_csv(args.results_dir / "summary.csv", index=False)
    print("PASS: fresh three-seed LOSO matches every primary AC-AUROC target.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
