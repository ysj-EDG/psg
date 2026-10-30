"""Rank-mean ensemble and official aggregate scoring."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .route import mean_tie_percentile_ranks


def rank_mean(scores: np.ndarray) -> np.ndarray:
    return mean_tie_percentile_ranks(scores)[1]


def score_final_vectors(labels, ages, scores, official):
    labels = np.asarray(labels, dtype=np.int8)
    ages = np.asarray(ages, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    if not (labels.shape == ages.shape == scores.shape):
        raise ValueError("labels, ages, and scores must be aligned")
    pairs = int(sum(np.count_nonzero(np.abs(ages[labels == 0] - age) <= 2) for age in ages[labels == 1]))
    return {
        "n_records": int(len(labels)),
        "n_positive": int(np.sum(labels == 1)),
        "ac_pair_count": pairs,
        "ac_auroc": float(official.compute_auroc_age(labels, scores, ages, gap=2)),
        "auroc": float(official.compute_auroc(labels, scores)),
        "auprc": float(official.compute_auprc(labels, scores)),
    }


def write_aggregate_results(rows, summary, output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "loso_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("route", "held_out_site", "n_records", "n_positive", "ac_pair_count", "ac_auroc", "auroc", "auprc"))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("route", "small_site_ac_auroc", "macro_ac_auroc", "macro_auroc", "macro_auprc"))
        writer.writeheader()
        writer.writerow(summary)
