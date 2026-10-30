"""Fresh parent-gated DEDUP_FUSION runner and official aggregate scorer."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_model import compute_auprc, compute_auroc, compute_auroc_age
from fusion.route import FusionStop, fuse_final_scores, validate_parent_identity

SITES = ("I0002", "I0006", "S0001")
EXPECTED_CORE = {
    "I0002": (.6935312831389183, .7331460674157304, .3855376014994263),
    "I0006": (.6745325329469399, .7428874826629681, .3076191308949592),
    "S0001": (.6693029123018702, .7464956663156518, .20366315612725175),
}
EXPECTED_CAISR = {
    "I0002": (.7343584305408272, .7532771535580525, .4907637732055276),
    "I0006": (.6571448285379149, .7118368585298197, .19319281080550868),
    "S0001": (.6116437229199803, .6762946531494762, .12901730597888802),
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--core-root", type=Path, required=True)
    parser.add_argument("--caisr-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    return parser


def _rank_mean(columns: np.ndarray) -> np.ndarray:
    values = np.asarray(columns, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 15 or not np.isfinite(values).all():
        raise FusionStop("CAISR parent requires finite N x 15 seed scores")
    ranks = np.empty_like(values)
    for column in range(values.shape[1]):
        _, inverse, counts = np.unique(values[:, column], return_inverse=True, return_counts=True)
        less = np.concatenate(([0], np.cumsum(counts[:-1])))
        ranks[:, column] = (less[inverse] + 0.5 * counts[inverse]) / values.shape[0]
    return ranks.mean(axis=1, dtype=np.float64)


def _score(labels: np.ndarray, ages: np.ndarray, scores: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(labels, dtype=np.int8); a = np.asarray(ages, dtype=np.float64); s = np.asarray(scores, dtype=np.float64)
    positives = np.flatnonzero(y == 1); negatives = np.flatnonzero(y == 0)
    pairs = sum(abs(float(a[i]) - float(a[j])) <= 2 for i in positives for j in negatives)
    return {"n_records": int(y.size), "n_positive": int(y.sum()), "ac_pair_count": int(pairs), "ac_auroc": float(compute_auroc_age(y, s, a, gap=2)), "auroc": float(compute_auroc(y, s)), "auprc": float(compute_auprc(y, s))}


def _assert_expected(site: str, metrics: dict[str, float | int], expected: dict[str, tuple[float, float, float]]) -> None:
    actual = (float(metrics["ac_auroc"]), float(metrics["auroc"]), float(metrics["auprc"]))
    target = expected[site]
    if any(abs(a - b) > 1e-12 for a, b in zip(actual, target)):
        raise FusionStop(f"parent metric drift for {site}: actual={actual} expected={target}")


def main() -> None:
    args = _parser().parse_args()
    manifest = pd.read_csv(args.manifest, dtype={"record_id": str, "site": str})
    required = {"record_id", "site", "label", "age"}
    if set(manifest.columns) < required or len(manifest) != 6600:
        raise FusionStop("manifest identity schema/count mismatch")
    if manifest.record_id.nunique() != 6600 or int(manifest.label.sum()) != 498:
        raise FusionStop("manifest identity/label uniqueness mismatch")
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for site in SITES:
        expected_indices = np.flatnonzero(manifest.site.to_numpy(str) == site).astype(np.int64)
        with np.load(args.core_root / "final" / f"holdout_{site}" / "payload.npz", allow_pickle=False) as archive:
            core_indices = np.asarray(archive["test_global_indices"], dtype=np.int64)
            core_scores = np.asarray(archive["CORE_NO_CAISR20_FINAL"], dtype=np.float64)
        validate_parent_identity(expected_indices, core_indices, site)
        core_metrics = _score(manifest.loc[core_indices, "label"], manifest.loc[core_indices, "age"], core_scores)
        _assert_expected(site, core_metrics, EXPECTED_CORE)
        seed_columns = []
        caisr_indices = None
        for seed in range(42, 57):
            unit = args.caisr_root / "predictions" / f"seed{seed}" / f"holdout_{site}"
            metadata = json.loads((unit / "metadata.json").read_text(encoding="utf-8"))
            if metadata.get("route_id") != "CAISR20_CORE_PROTOCOL" or metadata.get("holdout_site") != site:
                raise FusionStop(f"CAISR parent route identity mismatch: {unit}")
            with np.load(unit / "payload.npz", allow_pickle=False) as archive:
                indices = np.asarray(archive["test_global_indices"], dtype=np.int64)
                scores = np.asarray(archive["scores"], dtype=np.float64)
            validate_parent_identity(expected_indices, indices, site)
            if caisr_indices is None: caisr_indices = indices
            elif not np.array_equal(caisr_indices, indices): raise FusionStop(f"CAISR seed order mismatch: {site}")
            seed_columns.append(scores)
        caisr_scores = _rank_mean(np.column_stack(seed_columns))
        validate_parent_identity(core_indices, caisr_indices, site)
        caisr_metrics = _score(manifest.loc[caisr_indices, "label"], manifest.loc[caisr_indices, "age"], caisr_scores)
        _assert_expected(site, caisr_metrics, EXPECTED_CAISR)
        fused = fuse_final_scores(core_scores, caisr_scores)
        metrics = _score(manifest.loc[core_indices, "label"], manifest.loc[core_indices, "age"], fused)
        rows.append({"route": "DEDUP_FUSION", "held_out_site": site, **metrics})
        np.savez_compressed(args.output_root / f"holdout_{site}.npz", test_global_indices=core_indices, DEDUP_FUSION_FINAL=fused)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "loso_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    ac = [float(row["ac_auroc"]) for row in rows]
    summary = {"route": "DEDUP_FUSION", "small_site_ac_auroc": (ac[0] + ac[1]) / 2, "macro_ac_auroc": sum(ac) / 3, "macro_auroc": sum(float(row["auroc"]) for row in rows) / 3, "macro_auprc": sum(float(row["auprc"]) for row in rows) / 3}
    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary)); writer.writeheader(); writer.writerow(summary)
    print(summary)

if __name__ == "__main__":
    main()
