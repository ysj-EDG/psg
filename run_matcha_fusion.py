"""Fresh parent-gated AB_EQUAL_50_50 runner and official aggregate scorer."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_model import compute_auprc, compute_auroc, compute_auroc_age
from matcha_fusion.route import FusionStop, fuse_ranked_scores, validate_parent_identity

SITES = ("I0002", "I0006", "S0001")
EXPECTED_DEDUP = {
    "I0002": (.7423117709437964, .7713195044655718, .4940394848794116),
    "I0006": (.7045470227006141, .7668949375866851, .2934051494793416),
    "S0001": (.6618493939560138, .738258862088518, .20486333514451122),
}
EXPECTED_MATCHA = {
    "I0002": (.7481442205726405, .7851483722270238, .5209606504394242),
    "I0006": (.7393224315186642, .7901005547850207, .2945779196802375),
    "S0001": (.6680414632808643, .7439418769121485, .1867674115690459),
}
EXPECTED_FUSED = {
    "I0002": (.7582184517497349, .794583693460098, .5166653860047571),
    "I0006": (.7457738218450286, .8002427184466019, .33623939962453203),
    "S0001": (.684056381286678, .7660143812271399, .21436182065328968),
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dedup-root", type=Path, required=True)
    parser.add_argument("--matcha-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    return parser


def _score(labels: np.ndarray, ages: np.ndarray, scores: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(labels, dtype=np.int8)
    a = np.asarray(ages, dtype=np.float64)
    s = np.asarray(scores, dtype=np.float64)
    positives = np.flatnonzero(y == 1); negatives = np.flatnonzero(y == 0)
    pairs = sum(abs(float(a[i]) - float(a[j])) <= 2 for i in positives for j in negatives)
    return {"n_records": int(y.size), "n_positive": int(y.sum()), "ac_pair_count": int(pairs), "ac_auroc": float(compute_auroc_age(y, s, a, gap=2)), "auroc": float(compute_auroc(y, s)), "auprc": float(compute_auprc(y, s))}


def _assert_expected(site: str, metrics: dict[str, float | int], expected: dict[str, tuple[float, float, float]]) -> None:
    actual = (float(metrics["ac_auroc"]), float(metrics["auroc"]), float(metrics["auprc"]))
    target = expected[site]
    if any(abs(a - b) > 1e-12 for a, b in zip(actual, target)):
        raise FusionStop(f"metric drift for {site}: actual={actual} expected={target}")


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
        with np.load(args.dedup_root / f"holdout_{site}.npz", allow_pickle=False) as archive:
            dedup_indices = np.asarray(archive["test_global_indices"], dtype=np.int64)
            dedup_scores = np.asarray(archive["DEDUP_FUSION_FINAL"], dtype=np.float64)
        validate_parent_identity(expected_indices, dedup_indices, site)
        dedup_metrics = _score(manifest.loc[dedup_indices, "label"], manifest.loc[dedup_indices, "age"], dedup_scores)
        _assert_expected(site, dedup_metrics, EXPECTED_DEDUP)

        matcha_path = args.matcha_root / "fused" / f"holdout_{site}" / "payload.npz"
        with np.load(matcha_path, allow_pickle=False) as archive:
            matcha_indices = np.asarray(archive["global_index"], dtype=np.int64)
            seed_scores = np.column_stack([archive[f"B_seed_{seed}_logit"] for seed in ("20260806", "20260807", "20260808")]).astype(np.float64)
            matcha_scores = seed_scores.mean(axis=1, dtype=np.float64)
            stored_final = np.asarray(archive["B_final"], dtype=np.float64)
        validate_parent_identity(expected_indices, matcha_indices, site)
        if not np.array_equal(matcha_scores, stored_final):
            raise FusionStop(f"Matcha raw-logit mean drift: {site}")
        matcha_metrics = _score(manifest.loc[matcha_indices, "label"], manifest.loc[matcha_indices, "age"], matcha_scores)
        _assert_expected(site, matcha_metrics, EXPECTED_MATCHA)
        validate_parent_identity(dedup_indices, matcha_indices, site)

        fused = fuse_ranked_scores(dedup_scores, matcha_scores)
        metrics = _score(manifest.loc[dedup_indices, "label"], manifest.loc[dedup_indices, "age"], fused)
        _assert_expected(site, metrics, EXPECTED_FUSED)
        rows.append({"route": "AB_EQUAL_50_50", "held_out_site": site, **metrics})
        np.savez_compressed(args.output_root / f"holdout_{site}.npz", test_global_indices=dedup_indices, DEDUP_FUSION_FINAL=dedup_scores, MATCHA_M2_FINAL=matcha_scores, AB_EQUAL_50_50_FINAL=fused)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "loso_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    ac = [float(row["ac_auroc"]) for row in rows]
    summary = {"route": "AB_EQUAL_50_50", "small_site_ac_auroc": (ac[0] + ac[1]) / 2, "macro_ac_auroc": sum(ac) / 3, "macro_auroc": sum(float(row["auroc"]) for row in rows) / 3, "macro_auprc": sum(float(row["auprc"]) for row in rows) / 3}
    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary)); writer.writeheader(); writer.writerow(summary)
    print(summary)


if __name__ == "__main__":
    main()
