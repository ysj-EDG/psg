"""Score a frozen CORE_NO_CAISR20 final prediction bundle."""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np

from siren.ensemble import score_final_vectors, write_aggregate_results


def _load_official(path: Path):
    spec = importlib.util.spec_from_file_location("physionet_official", Path(path))
    if spec is None or spec.loader is None:
        raise ValueError(f"unable to load official scorer: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--final-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True, help="private bundle containing labels and ages")
    parser.add_argument("--official-scoring", type=Path, default=Path("evaluate_model.py"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args(argv)
    official = _load_official(args.official_scoring)
    rows = []
    for site in ("I0002", "I0006", "S0001"):
        with np.load(args.final_root / f"holdout_{site}" / "payload.npz", allow_pickle=False) as final:
            scores = np.asarray(final["CORE_NO_CAISR20_FINAL"], dtype=np.float64)
        with np.load(args.reference_root / f"{site}_inductive" / "result.npz", allow_pickle=False) as reference:
            labels = np.asarray(reference["labels"], dtype=np.int8)
            ages = np.asarray(reference["ages"], dtype=np.float64)
        if labels.shape != scores.shape or ages.shape != scores.shape:
            raise ValueError(f"reference/final shape mismatch for {site}")
        scored = score_final_vectors(labels, ages, scores, official)
        rows.append({"route": "CORE_NO_CAISR20", "held_out_site": site, **scored})
    macro = {key: float(np.mean([row[key] for row in rows])) for key in ("ac_auroc", "auroc", "auprc")}
    small = float(np.mean([row["ac_auroc"] for row in rows[:2]]))
    write_aggregate_results(rows, {"route": "CORE_NO_CAISR20", "small_site_ac_auroc": small,
                                   "macro_ac_auroc": macro["ac_auroc"], "macro_auroc": macro["auroc"],
                                   "macro_auprc": macro["auprc"]}, args.output_dir)


if __name__ == "__main__":
    main()
