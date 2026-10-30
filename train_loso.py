"""Train/infer the CORE_NO_CAISR20 V5 branch on frozen Rank256 units."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from per_epoch_features.extensions.core_no_caisr20_runner import run_v5_no_caisr20
from reproduction.siren_a.pipeline.label_join import load_metadata_index


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank-root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--g2-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--caisr-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--old-root", type=Path, required=True)
    args = parser.parse_args(argv)
    metadata = load_metadata_index(args.metadata, args.g2_root)
    summary = run_v5_no_caisr20(
        rank_root=args.rank_root,
        metadata=metadata,
        manifest_path=args.manifest,
        caisr_manifest_path=args.caisr_manifest,
        source_fingerprints={},
        output_root=args.output_root,
        old_root=args.old_root,
        run_identity={
            "schema": "siren_fiveview_train_v1",
            "route_id": "CORE_NO_CAISR20",
            "feature_view": "V5",
            "feature_count": 265,
            "rank256_rebuild_count": 0,
            "v1_v4_refit_count": 0,
            "v1_v4_prediction_recompute_count": 0,
        },
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
