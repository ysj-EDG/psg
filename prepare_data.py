#!/usr/bin/env python3
"""Build the historical M2 prepared route from raw PhysioNet EDF files.

raw EDF -> inventory/derivation plan -> annotation alignment ->
historical M2 canonical HDF5 -> training_manifest.parquet.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pyedflib

from preprocessing_primitives import CANONICAL_BY_MODALITY


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--channel-table", type=Path, default=Path(__file__).with_name("channel_table.csv"))
    parser.add_argument("--record-metadata", type=Path, help="CSV or Parquet with patient_id, SiteID, Age, Sex, Race, channel_config and split columns")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _is_psg(path: Path) -> bool:
    stem = path.stem.lower()
    return path.suffix.lower() == ".edf" and not (stem.endswith("_caisr_annotations") or stem.endswith("_expert_annotations"))


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def inventory(data_root: Path, output_root: Path, channel_table: Path) -> None:
    from raw_channel_mapping import build_source_specs

    audit = output_root / "audit"
    manifests = output_root / "manifests"
    audit.mkdir(parents=True, exist_ok=True)
    manifests.mkdir(parents=True, exist_ok=True)
    records, channels, plans = [], [], []
    for edf_path in sorted(path for path in data_root.rglob("*.edf") if _is_psg(path)):
        record_id, site = edf_path.stem, edf_path.parent.name
        with pyedflib.EdfReader(str(edf_path)) as reader:
            duration = float(reader.file_duration)
            labels = [str(label) for label in reader.getSignalLabels()]
            headers = reader.getSignalHeaders()
        relative = _relative(edf_path, data_root)
        records.append({"record_id": record_id, "site": site, "duration_sec": duration, "relative_path": relative})
        for index, (label, header) in enumerate(zip(labels, headers)):
            channels.append({"record_id": record_id, "channel_index": index, "channel_label": label, "sampling_rate_hz": float(header["sample_frequency"])})
        specs = build_source_specs(labels, channel_table)
        for modality, targets in CANONICAL_BY_MODALITY.items():
            for target in targets:
                spec = specs.get(target)
                plans.append({"record_id": record_id, "target_group": target, "available_after_standardization": spec is not None, "derived_from_components": bool(spec and len(spec.indices) > 1), "method": "unavailable" if spec is None else spec.method, "reference_system": "" if spec is None else spec.reference, "source_labels": "" if spec is None else " | ".join(spec.labels)})
    if not records:
        raise RuntimeError(f"No PSG EDF files found below {data_root}")
    record_frame = pd.DataFrame(records).sort_values(["site", "record_id"])
    pd.DataFrame(channels).to_csv(audit / "psg_channel_headers.csv", index=False)
    record_frame.to_csv(audit / "psg_record_headers.csv", index=False)
    pd.DataFrame(plans).to_csv(audit / "canonical_derivation_plan_v3_by_record.csv", index=False)
    by_stem = {path.stem: _relative(path, data_root) for path in data_root.rglob("*.edf")}
    annotation_rows = []
    for row in records:
        stem = row["record_id"]
        annotation_rows.append({"record_id": stem, "has_caisr": f"{stem}_caisr_annotations" in by_stem, "has_human": f"{stem}_expert_annotations" in by_stem, "caisr_path": by_stem.get(f"{stem}_caisr_annotations", ""), "human_path": by_stem.get(f"{stem}_expert_annotations", "")})
    pd.DataFrame(annotation_rows).to_csv(manifests / "record_manifest.csv", index=False)
    print(f"Inventoried {len(record_frame)} PSG EDF files")


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    args.output_root.mkdir(parents=True, exist_ok=True)
    inventory(args.data_root, args.output_root, args.channel_table)
    environment = os.environ.copy()
    environment.update({"PSG_PREPARED_ROOT": str(args.output_root), "PSG_RAW_DATA_ROOT": str(args.data_root), "PSG_PREPARED_AUDIT_ROOT": str(args.output_root / "audit"), "PSG_PREPARED_MANIFEST_ROOT": str(args.output_root / "manifests"), "PSG_CANONICAL_CACHE_ROOT": str(args.output_root / "canonical" / "records")})
    alignment = Path(__file__).with_name("annotation_alignment_source.py")
    subprocess.run([sys.executable, str(alignment)], check=True, env=environment)
    builder = Path(__file__).with_name("preprocessing_primitives.py")
    command = [sys.executable, str(builder), "--workers", str(args.workers)]
    if args.max_records is not None:
        command += ["--max-records", str(args.max_records)]
    if args.overwrite:
        command.append("--overwrite")
    subprocess.run(command, check=True, env=environment)
    source = args.output_root / "manifests" / "full_v1_record_manifest.parquet"
    if not source.is_file():
        raise RuntimeError(f"Historical builder did not create {source}")
    frame = pd.read_parquet(source)
    if args.record_metadata is not None:
        metadata = pd.read_parquet(args.record_metadata) if args.record_metadata.suffix.lower() == ".parquet" else pd.read_csv(args.record_metadata)
        if "record_id" not in metadata.columns:
            raise RuntimeError("--record-metadata must contain record_id")
        frame = frame.drop(columns=[column for column in metadata.columns if column != "record_id" and column in frame.columns])
        frame = frame.merge(metadata, on="record_id", how="left", validate="one_to_one")
    frame["canonical_path"] = frame["cache_path"].map(lambda value: str(Path(value)))
    output = args.output_root / "training_manifest.parquet"
    frame.to_parquet(output, index=False)
    print(f"Prepared manifest: {output}")


if __name__ == "__main__":
    main()
