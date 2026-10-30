#!/usr/bin/env python3
"""Export historical E1 frozen 192-D embeddings from the prepared cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch


def torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_runtime(checkpoint_path: Path, device: torch.device):
    from domain_robust_encoder_model import DomainRobustPsgEncoder, MODALITY_CHANNELS
    from historical_signal_io import load_signal_batch

    expected = "e9f135c973c0540970f34acbde89096c028eb3a95a0a601d85f21e8f83d36ac0"
    checkpoint_hash = file_sha256(checkpoint_path)
    if checkpoint_hash != expected:
        raise ValueError("Expected the exact historical frozen E1 checkpoint")
    checkpoint = torch_load(checkpoint_path)
    state = checkpoint.get("model_state", checkpoint.get("teacher_state", checkpoint))
    encoder = DomainRobustPsgEncoder().to(device).eval()
    encoder.load_state_dict(state, strict=True)
    return MODALITY_CHANNELS, encoder, load_signal_batch, checkpoint_hash


def eligible_windows(handle: h5py.File, modalities, epoch_count: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    channel_valid = {}
    for modality, channels in modalities.items():
        present = np.asarray(handle.get(f"quality/channel_present/{modality}", np.ones(channels, dtype=bool)), dtype=bool)
        hard = np.asarray(handle[f"quality/channel_hard_valid_5s/{modality}"], dtype=bool)
        channel_valid[modality] = present[None, :] & (hard.reshape(epoch_count, 6, channels).mean(axis=1) >= 4.0 / 6.0)
    modality_valid = np.column_stack([np.any(channel_valid[modality], axis=1) for modality in modalities])
    neural_valid = modality_valid[:, 0] | modality_valid[:, 1]
    eligible = np.flatnonzero(neural_valid & (modality_valid.sum(axis=1) >= 2))
    return eligible.astype(np.int64), channel_valid


def encode_record(cache_path: Path, encoder, modalities, load_signal_batch, device: torch.device, batch_size: int):
    with h5py.File(cache_path, "r") as handle:
        epoch_count = int(handle.attrs["complete_epoch_count"])
        indices, channel_valid = eligible_windows(handle, modalities, epoch_count)
        encoded = []
        with torch.inference_mode():
            for start in range(0, len(indices), batch_size):
                selected = indices[start : start + batch_size]
                batch = load_signal_batch(str(cache_path), selected)
                batch = {name: value.to(device, non_blocking=True) for name, value in batch.items()}
                output = encoder(batch)["embedding"]
                if not torch.isfinite(output).all():
                    raise FloatingPointError(f"Non-finite embedding in {cache_path}")
                encoded.append(output.cpu().numpy().astype(np.float16))
    embedding = np.concatenate(encoded) if encoded else np.empty((0, 192), dtype=np.float16)
    return indices.astype(np.int32), embedding, epoch_count


def write_dataset(group: h5py.Group, name: str, values: np.ndarray) -> None:
    group.create_dataset(name, data=values, compression="lzf", shuffle=True)


def export_shard(shard_index: int, rows: pd.DataFrame, prepared: Path, output: Path, encoder, modalities, load_signal_batch, device, batch_size, checkpoint_hash) -> None:
    shard_path = output / "shards" / f"shard_{shard_index:04d}.h5"
    sidecar = output / "manifests" / f"shard_{shard_index:04d}.parquet"
    if shard_path.is_file() and sidecar.is_file():
        return
    temporary = shard_path.with_suffix(".h5.tmp")
    if temporary.exists():
        temporary.unlink()
    metadata = []
    with h5py.File(temporary, "w") as handle:
        handle.attrs["version"] = "historical_e1_embeddings_v1"
        handle.attrs["checkpoint_sha256"] = checkpoint_hash
        handle.attrs["shard_index"] = shard_index
        for position, row in enumerate(rows.to_dict(orient="records")):
            record_id = str(row["record_id"])
            indices, embedding, epochs = encode_record(prepared / "canonical" / "records" / f"{record_id}.h5", encoder, modalities, load_signal_batch, device, batch_size)
            group_key = f"record_{position:04d}"
            group = handle.create_group(group_key)
            group.attrs["record_id"] = record_id
            group.attrs["patient_id"] = str(row["patient_id"])
            group.attrs["site"] = str(row["SiteID"])
            group.attrs["complete_epoch_count"] = epochs
            write_dataset(group, "window_index", indices)
            write_dataset(group, "embedding", embedding)
            metadata.append({"record_id": record_id, "patient_id": str(row["patient_id"]), "SiteID": str(row["SiteID"]), "shard_index": shard_index, "shard_path": str(shard_path), "group_key": group_key, "complete_epoch_count": epochs, "embedding_count": int(len(indices))})
        handle.flush()
    temporary.replace(shard_path)
    side_tmp = sidecar.with_suffix(".parquet.tmp")
    pd.DataFrame(metadata).to_parquet(side_tmp, index=False)
    side_tmp.replace(sidecar)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--records-per-shard", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError("Use a fresh embedding output directory")
    manifest = pd.read_parquet(args.prepared_root / "training_manifest.parquet").copy()
    manifest["record_id"] = manifest["record_id"].astype(str)
    manifest = manifest.sort_values(["SiteID", "record_id"]).reset_index(drop=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    modalities, encoder, load_signal_batch, checkpoint_hash = load_runtime(args.checkpoint, device)
    output = args.output_dir
    (output / "shards").mkdir(parents=True, exist_ok=True)
    (output / "manifests").mkdir(parents=True, exist_ok=True)
    shard_count = math.ceil(len(manifest) / args.records_per_shard)
    started = time.time()
    for shard_index in range(shard_count):
        low = shard_index * args.records_per_shard
        high = min(low + args.records_per_shard, len(manifest))
        export_shard(shard_index, manifest.iloc[low:high], args.prepared_root, output, encoder, modalities, load_signal_batch, device, args.batch_size, checkpoint_hash)
        print(json.dumps({"shard": shard_index + 1, "shards": shard_count, "records": high, "elapsed_sec": time.time() - started}), flush=True)
    sidecars = sorted((output / "manifests").glob("shard_*.parquet"))
    exported = pd.concat([pd.read_parquet(path) for path in sidecars], ignore_index=True)
    if len(exported) != len(manifest) or exported["record_id"].duplicated().any():
        raise RuntimeError(f"incomplete embedding export: {len(exported)} / {len(manifest)}")
    exported = exported.sort_values(["SiteID", "record_id"]).reset_index(drop=True)
    exported.to_parquet(output / "embedding_manifest.parquet", index=False)
    summary = {"status": "complete", "checkpoint_sha256": checkpoint_hash, "records": len(exported), "embedding_windows": int(exported["embedding_count"].sum()), "shards": shard_count}
    (output / "export_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "_SUCCESS").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
