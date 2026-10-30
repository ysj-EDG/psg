"""Historical signal reader extracted unchanged from the recovered M2 execution layer."""
from __future__ import annotations
import h5py
import numpy as np
import torch
from domain_robust_encoder_model import MODALITY_CHANNELS

MODALITIES = tuple(MODALITY_CHANNELS)
SAMPLING_RATES = {"eeg": 128, "eog": 128, "ecg": 128, "resp": 32, "spo2": 1, "emg": 128}
EPOCHS_PER_STORAGE_CHUNK = 10


def epoch_channel_valid(handle: h5py.File, modality: str, n_epochs: int) -> np.ndarray:
    channels = MODALITY_CHANNELS[modality]
    present = np.asarray(
        handle.get(f"quality/channel_present/{modality}", np.ones(channels, dtype=bool)),
        dtype=bool,
    )
    hard = np.asarray(handle[f"quality/channel_hard_valid_5s/{modality}"], dtype=bool)
    expected = n_epochs * 6
    if hard.shape != (expected, channels):
        raise RuntimeError(f"Invalid {modality} quality shape {hard.shape}")
    return present[None, :] & (hard.reshape(n_epochs, 6, channels).mean(axis=1) >= 4.0 / 6.0)


def decode_strings(values: np.ndarray) -> list[str]:
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def read_modality_windows(
    handle: h5py.File, modality: str, indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    rate = SAMPLING_RATES[modality]
    samples_per_epoch = rate * 30
    channels = MODALITY_CHANNELS[modality]
    indices = np.asarray(indices, dtype=np.int64)
    output = np.zeros((len(indices), channels, samples_per_epoch), dtype=np.float32)
    dataset = handle[f"signals/{modality}"]

    # Official-server preprocessing stores normalized, legacy-compatible int16
    # 30-second epochs. Earlier inference caches used float16, while the original
    # research archive stored the same quantized signal continuously. Support all
    # layouts without changing the training objective or model.
    if dataset.ndim == 3:
        if dataset.shape[1:] != (channels, samples_per_epoch):
            raise RuntimeError(
                f"Invalid online {modality} signal shape {dataset.shape}"
            )
        unique, inverse = np.unique(indices, return_inverse=True)
        output = np.asarray(dataset[unique], dtype=np.float32)[inverse]
        if np.issubdtype(dataset.dtype, np.integer):
            output /= 256.0
        n_epochs = int(handle.attrs["complete_epoch_count"])
        valid = epoch_channel_valid(handle, modality, n_epochs)[indices].astype(np.float32)
        output = np.nan_to_num(output, nan=0.0, posinf=12.0, neginf=-12.0)
        output = np.clip(output, -12.0, 12.0) * valid[:, :, None]
        return output, valid

    if dataset.ndim != 2:
        raise RuntimeError(f"Invalid legacy {modality} signal shape {dataset.shape}")
    storage_modes = decode_strings(np.asarray(handle[f"quantization/{modality}/storage_mode"]))
    channel_names = decode_strings(np.asarray(handle[f"metadata/{modality}/channel_names"]))
    centers = np.asarray(handle[f"quantization/{modality}/center"], dtype=np.float64)
    scales = np.asarray(handle[f"quantization/{modality}/robust_scale"], dtype=np.float64)
    for chunk_id in np.unique(indices // EPOCHS_PER_STORAGE_CHUNK):
        positions = np.flatnonzero(indices // EPOCHS_PER_STORAGE_CHUNK == chunk_id)
        sample_start = int(chunk_id) * EPOCHS_PER_STORAGE_CHUNK * samples_per_epoch
        sample_end = min(
            sample_start + EPOCHS_PER_STORAGE_CHUNK * samples_per_epoch,
            dataset.shape[1],
        )
        chunk = np.asarray(dataset[:, sample_start:sample_end], dtype=np.float32) / 256.0
        for channel_index, mode in enumerate(storage_modes):
            if mode != "float64_fallback":
                continue
            full_path = f"overflow/{modality}/{channel_names[channel_index]}/full_values"
            full = np.asarray(handle[full_path][sample_start:sample_end], dtype=np.float32)
            chunk[channel_index] = (
                full - centers[channel_index]
            ) / max(scales[channel_index], 1e-12)
        for position in positions:
            local_epoch = int(indices[position] % EPOCHS_PER_STORAGE_CHUNK)
            start = local_epoch * samples_per_epoch
            output[position] = chunk[:, start : start + samples_per_epoch]
    n_epochs = int(handle.attrs["complete_epoch_count"])
    valid = epoch_channel_valid(handle, modality, n_epochs)[indices].astype(np.float32)
    output = np.nan_to_num(output, nan=0.0, posinf=12.0, neginf=-12.0)
    output = np.clip(output, -12.0, 12.0) * valid[:, :, None]
    return output, valid


def load_signal_batch(path: str, indices: np.ndarray) -> dict[str, torch.Tensor]:
    output: dict[str, torch.Tensor] = {}
    with h5py.File(path, "r") as handle:
        for modality in MODALITIES:
            signal, mask = read_modality_windows(handle, modality, indices)
            output[modality] = torch.from_numpy(signal)
            output[f"{modality}_mask"] = torch.from_numpy(mask)
    return output
