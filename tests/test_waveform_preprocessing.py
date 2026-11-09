import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

import waveform_preprocessing
from channel_sources import build_source_plans
from waveform_cache import EdfHeader, write_schema_cache, write_waveform_signals
from waveform_preprocessing import preprocess_selected_sources
from waveform_primitives import bandpass_filter, resample_finite_signal


LABELS = (
    "F3-M2", "F4-M1", "C3-M2", "C4-M1", "O1-M2", "O2-M1",
    "E1", "E2", "EKG", "CHIN", "LAT", "RAT",
    "NPT", "THERM", "C-FLOW", "C PRESS", "CHEST", "ABDOMINAL", "SaO2",
)


def _header() -> EdfHeader:
    return EdfHeader(
        start_datetime="2020-01-01 00:00:00",
        duration_sec=30.0,
        channel_names=LABELS,
        native_sampling_rates_hz=tuple(500.0 for _ in LABELS),
        native_sample_counts=tuple(15000 for _ in LABELS),
        physical_dimensions=tuple("uV" for _ in LABELS),
    )


class WaveformPreprocessingTests(unittest.TestCase):
    def test_bandpass_filter_rejects_nonfinite_input(self):
        with self.assertRaises(ValueError):
            bandpass_filter(np.array([0.0, np.nan, 1.0]), 500.0, 0.3, 35.0)

    def test_resample_finite_signal_rejects_nonfinite_input(self):
        with self.assertRaises(ValueError):
            resample_finite_signal(np.array([0.0, np.inf, 1.0]), 500.0, 128, 10)

    def test_finite_selected_sources_make_fixed_epoch_shapes_and_missing_nan_slot(self):
        header = _header()
        plans = build_source_plans(
            header.channel_names,
            header.native_sampling_rates_hz,
            header.native_sample_counts,
            header.physical_dimensions,
        )
        calls = []

        def read_signal(index):
            calls.append(index)
            x = np.linspace(0.0, 30.0, 15000, endpoint=False)
            return np.sin(2.0 * np.pi * (1.0 + index / 100.0) * x)

        result = preprocess_selected_sources(
            plans, header, read_signal=read_signal, n_epochs=1
        )
        self.assertEqual(len(calls), 19)
        self.assertEqual(set(calls), set(range(19)))
        self.assertEqual(result.signals["eeg"].shape, (1, 6, 3840))
        self.assertEqual(result.signals["eog"].shape, (1, 2, 3840))
        self.assertEqual(result.signals["ecg"].shape, (1, 1, 6000))
        self.assertEqual(result.signals["emg"].shape, (1, 3, 6000))
        self.assertEqual(result.signals["resp"].shape, (1, 7, 750))
        self.assertEqual(result.signals["spo2"].shape, (1, 1, 30))
        self.assertTrue(np.isnan(result.signals["resp"][:, 2, :]).all())
        self.assertTrue(np.isfinite(result.signals["eeg"][:, 0, :]).all())
        self.assertTrue(np.isfinite(result.signals["spo2"][:, 0, :]).all())

    def test_paired_source_reads_both_selected_components_and_subtracts(self):
        labels = ("F3", "A2")
        header = EdfHeader(
            start_datetime="2020-01-01 00:00:00",
            duration_sec=30.0,
            channel_names=labels,
            native_sampling_rates_hz=(500.0, 500.0),
            native_sample_counts=(15000, 15000),
            physical_dimensions=("uV", "uV"),
        )
        plans = build_source_plans(labels, (500.0, 500.0), (15000, 15000), ("uV", "uV"))
        calls = []

        def read_signal(index):
            calls.append(index)
            return np.full(15000, 3.0 if index == 0 else 1.0)

        result = preprocess_selected_sources(
            plans, header, read_signal=read_signal, n_epochs=1
        )
        self.assertEqual(calls, [0, 1])
        self.assertEqual(result.slot_results["EEG_F3"].source_indices, (0, 1))
        self.assertTrue(result.slot_results["EEG_F3"].processed)
        self.assertTrue(np.isfinite(result.signals["eeg"][:, 0, :]).all())

    def test_nonfinite_source_stops_that_slot_before_filter(self):
        labels = ("F3-M2",)
        header = EdfHeader(
            start_datetime="2020-01-01 00:00:00",
            duration_sec=30.0,
            channel_names=labels,
            native_sampling_rates_hz=(500.0,),
            native_sample_counts=(15000,),
            physical_dimensions=("uV",),
        )
        plans = build_source_plans(labels, (500.0,), (15000,), ("uV",))
        calls = []

        def read_signal(index):
            calls.append(index)
            values = np.ones(15000, dtype=float)
            values[10] = np.nan
            return values

        original = waveform_preprocessing.bandpass_filter
        waveform_preprocessing.bandpass_filter = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("filter must not run"))
        try:
            result = preprocess_selected_sources(
                plans, header, read_signal=read_signal, n_epochs=1
            )
        finally:
            waveform_preprocessing.bandpass_filter = original
        self.assertEqual(calls, [0])
        self.assertTrue(np.isnan(result.signals["eeg"]).all())
        stats = [item for item in result.source_inspections if item.source_index == 0]
        self.assertEqual(len(stats), 1)
        self.assertEqual(stats[0].nan_count, 1)
        self.assertEqual(result.slot_results["EEG_F3"].reason, "raw_nonfinite")

    def test_signal_writer_uses_float32_nan_chunked_compressed_datasets(self):
        psg = _header()
        caisr = EdfHeader(
            start_datetime="1985-01-01 00:00:00",
            duration_sec=30.0,
            channel_names=("stage_caisr",),
            native_sampling_rates_hz=(1 / 30,),
            native_sample_counts=(1,),
            physical_dimensions=("stage",),
        )
        arrays = {
            "eeg": np.full((1, 6, 3840), np.nan, dtype=np.float32),
            "eog": np.full((1, 2, 3840), np.nan, dtype=np.float32),
            "ecg": np.full((1, 1, 6000), np.nan, dtype=np.float32),
            "emg": np.full((1, 3, 6000), np.nan, dtype=np.float32),
            "resp": np.full((1, 7, 750), np.nan, dtype=np.float32),
            "spo2": np.full((1, 1, 30), np.nan, dtype=np.float32),
        }
        with tempfile.TemporaryDirectory() as directory:
            output = write_schema_cache(
                Path(directory) / "waveform.h5",
                record_id="sub-I0002150026011_ses-2",
                site="I0002",
                psg_header=psg,
                caisr_header=caisr,
            )
            write_waveform_signals(output, arrays)
            with h5py.File(output, "r") as handle:
                for modality, values in arrays.items():
                    dataset = handle[f"signals/{modality}"]
                    self.assertEqual(dataset.shape, values.shape)
                    self.assertEqual(dataset.dtype, np.dtype("float32"))
                    self.assertIsNotNone(dataset.chunks)
                    self.assertIsNotNone(dataset.compression)
                    self.assertTrue(np.isnan(dataset[:]).all())


if __name__ == "__main__":
    unittest.main()
