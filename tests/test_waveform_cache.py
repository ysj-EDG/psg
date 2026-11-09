import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import h5py
import numpy as np

import waveform_cache

from waveform_cache import (
    CANONICAL_CHANNEL_NAMES,
    CACHE_SCHEMA_VERSION,
    EPOCH_DURATION_SEC,
    PREPROCESSING_VERSION,
    TARGET_SAMPLING_RATES,
    EdfHeader,
    write_schema_cache,
)


class WaveformCacheSchemaTests(unittest.TestCase):
    def test_schema_only_cache_contains_metadata_timebase_and_empty_groups(self):
        psg_header = EdfHeader(
            start_datetime="2017-10-12 22:01:55",
            duration_sec=65.0,
            channel_names=("EEG F3", "ECG"),
            native_sampling_rates_hz=(128.0, 200.0),
            native_sample_counts=(8320, 13000),
            physical_dimensions=("uV", "mV"),
        )
        caisr_header = EdfHeader(
            start_datetime="1985-01-01 00:00:00",
            duration_sec=60.0,
            channel_names=("stage_caisr", "arousal_caisr"),
            native_sampling_rates_hz=(1 / 30, 2.0),
            native_sample_counts=(2, 120),
            physical_dimensions=("", ""),
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            output = write_schema_cache(
                Path(temporary_directory) / "smoke.h5",
                record_id="sub-I0002150026011_ses-2",
                site="I0002",
                psg_header=psg_header,
                caisr_header=caisr_header,
            )
            with h5py.File(output, "r") as handle:
                self.assertEqual(handle.attrs["record_id"], "sub-I0002150026011_ses-2")
                self.assertEqual(handle.attrs["site"], "I0002")
                self.assertEqual(handle.attrs["cache_schema_version"], CACHE_SCHEMA_VERSION)
                self.assertEqual(handle.attrs["preprocessing_version"], PREPROCESSING_VERSION)
                self.assertEqual(handle.attrs["complete_epoch_count"], 2)
                self.assertEqual(handle.attrs["epoch_duration_sec"], EPOCH_DURATION_SEC)
                np.testing.assert_array_equal(handle["timebase/epoch_index"][:], [0, 1])
                np.testing.assert_array_equal(handle["timebase/epoch_start_sec"][:], [0.0, 30.0])
                np.testing.assert_array_equal(handle["timebase/epoch_end_sec"][:], [30.0, 60.0])
                self.assertEqual(handle["metadata/psg/start_datetime"][()].decode(), psg_header.start_datetime)
                self.assertEqual(handle["metadata/caisr/start_datetime"][()].decode(), caisr_header.start_datetime)
                self.assertEqual(handle["metadata/caisr/channel_names"].asstr()[:].tolist(), list(caisr_header.channel_names))
                np.testing.assert_array_equal(
                    handle["metadata/caisr/native_sample_count"][:],
                    caisr_header.native_sample_counts,
                )
                provenance_fields = (
                    "channel_names",
                    "candidate_source_labels",
                    "selected_source_labels",
                    "channel_source",
                    "derivation_method",
                    "reference_system",
                    "native_sampling_rate_hz",
                    "native_sample_count",
                    "physical_dimension",
                )
                for modality, names in CANONICAL_CHANNEL_NAMES.items():
                    modality_group = handle[f"metadata/{modality}"]
                    self.assertEqual(
                        modality_group["channel_names"].asstr()[:].tolist(),
                        list(names),
                    )
                    self.assertNotIn("canonical_channel_names", modality_group)
                    for field in provenance_fields:
                        self.assertIn(field, modality_group)
                        self.assertEqual(modality_group[field].shape, (len(names),))
                    self.assertEqual(modality_group["native_sampling_rate_hz"].dtype, np.dtype("float64"))
                    self.assertEqual(modality_group["native_sample_count"].dtype, np.dtype("int64"))
                    for field in (
                        "channel_names",
                        "candidate_source_labels",
                        "selected_source_labels",
                        "channel_source",
                        "derivation_method",
                        "reference_system",
                        "physical_dimension",
                    ):
                        self.assertEqual(modality_group[field].dtype.kind, "O")
                    present = handle[f"quality/channel_present/{modality}"]
                    self.assertEqual(present.shape, (len(names),))
                    self.assertEqual(present.dtype, np.dtype("bool"))
                self.assertEqual(
                    json.loads(handle["metadata/ecg/candidate_source_labels"].asstr()[0]),
                    ["ECG"],
                )
                self.assertTrue(handle["quality/channel_present/ecg"][0])
                self.assertEqual(set(handle["signals"].keys()), set())
                self.assertEqual(set(handle["quality"].keys()), {"channel_present"})
                self.assertEqual(
                    set(handle["quality/channel_present"].keys()),
                    set(CANONICAL_CHANNEL_NAMES),
                )
                self.assertEqual(set(handle["annotations/caisr"].keys()), set())

    def test_print_record_metadata_includes_all_header_channels(self):
        psg_header = EdfHeader(
            start_datetime="2017-10-12 22:01:55",
            duration_sec=65.0,
            channel_names=("F3-M2", "ECG"),
            native_sampling_rates_hz=(500.0, 500.0),
            native_sample_counts=(100, 100),
            physical_dimensions=("uV", "mV"),
        )
        caisr_header = EdfHeader(
            start_datetime="1985-01-01 00:00:00",
            duration_sec=60.0,
            channel_names=("stage_caisr",),
            native_sampling_rates_hz=(1 / 30,),
            native_sample_counts=(2,),
            physical_dimensions=("stage",),
        )
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertTrue(hasattr(waveform_cache, "print_record_metadata"))
            waveform_cache.print_record_metadata(
                record_id="sub-I0002150026011_ses-2",
                site="I0002",
                psg_path=Path("/psg.edf"),
                caisr_path=Path("/caisr.edf"),
                psg_header=psg_header,
                caisr_header=caisr_header,
            )
        report = output.getvalue()
        self.assertIn("record_id: sub-I0002150026011_ses-2", report)
        self.assertIn("site: I0002", report)
        self.assertIn("PSG raw channels:", report)
        self.assertIn("0 | F3-M2 | fs=500.0 | samples=100 | physical_dimension=uV", report)
        self.assertIn("1 | ECG | fs=500.0 | samples=100 | physical_dimension=mV", report)
        self.assertIn("CAISR channels:", report)
        self.assertIn("0 | stage_caisr | fs=0.03333333333333333 | samples=2 | physical_dimension=stage", report)

    def test_source_slot_audit_continues_and_reports_incomplete_records(self):
        complete_labels = (
            "F3-M2", "F4-M1", "C3-M2", "C4-M1", "O1-M2", "O2-M1",
            "E1", "E2", "EKG", "CHIN", "LAT", "RAT",
            "NPT", "THERM", "AIRFLOW", "C-FLOW", "C PRESS",
            "CHEST", "ABDOMINAL", "SaO2",
        )
        incomplete_labels = tuple(label for label in complete_labels if label != "AIRFLOW")
        headers = {
            "complete": EdfHeader(
                start_datetime="2020-01-01 00:00:00",
                duration_sec=60.0,
                channel_names=complete_labels,
                native_sampling_rates_hz=tuple(500.0 for _ in complete_labels),
                native_sample_counts=tuple(100 for _ in complete_labels),
                physical_dimensions=tuple("uV" for _ in complete_labels),
            ),
            "incomplete": EdfHeader(
                start_datetime="2020-01-01 00:00:00",
                duration_sec=60.0,
                channel_names=incomplete_labels,
                native_sampling_rates_hz=tuple(500.0 for _ in incomplete_labels),
                native_sample_counts=tuple(100 for _ in incomplete_labels),
                physical_dimensions=tuple("uV" for _ in incomplete_labels),
            ),
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "I0002").mkdir()
            (root / "I0006").mkdir()
            complete_path = root / "I0002" / "complete.edf"
            incomplete_path = root / "I0006" / "incomplete.edf"
            complete_path.touch()
            incomplete_path.touch()

            def fake_header_reader(path):
                return headers[Path(path).stem]

            output = io.StringIO()
            self.assertTrue(hasattr(waveform_cache, "audit_source_slots"))
            report_path = root / "channel_insufficient.txt"
            summary = waveform_cache.audit_source_slots(
                root,
                expected_records=2,
                header_reader=fake_header_reader,
                stream=output,
                report_path=report_path,
            )
            audit_report = report_path.read_text(encoding="utf-8")

        report = output.getvalue()
        self.assertEqual(summary.total_records, 2)
        self.assertEqual(summary.warning_records, 1)
        self.assertIn("complete (20/20)", report)
        self.assertIn("incomplete (19/20) WARNING", report)
        self.assertIn("AIRFLOW_UNSPECIFIED", report)
        self.assertIn("present_slot_count:", report)
        self.assertIn("missing_slot_count:", report)
        self.assertIn("all_20_slots_present: 1/2", report)
        self.assertNotIn("insufficient_20_slot_records", report)
        self.assertIn("records_scanned: 2", audit_report)
        self.assertIn("\ncomplete\n", audit_report)
        self.assertIn("20/20   1", audit_report)
        self.assertIn("AIRFLOW_UNSPECIFIED    1", audit_report)
        self.assertIn("all_20_slots_present: 1/2", audit_report)
        self.assertIn("affected_record_count: 0", audit_report)

    def test_new_target_rates_are_independent_contract(self):
        self.assertEqual(TARGET_SAMPLING_RATES, {
            "eeg": 128,
            "eog": 128,
            "ecg": 200,
            "emg": 200,
            "resp": 25,
            "spo2": 1,
        })


if __name__ == "__main__":
    unittest.main()
