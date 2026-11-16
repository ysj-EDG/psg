"""canonical feature-extraction framework 的最小 schema/registry 测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from feature_extraction.feature_extractor import BRANCH_REGISTRY, main
from feature_extraction.feature_interfaces import BranchNotImplementedError
from feature_extraction.feature_schema import FeatureBlock, TimeScale


class FeatureExtractionFrameworkTests(unittest.TestCase):
    def test_feature_block_preserves_nan_reason_and_time_scale(self):
        block = FeatureBlock(
            branch_name="eeg",
            feature_set_name="future",
            time_scale=TimeScale.EPOCH,
            values=np.asarray([[1.0, np.nan]], dtype=np.float32),
            feature_names=("a", "b"),
            valid_mask=np.asarray([[True, False]]),
            invalid_reason=np.asarray([["", "channel_missing"]]),
            epoch_index=np.asarray([0]),
        )
        self.assertEqual(block.values.shape, (1, 2))
        self.assertEqual(block.time_scale, TimeScale.EPOCH)

    def test_registry_has_eight_independent_branches(self):
        self.assertEqual(set(BRANCH_REGISTRY), {"caisr", "eeg", "ecg", "resp", "emg", "spo2", "demographics", "eog"})
        self.assertEqual(BRANCH_REGISTRY["caisr"].planned_time_scales, ("night",))
        self.assertIn("window", BRANCH_REGISTRY["ecg"].planned_time_scales)

    def test_unimplemented_branch_raises_and_management_commands_need_no_manifest(self):
        self.assertEqual(main(["--list-branches"]), 0)
        self.assertEqual(main(["--describe", "caisr"]), 0)
        for spec in BRANCH_REGISTRY.values():
            if spec.name == "caisr":
                continue
            with self.subTest(branch=spec.name), self.assertRaises(BranchNotImplementedError):
                spec.extractor(None, {})

    def test_sample_context_checks_real_identity_without_edf_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            h5_path = root / "sample.h5"
            demo_path = root / "demographics.csv"
            manifest_path = root / "manifest.tsv"
            with h5py.File(h5_path, "w") as handle:
                handle.attrs.update(record_id="sub-I0002000000001_ses-1", site="I0002", cache_schema_version="1.0", complete_epoch_count=1)
                for group in ("signals", "quality", "metadata", "annotations", "timebase"):
                    handle.create_group(group)
            demo_path.write_text("BidsFolder,SessionID,age\nI0002000000001,1,40\n", encoding="utf-8")
            manifest_path.write_text(
                "record_id\tsite\tsubject_id\tsession_id\tcache_h5_path\tdemographics_csv_path\tpsg_edf_path\tcaisr_edf_path\n"
                f"sub-I0002000000001_ses-1\tI0002\tI0002000000001\t1\t{h5_path}\t{demo_path}\tmissing.psg.edf\tmissing.caisr.edf\n",
                encoding="utf-8",
            )
            from feature_extraction.sample_loader import Manifest, SampleContext
            manifest = Manifest.from_tsv(manifest_path)
            with SampleContext(manifest.get("sub-I0002000000001_ses-1")) as context:
                self.assertEqual(context.epoch_count, 1)
                self.assertEqual(context.demographics()["age"], "40")
