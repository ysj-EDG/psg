import unittest
from pathlib import Path

import numpy as np

from audit_spo2 import (
    build_summary,
    classify_scale_hint,
    is_integer_like,
    select_records,
    summarize_raw_spo2,
)


class SpO2AuditTests(unittest.TestCase):
    def test_raw_statistics_keep_nonfinite_counts_and_finite_scale_fractions(self):
        stats = summarize_raw_spo2(
            np.array([0.5, 1.0, np.nan, np.inf, -np.inf], dtype=float)
        )
        self.assertEqual(stats["nan_count"], 1)
        self.assertEqual(stats["posinf_count"], 1)
        self.assertEqual(stats["neginf_count"], 1)
        self.assertAlmostEqual(stats["finite_fraction"], 0.4)
        self.assertEqual(stats["fraction_in_0_1_5"], 1.0)
        self.assertEqual(stats["fraction_in_50_100_5"], 0.0)

    def test_scale_hint_and_integer_like_rules(self):
        self.assertEqual(classify_scale_hint(0.95), "fraction_like")
        self.assertEqual(classify_scale_hint(95.0), "percent_like")
        self.assertEqual(classify_scale_hint(10.0), "other")
        self.assertTrue(is_integer_like(25.0))
        self.assertFalse(is_integer_like(25.5))

    def test_select_records_takes_stable_three_per_site(self):
        records = tuple(
            (site, f"{site}_{index}", Path(f"{site}_{index}.edf"))
            for site in ("I0002", "I0006", "S0001")
            for index in range(5)
        )
        selected = select_records(records, per_site=3)
        self.assertEqual(len(selected), 9)
        self.assertEqual(
            [record[1] for record in selected],
            [
                "I0002_0", "I0002_1", "I0002_2",
                "I0006_0", "I0006_1", "I0006_2",
                "S0001_0", "S0001_1", "S0001_2",
            ],
        )

    def test_summary_reports_present_missing_and_scale_counts(self):
        fraction = summarize_raw_spo2(np.array([0.8, 0.9]))
        percent = summarize_raw_spo2(np.array([95.0, 96.0]))
        rows = [
            {
                "record_id": "fraction",
                "site": "I0002",
                "channel_present": True,
                "physical_dimension": "%",
                "native_fs": 25.0,
                "scale_hint": "fraction_like",
                "_raw_stats": fraction,
                **fraction,
            },
            {
                "record_id": "percent",
                "site": "I0006",
                "channel_present": True,
                "physical_dimension": "%",
                "native_fs": 25.0,
                "scale_hint": "percent_like",
                "_raw_stats": percent,
                **percent,
            },
            {
                "record_id": "missing",
                "site": "S0001",
                "channel_present": False,
                "physical_dimension": "",
                "native_fs": float("nan"),
                "scale_hint": "missing",
                "_raw_stats": {},
            },
        ]
        summary = build_summary(rows, expected_records=3)
        self.assertIn("records scanned: 3", summary)
        self.assertIn("SpO2 present: 2", summary)
        self.assertIn("SpO2 missing: 1", summary)
        self.assertIn("per-second reshape + nanmedian is structurally applicable", summary)


if __name__ == "__main__":
    unittest.main()
