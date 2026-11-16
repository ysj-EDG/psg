"""CAISR20 HDF5 审计汇总测试。"""

import unittest

import numpy as np

from feature_extraction.branches.caisr import CAISR20_FEATURE_NAMES
from feature_extraction.check.caisr.caisr_audit import summarize_records


class CaisrAuditTests(unittest.TestCase):
    def test_summary_counts_invalid_features_by_site(self):
        values = np.ones(20, dtype=np.float32)
        valid = np.ones(20, dtype=bool)
        valid[11] = False
        reasons = np.full(20, "", dtype="U64")
        reasons[11] = "caisr_unavailable"
        summary = summarize_records([("r1", "I0006", "manual", values, valid, reasons)])
        self.assertEqual(summary["features"][CAISR20_FEATURE_NAMES[11]]["invalid"], 1)
        self.assertEqual(summary["features"][CAISR20_FEATURE_NAMES[11]]["by_site"]["I0006"]["invalid"], 1)
        self.assertEqual(summary["features"][CAISR20_FEATURE_NAMES[11]]["reasons"]["caisr_unavailable"], 1)
        self.assertEqual(summary["records_extracted"], 1)

    def test_summary_keeps_source_and_site_totals(self):
        values = np.zeros(20, dtype=np.float32)
        valid = np.ones(20, dtype=bool)
        reasons = np.full(20, "", dtype="U64")
        summary = summarize_records([
            ("r1", "I0002", "auto", values, valid, reasons),
            ("r2", "S0001", "manual", values, valid, reasons),
        ])
        self.assertEqual(summary["records_scanned"], 2)
        self.assertEqual(summary["sources"]["auto"], 1)
        self.assertEqual(summary["sources"]["manual"], 1)
        self.assertEqual(summary["sites"]["I0002"], 1)
        self.assertEqual(summary["sites"]["S0001"], 1)


if __name__ == "__main__":
    unittest.main()
