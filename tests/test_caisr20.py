"""CAISR20 纯计算契约测试；不打开 EDF 或 waveform。"""

from __future__ import annotations

import unittest

import numpy as np

from feature_extraction.branches.caisr import CAISR20_DIMENSION, _compute_features


def _prob(rows):
    return np.asarray(rows, dtype=np.float32)


def _inputs(stage, *, probabilities=None, probability_valid=None, arousal=None, respiratory=None, limb=None):
    stage = np.asarray(stage, dtype=np.int8)
    if probabilities is None:
        probabilities = np.full((len(stage), 5), np.nan, dtype=np.float32)
    if probability_valid is None:
        probability_valid = np.zeros(len(stage), dtype=bool)
    return dict(
        stage=stage,
        stage_probability=np.asarray(probabilities, dtype=np.float32),
        stage_probability_valid=np.asarray(probability_valid, dtype=bool),
        arousal=np.zeros((len(stage), 60), dtype=np.int8) if arousal is None else np.asarray(arousal, dtype=np.int8),
        respiratory=np.zeros((len(stage), 30), dtype=np.int8) if respiratory is None else np.asarray(respiratory, dtype=np.int8),
        limb=np.zeros((len(stage), 30), dtype=np.int8) if limb is None else np.asarray(limb, dtype=np.int8),
    )


class Caisr20CalculationTests(unittest.TestCase):
    def test_all_n2_has_tst_but_no_rem_latency(self):
        values, valid, reasons, diagnostics = _compute_features(
            **_inputs([2, 2, 2]),
            availability={"stage": True, "stage_probability": False, "arousal": True, "respiratory": True, "limb": True},
        )
        self.assertEqual(values.shape, (CAISR20_DIMENSION,))
        self.assertEqual(values.dtype, np.float32)
        self.assertEqual(values[0], 90.0)
        self.assertTrue(np.isnan(values[3]))
        self.assertFalse(valid[3])
        self.assertEqual(diagnostics["tst_sec"], 90.0)

    def test_no_valid_sleep_marks_sleep_features_invalid(self):
        values, valid, reasons, _ = _compute_features(
            **_inputs([-1, 5, -1]),
            availability={"stage": True, "stage_probability": False, "arousal": True, "respiratory": True, "limb": True},
        )
        for index in range(0, 12):
            self.assertFalse(valid[index], index)
            self.assertTrue(np.isnan(values[index]), index)
            self.assertTrue(reasons[index])

    def test_sleep_boundaries_compute_sol_rem_latency_and_waso(self):
        values, valid, _, _ = _compute_features(
            **_inputs([5, 5, 2, 2, 5, 4, 4]),
            availability={"stage": True, "stage_probability": False, "arousal": True, "respiratory": True, "limb": True},
        )
        self.assertEqual(values[2], 60.0)
        self.assertEqual(values[3], 90.0)
        self.assertEqual(values[4], 30.0)
        self.assertTrue(valid[2:5].all())

    def test_invalid_stage_gap_breaks_transition_and_bout(self):
        values, valid, _, diagnostics = _compute_features(
            **_inputs([2, 2, -1, 2, 2]),
            availability={"stage": True, "stage_probability": False, "arousal": True, "respiratory": True, "limb": True},
        )
        self.assertEqual(values[9], 0.0)
        self.assertEqual(values[10], 1.0)
        self.assertEqual(diagnostics["sleep_bout_lengths_epochs"], [2, 2])
        self.assertTrue(valid[9:11].all())

    def test_short_bout_boundary_is_strictly_less_than_three_epochs(self):
        values, _, _, _ = _compute_features(
            **_inputs([2, 2, 2, 2, 2, 2]),
            availability={"stage": True, "stage_probability": False, "arousal": True, "respiratory": True, "limb": True},
        )
        self.assertEqual(values[10], 0.0)

    def test_stage_change_starts_a_new_sleep_bout(self):
        values, _, _, diagnostics = _compute_features(
            **_inputs([2, 2, 3, 3]),
            availability={"stage": True, "stage_probability": False, "arousal": True, "respiratory": True, "limb": True},
        )
        self.assertEqual(diagnostics["sleep_bout_lengths_epochs"], [2, 2])
        self.assertEqual(values[10], 1.0)

    def test_probability_entropy_uses_only_complete_valid_rows_without_renormalization(self):
        probabilities = _prob([[1, 0, 0, 0, 0], [0.5, 0.5, np.nan, 0, 0], [0, 1, 0, 0, 0]])
        values, valid, reasons, diagnostics = _compute_features(
            **_inputs([2, 2, 2], probabilities=probabilities, probability_valid=[True, True, True]),
            availability={"stage": True, "stage_probability": True, "arousal": True, "respiratory": True, "limb": True},
        )
        self.assertEqual(values[11], 0.0)
        self.assertTrue(valid[11])
        self.assertEqual(diagnostics["stage_probability_rows_used"], 2)
        self.assertEqual(reasons[11], "")

    def test_available_no_event_is_zero_but_unavailable_is_nan(self):
        available, valid_available, _, _ = _compute_features(
            **_inputs([2, 2]),
            availability={"stage": True, "stage_probability": False, "arousal": True, "respiratory": True, "limb": True},
        )
        unavailable, valid_unavailable, reasons_unavailable, _ = _compute_features(
            **_inputs([2, 2]),
            availability={"stage": True, "stage_probability": False, "arousal": False, "respiratory": False, "limb": False},
        )
        self.assertEqual(available[12], 0.0)
        self.assertEqual(available[15], 0.0)
        self.assertEqual(available[18], 0.0)
        self.assertTrue(valid_available[[12, 15, 18]].all())
        self.assertTrue(np.isnan(unavailable[[12, 15, 18]]).all())
        self.assertFalse(valid_unavailable[[12, 15, 18]].any())
        self.assertTrue(all(reasons_unavailable[index] for index in (12, 15, 18)))

    def test_contiguous_event_segments_and_invalid_gaps_are_preserved(self):
        arousal = np.zeros((1, 60), dtype=np.int8)
        arousal[0, 2:6] = 1
        respiratory = np.zeros((1, 30), dtype=np.int8)
        respiratory[0, 4:7] = 2
        respiratory[0, 7:10] = 3
        respiratory[0, 12:14] = -1
        limb = np.zeros((1, 30), dtype=np.int8)
        limb[0, 2:5] = 2
        values, valid, _, diagnostics = _compute_features(
            **_inputs([2], arousal=arousal, respiratory=respiratory, limb=limb),
            availability={"stage": True, "stage_probability": False, "arousal": True, "respiratory": True, "limb": True},
        )
        self.assertEqual(diagnostics["arousal_segment_count"], 1)
        self.assertEqual(diagnostics["respiratory_segment_count"], 2)
        self.assertEqual(diagnostics["limb_periodic_segment_count"], 1)
        self.assertTrue(valid[[12, 15, 18, 19]].all())
        self.assertGreater(values[12], 0.0)
        self.assertGreater(values[15], 0.0)
        self.assertGreater(values[18], 0.0)

    def test_output_is_fixed_twenty_features_with_reasons(self):
        values, valid, reasons, _ = _compute_features(
            **_inputs([-1]),
            availability={"stage": False, "stage_probability": False, "arousal": False, "respiratory": False, "limb": False},
        )
        self.assertEqual(values.shape, (20,))
        self.assertEqual(valid.shape, (20,))
        self.assertEqual(reasons.shape, (20,))
        self.assertTrue(np.isnan(values).all())
        self.assertFalse(valid.any())
        self.assertTrue(np.all(reasons != ""))


if __name__ == "__main__":
    unittest.main()
