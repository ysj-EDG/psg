import unittest

from channel_sources import (
    CANONICAL_BY_MODALITY,
    SourcePlan,
    build_source_plans,
    build_source_specs,
    find_resp_source_conflicts,
)


class ChannelSourceTests(unittest.TestCase):
    def test_canonical_slots_and_direct_sources_are_stable(self):
        labels = [
            "F3-M2", "F4-M1", "C3-M2", "C4-M1", "O1-M2", "O2-M1",
            "E1-M2", "E2-M1", "ECG", "CHIN", "LAT", "RAT",
            "ptaf", "thermal airflow", "airflow", "cflow", "cpres",
            "chest", "abd", "sao2",
        ]
        specs = build_source_specs(labels)
        self.assertEqual(
            {modality: len(names) for modality, names in CANONICAL_BY_MODALITY.items()},
            {"eeg": 6, "eog": 2, "ecg": 1, "emg": 3, "resp": 7, "spo2": 1},
        )
        self.assertEqual(specs["EEG_F3"].method, "direct")
        self.assertEqual(specs["EEG_F3"].indices, (0,))
        self.assertEqual(specs["EOG_E1"].indices, (6,))
        self.assertEqual(specs["ECG"].indices, (8,))
        self.assertEqual(specs["NASAL_PRESSURE"].indices, (12,))
        self.assertEqual(specs["SPO2"].indices, (19,))

    def test_bipolar_components_preserve_positive_minus_reference_semantics(self):
        labels = [
            "F3", "A2", "F4", "A1", "E1", "M2", "E2", "M1",
            "CHIN1", "CHIN2", "LLEG+", "LLEG-", "RLEG+", "RLEG-",
            "ECG", "abd", "chest", "ptaf", "airflow", "cflow", "cpres", "sao2",
        ]
        specs = build_source_specs(labels)
        self.assertEqual(specs["EEG_F3"].method, "component_minus_reference")
        self.assertEqual(specs["EEG_F3"].indices, (0, 1))
        self.assertEqual(specs["EEG_F3"].reference, "M2")
        self.assertEqual(specs["EOG_E1"].method, "component_minus_reference")
        self.assertEqual(specs["CHIN_EMG"].method, "positive_minus_negative")
        self.assertEqual(specs["LEG_EMG_LEFT"].method, "positive_minus_negative")
        self.assertEqual(specs["LEG_EMG_RIGHT"].method, "positive_minus_negative")

    def test_canonical_names_are_frozen(self):
        self.assertEqual(
            CANONICAL_BY_MODALITY,
            {
                "eeg": ("EEG_F3", "EEG_F4", "EEG_C3", "EEG_C4", "EEG_O1", "EEG_O2"),
                "eog": ("EOG_E1", "EOG_E2"),
                "ecg": ("ECG",),
                "emg": ("CHIN_EMG", "LEG_EMG_LEFT", "LEG_EMG_RIGHT"),
                "resp": (
                    "NASAL_PRESSURE", "THERMAL_AIRFLOW", "AIRFLOW_UNSPECIFIED",
                    "PAP_FLOW", "PAP_PRESSURE", "THORACIC_EFFORT", "ABDOMINAL_EFFORT",
                ),
                "spo2": ("SPO2",),
            },
        )

    def test_resp_duplicate_source_detector_reports_slot_pair(self):
        plans = {}
        for modality, names in CANONICAL_BY_MODALITY.items():
            for name in names:
                plans[name] = SourcePlan(
                    canonical_name=name,
                    candidate_source_labels=(),
                    selected_source_labels=(),
                    channel_present=False,
                    channel_source="unavailable",
                    derivation_method="unavailable",
                    reference_system="",
                    native_sampling_rate_hz=float("nan"),
                    native_sample_count=-1,
                    physical_dimension="",
                )
        plans["NASAL_PRESSURE"] = SourcePlan(
            **{**plans["NASAL_PRESSURE"].__dict__, "selected_source_indices": (7,)}
        )
        plans["PAP_FLOW"] = SourcePlan(
            **{**plans["PAP_FLOW"].__dict__, "selected_source_indices": (7,)}
        )
        self.assertEqual(
            find_resp_source_conflicts(plans),
            ((7, "NASAL_PRESSURE", "PAP_FLOW"),),
        )

    def test_source_plan_retains_candidates_and_reports_duplicate_priority(self):
        labels = ["F3-M2", "F3-M2", "F3", "A2"]
        plans = build_source_plans(
            labels,
            native_sampling_rates_hz=(500.0, 500.0, 500.0, 500.0),
            native_sample_counts=(10, 10, 10, 10),
            physical_dimensions=("uV", "uV", "uV", "uV"),
        )
        plan = plans["EEG_F3"]
        self.assertEqual(plan.candidate_source_labels, ("F3-M2", "F3-M2", "F3", "A2"))
        self.assertEqual(plan.selected_source_labels, ("F3-M2",))
        self.assertTrue(plan.channel_present)
        self.assertEqual(plan.channel_source, "direct")
        self.assertEqual(plan.derivation_method, "direct")
        self.assertEqual(plan.native_sampling_rate_hz, 500.0)
        self.assertEqual(plan.native_sample_count, 10)
        self.assertEqual(plan.physical_dimension, "uV")
        self.assertIn("multiple_same_priority_direct_candidates", plan.anomalies)

    def test_source_plan_keeps_unresolved_airflow_candidate(self):
        plans = build_source_plans(
            ["THERM"],
            native_sampling_rates_hz=(500.0,),
            native_sample_counts=(20,),
            physical_dimensions=("uV",),
        )
        plan = plans["AIRFLOW_UNSPECIFIED"]
        self.assertEqual(plan.candidate_source_labels, ("THERM",))
        self.assertEqual(plan.selected_source_labels, ())
        self.assertFalse(plan.channel_present)
        self.assertEqual(plan.channel_source, "unavailable")
        self.assertIn("source_unresolved", plan.anomalies)


if __name__ == "__main__":
    unittest.main()
