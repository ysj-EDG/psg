import unittest

from waveform_record import epoch_bounds, expected_output_names, inspection_epochs


class WaveformRecordInspectionTests(unittest.TestCase):
    def test_inspection_epochs_are_start_middle_end(self):
        self.assertEqual(
            inspection_epochs(960),
            (("START", 0), ("MIDDLE", 480), ("END", 959)),
        )

    def test_epoch_bounds_use_psg_relative_seconds(self):
        self.assertEqual(epoch_bounds(0, 500.0), (0, 15000))
        self.assertEqual(epoch_bounds(480, 128.0), (1843200, 1847040))

    def test_expected_output_names_are_six_png_and_one_txt(self):
        names = expected_output_names("sub-I0002150026011_ses-2")
        self.assertEqual(
            names,
            (
                "eeg.png",
                "eog.png",
                "ecg.png",
                "emg.png",
                "resp.png",
                "spo2.png",
                "sub-I0002150026011_ses-2.txt",
            ),
        )


if __name__ == "__main__":
    unittest.main()
