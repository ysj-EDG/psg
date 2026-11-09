import unittest

import numpy as np

from waveform_primitives import pad_or_trim, reshape_epoch_signal


class WaveformPrimitiveTests(unittest.TestCase):
    def test_pad_or_trim_uses_nan_only_for_missing_tail(self):
        actual = pad_or_trim(np.array([1.0, 2.0]), 4)
        expected = np.array([1.0, 2.0, np.nan, np.nan])
        self.assertTrue(np.allclose(actual, expected, equal_nan=True))
        np.testing.assert_array_equal(pad_or_trim(np.arange(4.0), 2), [0.0, 1.0])

    def test_reshape_epoch_signal_keeps_contiguous_epoch_order(self):
        values = np.arange(60.0)
        reshaped = reshape_epoch_signal(values, n_epochs=2, sampling_rate=1)
        self.assertEqual(reshaped.shape, (2, 30))
        np.testing.assert_array_equal(reshaped[0], np.arange(30.0))
        np.testing.assert_array_equal(reshaped[1], np.arange(30.0, 60.0))


if __name__ == "__main__":
    unittest.main()
