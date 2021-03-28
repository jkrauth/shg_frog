""" Testing """
from logging import setLoggerClass
import unittest
import numpy as np
from shg_frog.model import phase_retrieval

class TestRMSDiff(unittest.TestCase):
    """ rms diff tests """

    def setUp(self) -> None:
        vec1 = np.array([1, 3, 2, 2])
        vec2 = np.array([1, 1, 1, 2])
        self.rms_diff = phase_retrieval.rms_diff(vec1, vec2)

    def test_result(self):
        self.assertEqual(
            self.rms_diff,
            1.118033988749895,
            "Should be the rms difference"
        )

    def test_type(self):
        self.assertIsInstance(self.rms_diff, float, "Should be float")


class TestMakeAxis(unittest.TestCase):
    """ make axis tests """

    def setUp(self) -> None:
        self.axis = np.array([-2., -1., 0., 1.])
        length = len(self.axis)
        step = self.axis[1] - self.axis[0]
        self.result = phase_retrieval.make_axis(length, step)


    def test_entries(self):
        self.assertSequenceEqual(
            self.result.tolist(),
            self.axis.tolist()
        )

    def test_type(self):
        self.assertIsInstance(self.result, np.ndarray)


class TestGetFWHM(unittest.TestCase):
    """ get fwhm tests """
    def setUp(self) -> None:
        intensity = np.array([0., 1., 0.])
        axis = np.array([0., 1., 2.])
        self.result = phase_retrieval.get_fwhm(intensity, axis)

    def test_fwhm(self):
        self.assertEqual(self.result, 1.)

if __name__ == "__main__":
    unittest.main()
