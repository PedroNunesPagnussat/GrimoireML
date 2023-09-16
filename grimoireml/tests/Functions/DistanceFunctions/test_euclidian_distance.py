import unittest
import numpy as np
from src.Functions.DistanceFunctions.distance_function import DistanceFunction
from src.Functions.DistanceFunctions.euclidian_distance import EuclidianDistance  # Assuming this is the correct import path

class TestEuclidianDistance(unittest.TestCase):

    def setUp(self):
        self.distance_function = EuclidianDistance()

    def test_call(self):
        x = np.array([1, 2, 3])
        y = np.array([4, 6, 8])
        result = self.distance_function(x, y)
        self.assertAlmostEqual(result, np.sqrt(50), places=5)

#     def test_within_threshold(self):
#         x = np.array([1, 2, 3])
#         y = np.array([4, 6, 8])
#         self.assertTrue(self.distance_function.within_threshold(x, y, 6))
#         self.assertFalse(self.distance_function.within_threshold(x, y, 4))

#     def test_get_distance_matrix(self):
#         x = np.array([[1, 2], [2, 2], [3, 3]])
#         y = np.array([[2, 2], [3, 3], [4, 4]])
#         result = self.distance_function.get_distance_matrix(x, y)
#         expected_result = np.array([
#             [1.0, np.sqrt(2), np.sqrt(8)],
#             [0.0, np.sqrt(2), np.sqrt(8)],
#             [np.sqrt(2), 0.0, np.sqrt(2)]
#         ])
#         np.testing.assert_almost_equal(result, expected_result, decimal=5)

#     def test_zero_vector(self):
#         x = np.array([0, 0, 0])
#         y = np.array([0, 0, 0])
#         result = self.distance_function(x, y)
#         self.assertEqual(result, 0.0)

#     def test_str(self):
#         self.assertEqual(str(self.distance_function), "Euclidian Distance")

# if __name__ == '__main__':
#     unittest.main()
