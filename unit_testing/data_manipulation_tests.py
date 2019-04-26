import unittest
from common.data_manipulation import DataManipulation


class TestDataManipulation(unittest.TestCase):

    @classmethod
    def setUpClass(c):
        pass

    @classmethod
    def tearDownClass(c):
        pass

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_min_max_normalise(self):
        # Setup
        features = [12.0, 27.0, 34.0]
        correct_normalised_values = [0.0, 0.6818181818181818, 1.0]

        # Execute
        normalised_values = DataManipulation.min_max_normalise(features)

        # Test
        self.assertEqual(normalised_values, correct_normalised_values)

    def test_normalise_features(self):
        # Setup
        features = [[1.0, 12.0],
                    [2.0, 27.0],
                    [3.0, 34.0]]
        correct_normalised_values = [[0.0, 0.0],
                                     [0.5, 0.6818181818181818],
                                     [1.0, 1.0]]

        # Execute
        normalised_values = DataManipulation.normalise_features(features)

        # Test
        self.assertEqual(normalised_values, correct_normalised_values)


'''Allows unit tests to be run from cmd the same way as normal .py file'''
if __name__ == '__main__':
    unittest.main()
