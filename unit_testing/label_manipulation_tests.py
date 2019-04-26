import unittest
from common.label_manipulation import LabelManipulator


class TestLabelManipulation(unittest.TestCase):

    @classmethod
    def setUpClass(c):
        pass

    @classmethod
    def tearDownClass(c):
        pass

    def setUp(self):
        self.genre_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 25, 26, 27, 30,
                          31, 32, 33, 36, 37, 38, 41, 42, 43, 45, 46, 47, 49, 53, 58, 63, 64, 65, 66, 70, 71, 74, 76,
                          77, 79, 81, 83, 85, 86, 88, 89, 90, 92, 94, 97, 98, 100, 101, 102, 103, 107, 109, 111, 113, 117,
                          118, 125, 130, 137, 138, 166, 167, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180,
                          181, 182, 183, 184, 185, 186, 187, 188, 189, 214, 224, 232, 236, 240, 247, 250, 267, 286, 296,
                          297, 311, 314, 322, 337, 359, 360, 361, 362, 374, 377, 378, 400, 401, 404, 428, 439, 440, 441,
                          442, 443, 444, 456, 465, 468, 491, 493, 495, 502, 504, 514, 524, 538, 539, 542, 567, 580, 602,
                          619, 651, 659, 693, 695, 741, 763, 808, 810, 811, 906, 1032, 1060, 1156, 1193, 1235]
        self.top_level_genre_ids = [2, 3, 4, 58, 9, 10, 12, 13, 14, 15, 17, 20, 21, 38, 1235]
        self.lm = LabelManipulator()

    def tearDown(self):
        pass

    def test_categorise_genre(self):
        # Setup
        self.allowed_values = range(0, 16)

        # Execute Loop
        for index, id in enumerate(top_level_genre_ids):
            # Test
            self.assertEqual(self.lm.categorise_genre(id), index)

    def test_load_genres(self):
        # Setup
        # check that the value holders are blank as expected
        self.assertEqual(self.lm.genre_top_levels, [])
        self.assertEqual(self.lm.top_level_genres, [])

        # Execute
        self.lm.load_genres()

        # Test
        # self.assertEqual(self.lm.genre_top_levels, self.top_level_genre_ids)
        self.assertEqual(self.lm.top_level_genres, self.top_level_genre_ids)


'''Allows unit tests to be run from cmd the same way as normal .py file'''
if __name__ == '__main__':
    unittest.main()
