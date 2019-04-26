import unittest


class CLassName(unittest.TestCase):

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

    def test_method(self):
        # Setup

        # Execute

        # Test

        pass

'''Allows unit tests to be run from cmd the same way as normal .py file'''
if __name__ == '__main__':
    unittest.main()
