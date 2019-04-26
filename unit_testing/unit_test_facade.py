import unittest
from unit_testing.data_manipulation_tests import TestDataManipulation
from unit_testing.label_manipulation_tests import TestLabelManipulation


def run_unit_tests():
    test_classes_to_run = [TestDataManipulation, TestLabelManipulation]

    loader = unittest.TestLoader()

    suites_list = []
    for test_class in test_classes_to_run:
        suite = loader.loadTestsFromTestCase(test_class)
        suites_list.append(suite)

    big_suite = unittest.TestSuite(suites_list)

    runner = unittest.TextTestRunner()
    results = runner.run(big_suite)
    print(results)
