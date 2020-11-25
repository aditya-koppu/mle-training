import unittest

import scripts.data_test_train


class Testdata_test_train(unittest.TestCase):
    def test_data(self):
        res = scripts.data_test_train.load_housing_data()
        self.assertIsNotNone(res)


if __name__ == "__main__":
    unittest.main()
