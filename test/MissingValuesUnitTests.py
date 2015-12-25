from dataghobot.featengine import MissingValues
import unittest
import pandas as pd
import numpy as np


class OptimTestCase(unittest.TestCase):

    def test_missingValue(self):
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [1, np.NaN, 4]})
        MissingValues.add_miss_val_indicator(df)
        self.assertTrue((df.b_is_nan.values == [0, 1, 0]).all())

    def test_GivenMissingValue(self):
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [1, np.NaN, 4]})
        MissingValues.add_miss_val_indicator(df, miss_val=2)
        self.assertTrue((df.a_is_nan.values == [0, 1, 0]).all())

    def test_GivenMissingValueDic(self):
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [1, np.NaN, 4]})
        MissingValues.add_miss_val_indicator_from_dic(df, miss_dic={'a': 1})
        self.assertTrue((df.a_is_nan.values == [1, 0, 0]).all())

    def test_fill_mean(self):
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [1, np.NaN, 4]})
        MissingValues.fill_with_mean(df)
        self.assertTrue((df.b.values == [1, 2.5, 4]).all())

    def test_fill_mean_object(self):
        df = pd.DataFrame({'a': [1, 2, 3], 'b': ['1', np.NaN, 4]})
        MissingValues.fill_with_mean(df)
        self.assertTrue(np.isnan(df.b.values[1]).all())


if __name__ == '__main__':
    unittest.main()
