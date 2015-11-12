from dataghobot.featengine import Numerisation
from sklearn import datasets
import unittest
import pandas as pd
import numpy as np


class OptimTestCase(unittest.TestCase):

    def test_handle_columns(self):
        dataset = datasets.load_iris()
        df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
        df['feat_todummy'] = 'test'
        df['feat_tobedef'] = df[df.columns[0]].map(lambda s: str(s))
        df['feat_toother'] = df[df.columns[0]].map(lambda s: str(int(3*s)))
        num_cols, todummy_cols, tootherisation_cols, tobedefined_cols = Numerisation.handle_columns(df, [], [])
        self.assertEqual(len(num_cols), 4)
        self.assertEqual(len(todummy_cols), 1)
        self.assertEqual(len(tootherisation_cols), 1)
        self.assertEqual(len(tobedefined_cols), 1)

    def test_handle_columns_entropy_inf(self):
        dataset = datasets.load_iris()
        df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
        df['feat_tobedef'] = df[df.columns[0]].map(lambda s: str(s))
        df['feat_toother'] = df[df.columns[0]].map(lambda s: str(int(3*s)))
        num_cols, todummy_cols, tootherisation_cols, tobedefined_cols = \
            Numerisation.handle_columns(df, [], [], entropy_max=np.inf)
        self.assertEqual(len(tootherisation_cols), 2)
        self.assertEqual(len(tobedefined_cols), 0)

    def test_handle_columns_entropy_zero(self):
        dataset = datasets.load_iris()
        df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
        df['feat_tobedef'] = df[df.columns[0]].map(lambda s: str(s))
        df['feat_toother'] = df[df.columns[0]].map(lambda s: str(int(3*s)))
        num_cols, todummy_cols, tootherisation_cols, tobedefined_cols = \
            Numerisation.handle_columns(df, [], [], entropy_max=0)
        self.assertEqual(len(todummy_cols), 0)
        self.assertEqual(len(tootherisation_cols), 0)
        self.assertEqual(len(tobedefined_cols), 2)

    def test_treat_dataframe(self):
        dataset = datasets.load_iris()
        df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
        df['feat_todummy'] = 'test'
        df['feat_tobedef'] = df[df.columns[0]].map(lambda s: str(s))
        df['feat_toother'] = df[df.columns[0]].map(lambda s: str(int(3*s)))
        num_cols, todummy_cols, tootherisation_cols, tobedefined_cols = Numerisation.handle_columns(df, [], [])
        df2 = Numerisation.treat_dataframe(df, num_cols, [], todummy_cols, tootherisation_cols)
        self.assertTrue(df2.applymap(np.isreal).all(1).all())


if __name__ == '__main__':
    unittest.main()
