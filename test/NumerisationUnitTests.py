from dataghobot.featengine import Numerisation
from dataghobot.utils import DataGenerator
import unittest
import numpy as np


class OptimTestCase(unittest.TestCase):

    def test_handle_columns(self):
        x, y = DataGenerator.get_iris_data()
        x['feat_todummy'] = 'test'
        x['feat_tobedef'] = x[x.columns[0]].map(lambda s: str(s))
        x['feat_toother'] = x[x.columns[0]].map(lambda s: str(int(3*s)))
        num_cols, todummy_cols, tootherisation_cols, tobedefined_cols = Numerisation.handle_columns(x, [], [])
        self.assertEqual(len(num_cols), 4)
        self.assertEqual(len(todummy_cols), 1)
        self.assertEqual(len(tootherisation_cols), 1)
        self.assertEqual(len(tobedefined_cols), 1)

    def test_handle_columns_entropy_inf(self):
        x, y = DataGenerator.get_iris_data()
        x['feat_tobedef'] = x[x.columns[0]].map(lambda s: str(s))
        x['feat_toother'] = x[x.columns[0]].map(lambda s: str(int(3*s)))
        num_cols, todummy_cols, tootherisation_cols, tobedefined_cols = \
            Numerisation.handle_columns(x, [], [], numerize_entropy_max=np.inf)
        self.assertEqual(len(tootherisation_cols), 2)
        self.assertEqual(len(tobedefined_cols), 0)

    def test_handle_columns_entropy_zero(self):
        x, y = DataGenerator.get_iris_data()
        x['feat_tobedef'] = x[x.columns[0]].map(lambda s: str(s))
        x['feat_toother'] = x[x.columns[0]].map(lambda s: str(int(3*s)))
        num_cols, todummy_cols, tootherisation_cols, tobedefined_cols = \
            Numerisation.handle_columns(x, [], [], numerize_entropy_max=0)
        self.assertEqual(len(todummy_cols), 0)
        self.assertEqual(len(tootherisation_cols), 0)
        self.assertEqual(len(tobedefined_cols), 2)

    def test_treat_dataframe(self):
        x, y = DataGenerator.get_iris_data()
        x['feat_todummy'] = 'test'
        x['feat_tobedef'] = x[x.columns[0]].map(lambda s: str(s))
        x['feat_toother'] = x[x.columns[0]].map(lambda s: str(int(3*s)))
        num_cols, todummy_cols, tootherisation_cols, tobedefined_cols = Numerisation.handle_columns(x, [], [])
        df2 = Numerisation.treat_dataframe(x, num_cols, [], todummy_cols, tootherisation_cols)
        self.assertTrue(df2.applymap(np.isreal).all(1).all())


if __name__ == '__main__':
    unittest.main()
