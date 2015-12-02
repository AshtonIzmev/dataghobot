from sklearn.cross_validation import train_test_split
import operator
import unittest
import pandas as pd
from dataghobot.featengine import ChaosGeneration
from dataghobot.utils import DataGenerator


class OptimTestCase(unittest.TestCase):

    def test_cross_val_stack(self):
        x, y = DataGenerator.get_digits_data()

        # In order to obtain some categorical columns
        x['i63'] = x['i63'].map(str)
        x['i62'] = x['i62'].map(str)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
        dic = {}
        x_shadow = x_train.copy()

        x_train.loc[:, 'source'] = 0
        x_shadow.loc[:, 'source'] = 1
        x_all = pd.concat([x_train, x_shadow])
        shadow_selector = x_all['source'] == 0
        ChaosGeneration.chaos_feature_importance(x_all, y_train, shadow_selector, feat_dic=dic,
                                                 chaos_feat_iter=10, nb_features=20, chaos_gen_iter=30)
        sorted_x = sorted(dic.items(), key=operator.itemgetter(1))
        print sorted_x
        self.assertGreater(len(dic), len(x_train.columns))
        self.assertGreater(len(dic), len(x_shadow.columns))

    def test_cross_val_stack_none(self):
        x, y = DataGenerator.get_digits_data()

        # In order to obtain some categorical columns
        x['i63'] = x['i63'].map(str)
        x['i62'] = x['i62'].map(str)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
        dic = {}

        x_train.loc[:, 'source'] = 0
        shadow_selector = x_train['source'] == 0
        ChaosGeneration.chaos_feature_importance(x_train, y_train, shadow_selector, feat_dic=dic, chaos_feat_iter=10,
                                                 nb_features=20, chaos_gen_iter=30)
        sorted_x = sorted(dic.items(), key=operator.itemgetter(1))
        print sorted_x
        self.assertGreater(len(dic), len(x_train.columns))


if __name__ == '__main__':
    unittest.main()
