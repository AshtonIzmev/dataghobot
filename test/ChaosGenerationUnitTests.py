from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
import operator
import unittest
import pandas as pd
from dataghobot.featengine import ChaosGeneration


class OptimTestCase(unittest.TestCase):

    def test_cross_val_stack(self):
        data = load_digits(2)
        cols = ['i'+str(i) for i in range(data.data.shape[1])]
        x = pd.DataFrame(data.data, columns=cols)
        y = pd.Series(data.target)
        # In order to obtain some categorical columns
        x['i63'] = x['i63'].map(str)
        x['i62'] = x['i62'].map(str)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
        dic = {}
        ChaosGeneration.chaos_feature_importance(dic, x_train, y_train, chaos_feat_iter=10,
                                                 chaos_gen_iter=30, nb_features=20)
        sorted_x = sorted(dic.items(), key=operator.itemgetter(1))
        print sorted_x
        self.assertGreater(len(dic), len(x_train.columns))


if __name__ == '__main__':
    unittest.main()
