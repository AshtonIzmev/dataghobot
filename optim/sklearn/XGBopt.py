from sklearn import cross_validation, metrics
import numpy as np
import xgboost as xgb
import re
from hyperopt import fmin, tpe, STATUS_OK


class XGBopt:

    def __init__(self, x_data, y_data, verbose=False,):
        self.verbose = verbose
        self.x_data = x_data
        self.y_data = y_data
        return

    def cross_val_pred_xgb(self, params_arg):
        # remove non-valid xgb columns name caracters
        def straln(s):
            return re.sub(r'[^a-zA-Z0-9]+', 'xx', s)

        kf = cross_validation.KFold(len(self.x_data), n_folds=params_arg['cv'], shuffle=True)
        preds_probas = np.zeros(len(self.x_data))
        for train_index, test_index in kf:
            x_train, x_test = self.x_data.iloc[train_index], self.x_data.iloc[test_index]
            y_train = self.y_data.iloc[train_index]
            y_test = self.y_data.iloc[test_index]
            dtrain = xgb.DMatrix(x_train.rename(columns=straln), label=y_train)
            dtest = xgb.DMatrix(x_test.rename(columns=straln), label=y_test)
            param = {
                'booster': params_arg['booster'],
                'objective': params_arg['objective'],
                'eta': params_arg['eta'],
                'gamma': params_arg['gamma'],
                'min_child_weight': params_arg['min_child_weight'],
                'max_depth': params_arg['max_depth'],
                'subsample': params_arg['subsample'],
                'colsample_bytree': params_arg['colsample_bytree'],
                'num_round': params_arg['num_round'],
                'nthread': params_arg['nthread'],
                'silent': params_arg['silent'],
                'seed': params_arg['seed']
            }
            plst = param.items()
            plst += [('eval_metric', params_arg['eval_metric'])]

            evallist = [(dtest, 'eval'), (dtrain, 'train')]

            bst = xgb.train(plst, dtrain, 25, evallist, verbose_eval=False)
            preds_probas[test_index] = bst.predict(dtest)
        return preds_probas

    def get_score_xgb(self, params_arg):
        preds_probas_rf = self.cross_val_pred_xgb(params_arg)
        return metrics.roc_auc_score(self.y_data, preds_probas_rf)

    def objective_xgb(self, params_arg):
        auc_score = self.get_score_xgb(params_arg)
        if self.verbose:
            print "\tScore {0}".format(auc_score)
        return {'loss': -auc_score, 'status': STATUS_OK}

    def run_hp_xgb(self, params_arg):
        self.assert_params_ok(params_arg)
        return fmin(self.objective_xgb, params_arg, algo=tpe.suggest, max_evals=params_arg['max_evals'])

    @staticmethod
    def assert_params_ok(params_arg):
        # sklearn params
        assert 'cv' in params_arg
        # xgb params
        assert 'booster' in params_arg
        assert 'objective' in params_arg
        assert 'eta' in params_arg
        assert 'gamma' in params_arg
        assert 'min_child_weight' in params_arg
        assert 'max_depth' in params_arg
        assert 'subsample' in params_arg
        assert 'colsample_bytree' in params_arg
        assert 'num_round' in params_arg
        assert 'nthread' in params_arg
        assert 'silent' in params_arg
        assert 'seed' in params_arg
        # hyperopt params
        assert 'max_evals' in params_arg
        # metric params
        assert 'eval_metric' in params_arg
