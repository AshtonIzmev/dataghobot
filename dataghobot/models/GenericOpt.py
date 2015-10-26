from sklearn import metrics, cross_validation
import numpy as np
from hyperopt import fmin, tpe, STATUS_OK


class GenericOpt:

    auc_label = 'auc'
    logloss_label = 'logloss'

    score = np.iinfo(np.int32).max

    def __init__(self, x_data, y_data, verbose=False):
        self.verbose = verbose
        self.x_data = x_data
        self.y_data = y_data
        return

    @staticmethod
    def predict_hopt(clf_arg, preds, test_index, x_test):
        raise Exception('Not meant to be implemented')

    @staticmethod
    def create_fit_hopt(x_train, y_train, params_arg):
        raise Exception('Not meant to be implemented')

    def cross_val_pred(self, params_arg):
        cv = params_arg['cv']
        kf = cross_validation.KFold(len(self.x_data), n_folds=cv, shuffle=True)
        preds = np.zeros(len(self.x_data))
        preds_probas = np.zeros((len(self.x_data)))
        for train_index, test_index in kf:
            x_train, x_test = self.x_data.iloc[train_index], self.x_data.iloc[test_index]
            y_train = self.y_data.iloc[train_index]
            clf_fit = self.create_fit_hopt(x_train, y_train, params_arg)
            preds_probas[test_index] = self.predict_hopt(clf_fit, preds, test_index, x_test)
        return preds_probas

    def get_score(self, params_arg):
        preds_probas_rf = self.cross_val_pred(params_arg)
        if params_arg['eval_metric'] == GenericOpt.auc_label:
            return -metrics.roc_auc_score(self.y_data, preds_probas_rf)
        if params_arg['eval_metric'] == GenericOpt.logloss_label:
            return metrics.log_loss(self.y_data, preds_probas_rf)
        raise Exception('Eval metric error : auc or logloss')

    def objective(self, params_arg):
        score = self.get_score(params_arg)
        if score < self.score:
            self.score = score
        if self.verbose:
            print "\tScore {0}\tParams{1}".format(score, params_arg)
        return {'loss': score, 'status': STATUS_OK}

    def run_hp(self, params_arg):
        self.assert_params_ok(params_arg)
        return fmin(self.objective, params_arg, algo=tpe.suggest, max_evals=params_arg['max_evals'])

    @staticmethod
    def assert_params_ok(params_arg):
        # sklearn params
        assert 'cv' in params_arg
        # hyperopt params
        assert 'max_evals' in params_arg
        # metric params
        assert 'eval_metric' in params_arg
        assert params_arg['eval_metric'] in [GenericOpt.auc_label, GenericOpt.logloss_label]
        return

