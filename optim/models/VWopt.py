from sklearn import cross_validation, metrics
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from hyperopt import fmin, tpe, STATUS_OK
from wabbit_wappa import *


class VWopt:

    auc_label = 'auc'
    logloss_label = 'logloss'
    epsilon = 1e-15
    scaler = MinMaxScaler(feature_range=(epsilon, 1-epsilon))

    score = np.iinfo(np.int32).max

    def __init__(self, x_data, y_data, verbose=False,):
        self.verbose = verbose
        self.x_data = x_data
        self.y_data = y_data
        return

    @staticmethod
    def convert_numeric_row(row):
        l = row.values.tolist()
        features = ["{0}:{1}".format(b_, a_) for a_, b_ in zip(l, ['f' + str(i) for i in range(len(l))])
                    if a_ != 0]
        return features

    # Label for logistic/hinge loss must be +1/-1
    @staticmethod
    def convert_label_vw(y_train_arg):
        return 2 * y_train_arg - 1

    @staticmethod
    def convert_prediction_vw(vw_pred_list_arg):
        return VWopt.scaler.fit_transform(vw_pred_list_arg)

    @staticmethod
    def fit(vw, x_train_arg, y_train_arg):
        for i, row in x_train_arg.iterrows():
            features = VWopt.convert_numeric_row(row)
            vw.send_example(VWopt.convert_label_vw(y_train_arg.ix[i]), features=features)

    @staticmethod
    def predict(vw, x_test_arg):
        res_list = []
        for i, row in x_test_arg.iterrows():
            features = VWopt.convert_numeric_row(row)
            res_list.append(vw.get_prediction(features).prediction)
        return VWopt.convert_prediction_vw(res_list)

    def cross_val_pred_vw(self, params_arg):
        kf = cross_validation.KFold(len(self.x_data), n_folds=params_arg['cv'], shuffle=True)
        preds = np.zeros(len(self.x_data))
        for train_index, test_index in kf:
            for _ in range(int(params_arg['passes'])):
                vw = VW(loss_function=params_arg['loss_function'],
                        l1=params_arg['l1'],
                        l2=params_arg['l2'],
                        decay_learning_rate=params_arg['decay_learning_rate'],
                        learning_rate=params_arg['learning_rate'])
            x_train, x_test = self.x_data.iloc[train_index], self.x_data.iloc[test_index]
            y_train = self.y_data.iloc[train_index]
            self.fit(vw, x_train, y_train)
            preds[test_index] = self.predict(vw, x_test)
        return preds

    def get_score_vw(self, params_arg):
        preds_vw = self.cross_val_pred_vw(params_arg)
        if params_arg['eval_metric'] == VWopt.auc_label:
            return -metrics.roc_auc_score(self.y_data, preds_vw)
        if params_arg['eval_metric'] == VWopt.logloss_label:
            return metrics.log_loss(self.y_data, preds_vw)
        raise Exception('Eval metric error : auc or logloss')

    def objective_vw(self, params_arg):
        score = self.get_score_vw(params_arg)
        if score < self.score:
            self.score = score
        if self.verbose:
            print "\tScore {0}\tParams{1}".format(score, params_arg)
        return {'loss': score, 'status': STATUS_OK}

    def run_hp_vw(self, params_arg):
        self.assert_params_ok(params_arg)
        return fmin(self.objective_vw, params_arg, algo=tpe.suggest, max_evals=params_arg['max_evals'])

    @staticmethod
    def assert_params_ok(params_arg):
        # models params
        assert 'cv' in params_arg
        # vw params
        assert 'loss_function' in params_arg
        assert 'passes' in params_arg
        assert 'l1' in params_arg
        assert 'l2' in params_arg
        assert 'decay_learning_rate' in params_arg
        assert 'learning_rate' in params_arg
        # hyperopt params
        assert 'max_evals' in params_arg
        # metric params
        assert 'eval_metric' in params_arg
        assert params_arg['eval_metric'] in [VWopt.auc_label, VWopt.logloss_label]