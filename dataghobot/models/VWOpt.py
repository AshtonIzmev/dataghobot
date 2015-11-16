from sklearn.preprocessing import MinMaxScaler
from wabbit_wappa import *
from GenericOpt import GenericOpt as Gopt


class VWOpt(Gopt):

    epsilon = 1e-15
    scaler = MinMaxScaler(feature_range=(epsilon, 1-epsilon))

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
        return VWOpt.scaler.fit_transform(vw_pred_list_arg)

    @staticmethod
    def build_model(params_arg):
        return VW(loss_function=params_arg['loss_function'],
                  l1=params_arg['l1'], l2=params_arg['l2'],
                  decay_learning_rate=params_arg['decay_learning_rate'],
                  learning_rate=params_arg['learning_rate'])

    @staticmethod
    def create_fit_hopt(x_train, y_train, params_arg):
        clf = VWOpt.build_model(params_arg)
        for _ in range(int(params_arg['passes'])):
            for i, row in x_train.iterrows():
                features = VWOpt.convert_numeric_row(row)
                clf.send_example(VWOpt.convert_label_vw(y_train.ix[i]), features=features)
        return clf

    @staticmethod
    def predict_hopt(clf_arg, x_test):
        res_list = []
        for i, row in x_test.iterrows():
            features = VWOpt.convert_numeric_row(row)
            res_list.append(clf_arg.get_prediction(features).prediction)
        return VWOpt.convert_prediction_vw(res_list)

    @staticmethod
    def assert_params_ok(params_arg):
        Gopt.assert_params_ok(params_arg)
        # vw params
        assert 'loss_function' in params_arg
        assert 'passes' in params_arg
        assert 'l1' in params_arg
        assert 'l2' in params_arg
        assert 'decay_learning_rate' in params_arg
        assert 'learning_rate' in params_arg
