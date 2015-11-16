from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from dataghobot.models import SklearnOpt, XGBOpt, VWOpt
from dataghobot.hyperopt import HyperoptParam
import pandas as pd
import numpy as np


def get_stack_opt(gopt, goptparam, x_test1, x_test2):
    best = goptparam[0]
    for k, v in goptparam[1].iteritems():
        if k not in best:
            best[k] = v
    opt_clf = gopt.create_fit_hopt(gopt.x_data, gopt.y_data, best)
    return gopt.predict_hopt(opt_clf, x_test1), gopt.predict_hopt(opt_clf, x_test2)


def get_best_xgbopt(x, y):
    xgbopt = XGBOpt.XGBOpt(x, y)
    param = HyperoptParam.HyperoptParam.param_space_reg_xgb_tree
    param['eval_metric'] = 'auc'
    return xgbopt.run_hp(param), param


def get_best_sklopt(x, y):
    skopt = SklearnOpt.SklearnOpt(x, y)
    param = HyperoptParam.HyperoptParam.param_space_reg_skl_rf
    param['eval_metric'] = 'auc'
    param['type'] = 'random_forest'
    return skopt.run_hp(param), param


def get_best_etopt(x, y):
    skopt = SklearnOpt.SklearnOpt(x, y)
    param = HyperoptParam.HyperoptParam.param_space_reg_skl_rf
    param['eval_metric'] = 'auc'
    param['type'] = 'extra_trees'
    return skopt.run_hp(param), param


def cross_val_stack(x_train, y_train, x_test, xgbparam, sklparam, etparams, cv=3):
    kf = cross_validation.KFold(len(x_train), n_folds=cv, shuffle=True)
    res = []
    for train_train_index, train_stack_index in kf:
        x_train_train = x_train.iloc[train_train_index]
        y_train_train = y_train.iloc[train_train_index]
        x_train_stack = x_train.iloc[train_stack_index]
        y_train_stack = y_train.iloc[train_stack_index]

        xgbopt = XGBOpt.XGBOpt(x_train_train, y_train_train)
        y_pred_stack_1, y_pred_test_1 = get_stack_opt(xgbopt, xgbparam, x_train_stack, x_test)

        skopt = SklearnOpt.SklearnOpt(x_train_train, y_train_train)
        y_pred_stack_2, y_pred_test_2 = get_stack_opt(skopt, sklparam, x_train_stack, x_test)

        skopt = SklearnOpt.SklearnOpt(x_train_train, y_train_train)
        y_pred_stack_3, y_pred_test_3 = get_stack_opt(skopt, etparams, x_train_stack, x_test)

        x_pred_stack = pd.DataFrame(np.transpose(np.array([y_pred_stack_1, y_pred_stack_2, y_pred_stack_3])))
        x_pred_test = pd.DataFrame(np.transpose(np.array([y_pred_test_1, y_pred_test_2, y_pred_test_3])))
            
        lr = LogisticRegression()
        lr.fit(x_pred_stack, y_train_stack)
        res.append(lr.predict_proba(x_pred_test))
    return res
