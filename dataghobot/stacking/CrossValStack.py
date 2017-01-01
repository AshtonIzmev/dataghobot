from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from dataghobot.models import SklearnOpt, XGBOpt
import pandas as pd
import numpy as np
import logging
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def predict_opt_clf(gopt, goptparam, x_test1, x_test2):
    best = goptparam[0]
    for k, v in goptparam[1].iteritems():
        if k not in best:
            best[k] = v
    opt_clf = gopt.create_fit_hopt(gopt.x_data, gopt.y_data, best)
    return gopt.predict_hopt(opt_clf, x_test1), gopt.predict_hopt(opt_clf, x_test2)


def get_best_xgbopt(x, y, params):
    return XGBOpt.XGBOpt(x, y).run_hp(params), params


def get_best_sklopt(x, y, params):
    return SklearnOpt.SklearnOpt(x, y).run_hp(params), params


def get_best_etopt(x, y, params):
    return SklearnOpt.SklearnOpt(x, y).run_hp(params), params


def cross_val_stack(x_train, y_train, x_test, xgbparam, sklparam, etparams, **cross_val_stack_args):

    csvstack_cv = cross_val_stack_args.get('csvstack_cv', 5)
    res = []
    i = 0
    logging.info(" >>> DGH >>> Cross-val-stacking")
    for train_train_idx, train_stack_idx in cross_validation.KFold(len(x_train), n_folds=csvstack_cv, shuffle=True):
        logging.info(" >>> DGH >>> Cross-val-stacking round " + str(i))
        y_probas = stack_that(x_train, y_train, x_test, train_train_idx, train_stack_idx, sklparam, etparams, xgbparam)
        res.append(y_probas)
    return res


def cross_val_meta_stack(x_train, y_train, x_test, xgbparam, sklparam, etparams, **cross_val_stack_args):

    csvstack_cv = cross_val_stack_args.get('csvstack_cv', 3)
    res = []
    i = 0
    logging.info(" >>> DGH >>> Cross-val-meta-stacking")
    for train_train_idx, train_stack_idx in cross_validation.KFold(len(x_train), n_folds=csvstack_cv, shuffle=True):
        logging.info(" >>> DGH >>> Cross-val-meta-stacking round " + str(i))
        y_probas = meta_stack_that(x_train, y_train, x_test, train_train_idx, train_stack_idx, sklparam, etparams, xgbparam)
        res.append(y_probas)
    return res


def stack_that(x_train, y_train, x_test, train_idx, stack_idx, rfparams, extparams, xgbparams):
    x_train_train = x_train.iloc[train_idx]
    y_train_train = y_train.iloc[train_idx]
    x_train_stack = x_train.iloc[stack_idx]
    y_train_stack = y_train.iloc[stack_idx]

    logging.info(" >>> DGH >>>  prediction")
    xgbopt = XGBOpt.XGBOpt(x_train_train, y_train_train)
    y_pred_stack_1, y_pred_test_1 = predict_opt_clf(xgbopt, xgbparams, x_train_stack, x_test)

    skopt = SklearnOpt.SklearnOpt(x_train_train, y_train_train)
    y_pred_stack_2, y_pred_test_2 = predict_opt_clf(skopt, rfparams, x_train_stack, x_test)

    skopt = SklearnOpt.SklearnOpt(x_train_train, y_train_train)
    y_pred_stack_3, y_pred_test_3 = predict_opt_clf(skopt, extparams, x_train_stack, x_test)

    logging.info(" >>> DGH >>>  prediction  =>  stacking")
    x_pred_stack = pd.DataFrame(np.transpose(np.array([y_pred_stack_1, y_pred_stack_2, y_pred_stack_3])))
    x_pred_test = pd.DataFrame(np.transpose(np.array([y_pred_test_1, y_pred_test_2, y_pred_test_3])))

    lr = LogisticRegression()
    lr.fit(x_pred_stack, y_train_stack)

    return lr.predict_proba(x_pred_test)


def meta_stack_that(x_train, y_train, x_test, train_idx, stack_idx, rfparams, extparams, xgbparams):

    pca = PCA(n_components=10)
    kmeans = KMeans(n_clusters=3)

    x_train_train = x_train.iloc[train_idx]
    y_train_train = y_train.iloc[train_idx]
    x_train_stack = x_train.iloc[stack_idx]
    y_train_stack = y_train.iloc[stack_idx]

    logging.info(" >>> DGH >>> kmean-pca")
    x_train_stack_cls = kmeans.fit_predict((pca.fit_transform(x_train_stack)))
    x_test_stack_cls = kmeans.predict((pca.transform(x_test)))

    x_cls_stack = pd.get_dummies(x_train_stack_cls, prefix='cls').reset_index(drop=True)
    x_cls_test = pd.get_dummies(x_test_stack_cls, prefix='cls').reset_index(drop=True)

    logging.info(" >>> DGH >>> kmean-pca  =>  prediction")
    xgbopt = XGBOpt.XGBOpt(x_train_train, y_train_train)
    y_pred_stack_1, y_pred_test_1 = predict_opt_clf(xgbopt, xgbparams, x_train_stack, x_test)

    skopt = SklearnOpt.SklearnOpt(x_train_train, y_train_train)
    y_pred_stack_2, y_pred_test_2 = predict_opt_clf(skopt, rfparams, x_train_stack, x_test)

    skopt = SklearnOpt.SklearnOpt(x_train_train, y_train_train)
    y_pred_stack_3, y_pred_test_3 = predict_opt_clf(skopt, extparams, x_train_stack, x_test)

    logging.info(" >>> DGH >>> kmean-pca  =>  prediction  =>  stacking")
    x_pred_stack = pd.DataFrame(np.transpose(np.array([y_pred_stack_1, y_pred_stack_2, y_pred_stack_3])))
    x_pred_test = pd.DataFrame(np.transpose(np.array([y_pred_test_1, y_pred_test_2, y_pred_test_3])))

    for col1 in x_cls_stack.columns:
        for col2 in x_pred_stack.columns:
            x_cls_stack['ms_'+str(col1)+'_'+str(col2)] = x_cls_stack[col1] * x_pred_stack[col2].reset_index(drop=True)
            x_cls_test['ms_'+str(col1)+'_'+str(col2)] = x_cls_test[col1] * x_pred_test[col2].reset_index(drop=True)

    lr = LogisticRegression()
    lr.fit(x_cls_stack[[c for c in x_cls_stack.columns if c.startswith('ms')]], y_train_stack)

    return lr.predict_proba(x_cls_test[[c for c in x_cls_test.columns if c.startswith('ms')]])
