import pandas as pd
import numpy as np
from sklearn import ensemble
import random
import math


def product_gen(x, col1, col2):
    x.loc[:, 'x_' + col1 + '_' + col2] = x[col1] * x[col2]


def sum_gen(x, col1, col2):
    x.loc[:, 'p_' + col1 + '_' + col2] = x[col1] + x[col2]


def minus_gen(x, col1, col2):
    x.loc[:, 'm_' + col1 + '_' + col2] = x[col1] - x[col2]


def norm_cat_gen(x, numcol, catcol):
    col = 'nc_' + numcol + '_' + catcol
    x.loc[:, col] = 0
    for vCat in x[catcol].unique():
        row_sel = x[catcol] == vCat
        x_sel = x[row_sel]
        x_sel_norm = (x_sel[numcol] - x_sel[numcol].mean()) / (x_sel[numcol].max() - x_sel[numcol].min())
        x.loc[row_sel, col] = x_sel_norm


def log_gen(x, numcol):
    x.loc[:, 'log_' + numcol] = x[numcol].map(lambda v: math.log(1 + abs(v)))


def num_dum_gen(x, numcol, dummy_max):
    if len(x[numcol].unique()) <= dummy_max:
        dummy_x = pd.get_dummies(x[numcol], prefix='dn_')
        for col in dummy_x.columns:
            x.loc[:, col] = dummy_x[col]


def get_clf_feat(clf, x, nb_features=30):
    feature_importances = clf.feature_importances_
    sorted_idx = np.argsort(feature_importances)[-nb_features:]
    return zip(np.array(x.columns)[sorted_idx], feature_importances[sorted_idx])


def chaos_gen_do(x, numcol1, numcol2, catcol, choice, dummy_max):
    if choice == 0:
        product_gen(x, numcol1, numcol2)
    if choice == 1:
        sum_gen(x, numcol1, numcol2)
    if choice == 2:
        minus_gen(x, numcol1, numcol2)
    if choice == 3:
        norm_cat_gen(x, numcol1, catcol)
    if choice == 4:
        log_gen(x, numcol1)
    if choice == 5:
        num_dum_gen(x, numcol1, dummy_max=dummy_max)


def chaos_gen(x, numcols, catcols, chaos_gen_iter, dummy_max):
    for j in range(chaos_gen_iter):
        numcol1 = random.choice(numcols)
        numcol2 = random.choice(numcols)
        catcol = random.choice(catcols)
        choice = random.randint(0, 5)
        chaos_gen_do(x, numcol1, numcol2, catcol, choice, dummy_max)


def chaos_feature_importance(x_feat, y_feat, shadow_selector, feat_dic={}, chaos_feat_iter=10,
                             n_estimators=10, nb_features=30, chaos_gen_iter=20, dummy_max=20):
    ori_numcols = x_feat.columns
    sel_numcols = []
    for j in range(chaos_feat_iter):
        x_feat = x_feat[list(set(ori_numcols) | set(sel_numcols))]
        #print 'CHAOS feature importance '+str(j)
        clf = ensemble.ExtraTreesClassifier(n_estimators=n_estimators, n_jobs=-1)
        numcols = [c for c in x_feat.columns if x_feat[c].dtype.name != 'object']
        catcols = [c for c in x_feat.columns if x_feat[c].dtype.name == 'object']
        chaos_gen(x_feat, numcols, catcols, chaos_gen_iter=chaos_gen_iter, dummy_max=dummy_max)
        numcol2 = [c for c in x_feat.columns if x_feat[c].dtype.name != 'object']
        x_feat_sel = x_feat[shadow_selector][numcol2]
        y_feat_sel = y_feat
        clf.fit(x_feat_sel.replace(np.inf, 0).replace(-np.inf, 0).fillna(-1), y_feat_sel)
        for f, v in get_clf_feat(clf, x_feat, nb_features=nb_features):
            sel_numcols.append(f)
            if f in feat_dic:
                feat_dic[f] += v
            else:
                feat_dic[f] = v
