import pandas as pd
import numpy as np
from sklearn import ensemble
import random
import math


def product_gen(df, col1, col2):
    df.loc[:, col1+'_x_'+col2] = df[col1] * df[col2]


def sum_gen(df, col1, col2):
    df.loc[:, col1+'_p_'+col2] = df[col1] + df[col2]


def minus_gen(df, col1, col2):
    df.loc[:, col1+'_m_'+col2] = df[col1] - df[col2]


def norm_cat_gen(df, numcol, catcol):
    df.loc[:, numcol+'_nc_'+catcol] = 0
    for vCat in df[catcol].unique():
        row_sel = df[catcol] == vCat
        df_sel = df[row_sel]
        df_sel_norm = (df_sel[numcol] - df_sel[numcol].mean()) / (df_sel[numcol].max() - df_sel[numcol].min())
        df.loc[row_sel, numcol + '_nc_' + catcol] = df_sel_norm


def log_gen(df, numcol):
    df.loc[:, numcol+'_log'] = df[numcol].map(lambda v: math.log(1+abs(v)))


def num_dum_gen(df, numcol, dummy_max):
    if len(df[numcol].unique()) <= dummy_max:
        dummy_df = pd.get_dummies(df[numcol], prefix='dn_')
        for col in dummy_df.columns:
            df.loc[:, col] = dummy_df[col]


def get_clf_feat(clf, x, nb_features=30):
    feature_importances = clf.feature_importances_
    sorted_idx = np.argsort(feature_importances)[-nb_features:]
    return zip(np.array(x.columns)[sorted_idx], feature_importances[sorted_idx])


def chaos_gen(df, numcols, catcols, chaos_gen_iter, dummy_max):
    for j in range(chaos_gen_iter):
        numcol1 = random.choice(numcols)
        numcol2 = random.choice(numcols)
        catcol = random.choice(catcols)
        choice = random.randint(0, 5)
        if choice == 0:
            product_gen(df, numcol1, numcol2)
        if choice == 1:
            sum_gen(df, numcol1, numcol2)
        if choice == 2:
            minus_gen(df, numcol1, numcol2)
        if choice == 3:
            norm_cat_gen(df, numcol1, catcol)
        if choice == 4:
            log_gen(df, numcol1)
        if choice == 5:
            num_dum_gen(df, numcol1, dummy_max=dummy_max)


def chaos_feature_importance(dic, x, y, chaos_feat_iter=10, n_estimators=10, nb_features=30,
                             chaos_gen_iter=10, dummy_max=20):
    ori_numcols = x.columns
    sel_numcols = []
    for j in range(chaos_feat_iter):
        x = x[list(set(ori_numcols) | set(sel_numcols))]
        print 'CHAOS feature importance '+str(j)
        clf = ensemble.ExtraTreesClassifier(n_estimators=n_estimators, n_jobs=-1)
        numcols = [c for c in x.columns if x[c].dtype.name != 'object']
        catcols = [c for c in x.columns if x[c].dtype.name == 'object']
        chaos_gen(x, numcols, catcols, chaos_gen_iter=chaos_gen_iter, dummy_max=dummy_max)
        numcol2 = [c for c in x.columns if x[c].dtype.name != 'object']
        clf.fit(x[numcol2].replace(np.inf, 0).replace(-np.inf, 0).fillna(-1), y)
        for f, v in get_clf_feat(clf, x, nb_features=nb_features):
            sel_numcols.append(f)
            if f in dic:
                dic[f] += v
            else:
                dic[f] = v
