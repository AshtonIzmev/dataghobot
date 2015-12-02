import pandas as pd
import logging
from dataghobot.featengine import ChaosGeneration
from dataghobot.featengine import Numerisation
from dataghobot.stacking import CrossValStack
from sklearn.cross_validation import KFold


def robot(x_train, y_train, x_valid,
          rst=42,
          nb_auto_max=-1,
          cv_feat=6,
          cv_hopt=6,
          cv_stack=5):

    res = []
    nb_auto = 0

    for train1_idx, feat_idx in KFold(len(x_train), n_folds=cv_feat, shuffle=True, random_state=rst):
        x_train1 = x_train.iloc[train1_idx]
        y_train1 = y_train.iloc[train1_idx]
        x_feat = x_train.iloc[feat_idx]
        y_feat = y_train.iloc[feat_idx]

        logging.info("Chaos feature generation")
        x_train1, x_valid = chaosize(x_feat, x_train1, x_valid, y_feat)

        logging.info("Feature cleaning")
        x_train_num, x_valid_num = numerize(x_train1, x_valid)

        for train2_idx, hopt_idx in KFold(len(x_train_num), n_folds=cv_hopt, shuffle=True, random_state=rst):
            x_train2 = x_train_num.iloc[train2_idx]
            y_train2 = y_train1.iloc[train2_idx]
            x_hopt = x_train_num.iloc[hopt_idx]
            y_hopt = y_train1.iloc[hopt_idx]

            logging.info("Looking for hopt parameters")
            xgbparam = CrossValStack.get_best_xgbopt(x_hopt, y_hopt)
            sklparam = CrossValStack.get_best_sklopt(x_hopt, y_hopt)
            extparam = CrossValStack.get_best_etopt(x_hopt, y_hopt)
            res.append(CrossValStack.cross_val_stack(x_train2, y_train2, x_valid_num,
                                                     xgbparam, sklparam, extparam,
                                                     cv=cv_stack))

            nb_auto += 1
            if nb_auto == nb_auto_max:
                return res
            print 'nb auto increased: ', nb_auto
    return res


def stacking_res_to_one_pred(res):
    s = 0
    nb = 0
    for i in range(len(res)):
        for j in range(len(res[i])):
            s = s + res[i][j][:, 1]
            nb += 1
    return s / nb


def chaosize(x_feat, x_train1, x_valid, y_feat):
    x_feat.loc[:, 'source'] = 0
    x_train1.loc[:, 'source'] = 1
    x_valid.loc[:, 'source'] = 2
    x_all = pd.concat([x_feat, x_train1, x_valid])
    ChaosGeneration.chaos_feature_importance(x_all, y_feat, x_all['source'] == 0)
    x_train_res = x_all[x_all['source'] == 1].drop(['source'], axis=1)
    x_valid_res = x_all[x_all['source'] == 2].drop(['source'], axis=1)
    return x_train_res, x_valid_res


def numerize(x_train1, x_valid):
    x_train1.loc[:, 'source'] = 0
    x_valid.loc[:, 'source'] = 1
    x_all = pd.concat([x_train1, x_valid])
    num_cols, todummy_cols, tootherisation_cols, tobedefined_cols = Numerisation.handle_columns(x_all, [], [])
    x_all_num = Numerisation.treat_dataframe(x_all, num_cols, [], todummy_cols, tootherisation_cols)
    x_train_num = x_all_num[x_all_num['source'] == 0].drop(['source'], axis=1)
    x_valid_num = x_all_num[x_all_num['source'] == 1].drop(['source'], axis=1)
    return x_train_num, x_valid_num
