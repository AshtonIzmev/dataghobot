import pandas as pd


def add_miss_val_indicator(df, miss_val=None, suffix='_is_nan'):
    """
    Add new columns with a missing value indicator
    :param df:
    :param miss_val: value to interpret as a missing value
    :param suffix: column suffix for nan indicator
    :return:
    """
    if miss_val is None:
        for col in df.columns:
            col_null = pd.isnull(df[col])
            if len(col_null.value_counts()) == 2:
                df[col+suffix] = col_null.map(int)
    else:
        for col in df.columns:
            col_null = df[col] == miss_val
            if len(col_null.value_counts()) == 2:
                df[col+suffix] = col_null.map(int)


def add_miss_val_indicator_from_dic(df, miss_dic=None, suffix='_is_nan'):
    """
    Add new columns with a missing value indicator
    :param df:
    :param miss_dic: column => missing value dictionary
    :param suffix: column suffix for nan indicator
    :return:
    """
    for k, v in miss_dic.iteritems():
        col_null = df[k] == v
        if len(col_null.value_counts()) == 2:
            df[k+suffix] = col_null.map(int)


def fill_with_mean(df):
    """
    Fill missing values of numeric columns with mean
    :param df:
    :return:
    """
    for col in df.columns:
        coltype = df[col].dtype.name
        if coltype.startswith('int') or coltype.startswith('float'):
            col_null = pd.isnull(df[col])
            if len(col_null.value_counts()) == 2:
                df.loc[col_null, col] = df[col][~col_null].mean()
