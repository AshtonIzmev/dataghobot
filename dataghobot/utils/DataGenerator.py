from sklearn.datasets import load_digits, load_iris
import pandas as pd


def get_adult_data(data_path='../data/adult.data'):
    cols = ['age', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
            'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
            'native-country', 'target']
    df = pd.read_csv(data_path, names=cols)
    df['target'] = df['target'].map(lambda s: 0 if '<=' in s else 1)

    x = df.drop(['target'], axis=1)
    y = df['target']

    return x, y


def get_digits_data():
    data = load_digits(2)
    cols = ['i'+str(i) for i in range(data.data.shape[1])]
    x = pd.DataFrame(data.data, columns=cols)
    y = pd.Series(data.target)
    return x, y


def get_iris_data():
    data = load_iris()
    x = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    return x, y
