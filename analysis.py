"""
TODO:

- ISO map dimensionality reduction
"""
import time

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_extraction import DictVectorizer


vec = DictVectorizer()


def extract_date(value):
    value = time.strptime(value, "%m/%d/%Y")
    return value


def get_df(csv_file, is_training_set=True):
    df = pd.read_csv(csv_file, encoding="utf-8", index_col=0)
    # Split to year, month day
    df['Open Date'] = df['Open Date'].apply(extract_date)
    df['Day'] = df['Open Date'].map(lambda x: x.tm_mday)
    df['Month'] = df['Open Date'].map(lambda x: x.tm_mon)
    df['Year'] = df['Open Date'].map(lambda x: x.tm_year)

    # Convert to factors
    # df['Day'] = df['Day'].astype(object)
    # df['Month'] = df['Month'].astype(object)
    # df['Year'] = df['Year'].astype(object)

    # Remove Open Date column
    df.drop('Open Date', axis=1, inplace=True)

    # TODO: convert to labels
    # df.drop('City', axis=1, inplace=True)
    # df.drop('City Group', axis=1, inplace=True)
    # df.drop('Type', axis=1, inplace=True)

    # Remove revenue column
    revenue = None
    if is_training_set:
        revenue = df['revenue']
        df.drop('revenue', axis=1, inplace=True)
        df = vec.fit_transform(
            df[['City', 'City Group', 'Type']].T.to_dict().values()
        ).todense()
    else:
        df = vec.transform(
            df[['City', 'City Group', 'Type']].T.to_dict().values()
        ).todense()
    df = pd.DataFrame(df)

    return df, revenue


def apply_pca():
    pca = PCA(n_components=3)
    print("Data shape", train.shape)
    pca.fit(train)
    train_reduced = pca.transform(train)
    print("Reduced Data shape", train_reduced.shape)
    plt.scatter(train_reduced[:, 0], train_reduced[:, 1],
                cmap='RdYlBu')
    plt.show()


train, revenue = get_df('data/train.csv')
# test, _ = get_df('data/test.csv')
