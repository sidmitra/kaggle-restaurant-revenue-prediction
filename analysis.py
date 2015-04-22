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
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor


city_le = preprocessing.LabelEncoder()
city_group_le = preprocessing.LabelEncoder()
type_le = preprocessing.LabelEncoder()
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

    # Remove Open Date column
    df.drop('Open Date', axis=1, inplace=True)

    # Remove revenue column
    revenue = None
    if is_training_set:
        revenue = df['revenue']
        df.drop('revenue', axis=1, inplace=True)
        city_le.fit(df['City'])
        city_group_le.fit(df['City Group'])
        type_le.fit(df['Type'])
        df['City'] = city_le.transform(df['City'])
        df['City Group'] = city_group_le.transform(df['City Group'])
        df['Type'] = type_le.transform(df['Type'])
    else:
        city_le.fit(df['City'])
        city_group_le.fit(df['City Group'])
        type_le.fit(df['Type'])
        df['City'] = city_le.transform(df['City'])
        df['City Group'] = city_group_le.transform(df['City Group'])
        df['Type'] = type_le.transform(df['Type'])

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


def apply_random_forest(train, test, revenue):
    model = RandomForestRegressor(n_estimators=200)
    model.fit(train, revenue)
    print model.feature_importances_
    test_revenue = model.predict(test)

    sub = pd.read_csv('data/sampleSubmission.csv')
    sub['Prediction'] = test_revenue
    sub.to_csv('output/random_forest.csv', index=False)


if __name__ == "__main__":
    train, revenue = get_df('data/train.csv')
    test, _ = get_df('data/test.csv', is_training_set=False)
    apply_random_forest(train, test, revenue)
