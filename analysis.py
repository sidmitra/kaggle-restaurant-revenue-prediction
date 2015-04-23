"""
TODO:

- ISO map dimensionality reduction
- Elastic net
"""
import time

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error


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


def write_data(prediction, filename='output/out.csv'):
    sub = pd.read_csv('data/sampleSubmission.csv')
    sub['Prediction'] = prediction
    sub.to_csv(filename, index=False)


def apply_pca():
    pca = PCA(n_components=3)
    print("Data shape", train.shape)
    pca.fit(train)
    train_reduced = pca.transform(train)
    print("Reduced Data shape", train_reduced.shape)
    plt.scatter(train_reduced[:, 0], train_reduced[:, 1],
                cmap='RdYlBu')
    plt.show()


def apply_random_forest(train, test, train_predictions, n_estimators=200):
    model = RandomForestRegressor(n_estimators=n_estimators)
    model.fit(train, train_predictions)
    # print model.feature_importances_
    test_predictions = model.predict(test)
    return test_predictions


if __name__ == "__main__":
    train, revenue = get_df('data/train.csv')
    # test, _ = get_df('data/test.csv', is_training_set=False)

    #plt.(range(0, 1000), revenue)
    plt.hist(sorted(revenue))
    plt.show()

    x_train, x_test, y_train, y_test = train_test_split(
        train, revenue, test_size=0.8)

    # min_rmse = 99999999999999999999999999
    # min_n = None
    # for n_estimators in range(1, 401):

    #     y_test_predicted = apply_random_forest(
    #         x_train, x_test, y_train, n_estimators=n_estimators)
    #     rmse = np.sqrt(mean_squared_error(y_test_predicted, y_test))
    #     print 'n_estimators: {0}, RMSE: {1}'.format(n_estimators, rmse)
    #     if rmse < min_rmse:
    #         min_rmse = rmse
    #         min_n = n_estimators

    # print min_rmse, min_n

    y_test_predicted = apply_random_forest(
        x_train, x_test, y_train, n_estimators=8)
    #plt.scatter(y_test_predicted, y_test)
    #plt.show()
    # for i in range(0, len(y_test_predicted)):
    #     print y_test[], y_test_predicted[i]
