"""
TODO:

- dimensionality reduction
    - ISo map
    - TSNE
    - PCA
- Elastic net
- Classification and then per class model? Ensemble?
- neural net
- gaussian mixture models
- classification

"""
from datetime import datetime
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
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV


city_le = preprocessing.LabelEncoder()
city_group_le = preprocessing.LabelEncoder()
type_le = preprocessing.LabelEncoder()
vec = DictVectorizer()
pca = PCA(0.95)


def extract_date(value):
    return time.strptime(value, "%m/%d/%Y")


def years_since_1900(value):
    return value.tm_year - 1900


def get_df(csv_file, is_training_set=True, reduce_dim=False):
    df = pd.read_csv(csv_file, encoding="utf-8", index_col=0)
    # Split to year, month day
    df['Open Date'] = df['Open Date'].apply(extract_date)
    df['YearsSince1900'] = df['Open Date'].apply(years_since_1900)
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
        if reduce_dim:
            df = pca.fit_transform(df)
    else:
        city_le.fit(df['City'])
        city_group_le.fit(df['City Group'])
        type_le.fit(df['Type'])
        df['City'] = city_le.transform(df['City'])
        df['City Group'] = city_group_le.transform(df['City Group'])
        df['Type'] = type_le.transform(df['Type'])
        if reduce_dim:
            df = pca.transform(df)

    return df, revenue


def write_data(prediction, filename='output/out.csv'):
    sub = pd.read_csv('data/sampleSubmission.csv')
    sub['Prediction'] = prediction
    sub.to_csv(filename, index=False)


def calc_rmse(y_test, y_pred):
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print("RMSE: ", rmse)
    return rmse


train, revenue = get_df('data/train.csv')
test, _ = get_df('data/test.csv', is_training_set=False)

# X_train, X_test, y_train, y_test = train_test_split(train, revenue)

# model = RandomForestRegressor(n_jobs=4, verbose=3)
# parameters = {'n_estimators': range(1, 401),
#               'max_depth': range(1, 101)}

# clf = GridSearchCV(model, parameters)
# clf.fit(X_train, y_train)
# print clf.best_params_
# print clf.score(X_train,y_train)

# #y_pred = clf.predict(X_test)
# #calc_rmse(y_test, y_pred)
