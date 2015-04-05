"""
TODO
- fix extract date
"""
import time
import pandas as pd
from sklearn.decomposition import PCA

def extract_date(value):
    value = time.strptime(value, "%m/%d/%Y")
    return value

train = pd.read_csv("train.csv", encoding="utf-8")
train['Open Date'] = train['Open Date'].apply(extract_date)

# Split to year, month day



#pca = PCA(n_components=5)
#pca.fit(train)
