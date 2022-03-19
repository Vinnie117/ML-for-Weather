import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split

# load data
df_raw = pd.read_csv(r'A:\Projects\ML-for-Weather\data\raw\test_simple.csv') 

# Cleaning
df = df_raw.drop(columns=["station_id", "dataset"])
df = df.pivot(index="date", columns="parameter", values="value").reset_index()

# Split
train, test = train_test_split(df, test_size=0.2, shuffle = False)

######################################################
# Define a transformer that creates lags

class InsertLags(BaseEstimator, TransformerMixin):
    """
    Automatically Insert Lags
    """
    def __init__(self, lags):
        self.lags = lags

    def fit(self, X):
        return self

    def transform(self, X):
        X = X.to_numpy()
        # indices of 'np array columns', e.g. array with 3 columns -> [0,1,2]
        col_indices=list(range(len(X[0,:])))
        col_indices = col_indices[1:]
        for lag in self.lags:
            X_lagged=pd.DataFrame(X[:,col_indices]).shift(lag)
            X=np.concatenate((X,X_lagged), axis=1)
        return pd.DataFrame(X)