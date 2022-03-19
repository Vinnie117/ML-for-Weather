import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# load data
df_raw = pd.read_csv(r'A:\Projects\ML-for-Weather\data\raw\test_simple.csv') 

# Cleaning
df = df_raw.drop(columns=["station_id", "dataset"])
df = df.pivot(index="date", columns="parameter", values="value").reset_index()

######################################################
# Create Lags

class InsertLags(BaseEstimator, TransformerMixin):
    """
    Automatically Insert Lags
    """
    def __init__(self, lags):
        self.lags = lags

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        original_cols=list(range(len(X[0,:])))
        for lag in self.lags:
            X_lagged=pd.DataFrame(X[:,original_cols]).shift(lag)
            # add columns to df
            X=np.concatenate((X,X_lagged), axis=1)
        return X

# Lags definieren
add_lags=InsertLags([1,2,3])

# Transform pandas df to numpy array
df_np = df.to_numpy()
result = add_lags.fit_transform(df_np)
print(result)

df_result = pd.DataFrame(result)

print("END")