import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split

# load data
df_raw = pd.read_csv(r'A:\Projects\ML-for-Weather\data\raw\test_simple.csv') 

# Cleaning
df = df_raw.drop(columns=["station_id", "dataset"])
df = df.pivot(index="date", columns="parameter", values="value").reset_index()
df = df.rename(columns={'temperature_air_mean_200': 'temperature', 'cloud_cover_total': 'cloud_cover',
                        'wind_speed': 'wind_speed'})


# Split
train, test = train_test_split(df, test_size=0.2, shuffle = False)

######################################################
# Define transformers to edit raw input data

class Debugger(BaseEstimator, TransformerMixin):
    """
    View pipeline data for debugging
    """
    def fit(self, data, y=None, **fit_params):
        # No need to fit anything, because this is not an actual  transformation. 
        return self
    
    def transform(self, data):
        # Here you just print what you need + return the actual data. You're not transforming anything. 
        print("Shape of Pre-processed Data:", data.shape)
        print(pd.DataFrame(data).head())
        return data

class InsertLags(BaseEstimator, TransformerMixin):
    """
    Automatically insert lags
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
        # create lags
        for lag in self.lags:
            X_lagged=pd.DataFrame(X[:,col_indices]).shift(lag)
            X=np.concatenate((X,X_lagged), axis=1)
        # create column names (= normal columns + lagged columns)
        cols = df.columns.tolist()  # df befindet sich außerhalb dieser Klasse! -> reinpacken? Bzw. als vorherigen Step in der Pipeline?
        start = [cols[cols.index("date")], cols[cols.index("temperature")]]
        unwanted = {"date", "temperature"}
        end = [x for x in cols if x not in unwanted]
        cols = start + end
        lag_col_names = []
        for x in range(len(self.lags)):
            for y in cols[1:]:
                lag_col_names.append(str(y) + '_lag_' + str(self.lags[x]))
        return pd.DataFrame(X, columns = cols + lag_col_names)






