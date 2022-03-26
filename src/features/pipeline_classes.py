import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

######################################################
# Define transformers to edit raw input data

class Debugger(BaseEstimator, TransformerMixin):
    """
    View pipeline data for debugging
    """
    def fit(self, data, y=None, **fit_params):
        # No need to fit anything, because this is not an actual transformation. 
        return self
    
    def transform(self, data):
        # Here just print what is needed + return the actual data. Nothing is transformed. 
        print("Shape of data", data.shape)
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
        data = X
        X = X.to_numpy()

        # indices of 'np array columns', e.g. array with 3 columns -> [0,1,2]
        col_indices=list(range(len(X[0,:])))
        col_indices = col_indices[4:] # weather variables start after 4th column (timestamp, month, day, hour are before)

        # create lags
        for lag in self.lags:
            X_lagged=pd.DataFrame(X[:,col_indices]).shift(lag)
            X=np.concatenate((X,X_lagged), axis=1)

        # create column names (= normal columns + lagged columns)
        cols = data.columns.tolist()
        lag_col_names = []
        for x in range(len(self.lags)):
            for y in cols[4:]:
                lag_col_names.append(str(y) + '_lag_' + str(self.lags[x]))
        return pd.DataFrame(X, columns = cols + lag_col_names)


class Times(BaseEstimator, TransformerMixin):

    def fit(self, X):
        return self

    def transform(self, data):
        # convert to CET (UTC +1), then remove tz
        data['timestamp'] = pd.to_datetime(data['date']).dt.tz_convert('Europe/Berlin').dt.tz_localize(None)
        data['month'] =  data['timestamp'].dt.month
        data['day'] =  data['timestamp'].dt.day 
        data['hour'] =  data['timestamp'].dt.hour
        data = data.drop('date', 1)

        #reorder columns
        cols = list(data.columns)
        cols = cols[-4:] + cols[:len(cols)-4]
        data = data[cols]
        return data

class Velocity(BaseEstimator, TransformerMixin):
    """
    Calculate differences
    """
    def __init__(self, vars, diff):
        self.diff = diff
        self.vars = vars

    def fit(self, X):
        return self

    def transform(self, X):
        data = X
        
        # create column names
        cols = []
        for i in range(len(self.vars)):
            cols.append(self.vars[i] + '_velo_' + str(self.diff))

        # create data
        for i in range(len(self.vars)):
            data[cols[i]] = data[self.vars[i]].diff(periods = self.diff)
        return data