import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

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


class Split(BaseEstimator, TransformerMixin):
    """
    Split data into train and test set -> sklearn.model_selection.TimeSeriesSplit
    """
    def __init__(self, test_size, shuffle):
        self.test_size = test_size
        self.shuffle = shuffle

    def fit(self, X):
        return self

    def transform(self, X):
        list_data = []
        train, test = train_test_split(X, test_size=self.test_size, shuffle = self.shuffle)
        list_data.append(train)
        list_data.append(test)

        return list_data


class Times(BaseEstimator, TransformerMixin):

    def fit(self, X):
        return self

    def transform(self, data):
        # convert to CET (UTC +1), then remove tz       
        for i in range(len(data)):
            data[i]['timestamp'] = pd.to_datetime(data[i]['date']).dt.tz_convert('Europe/Berlin').dt.tz_localize(None)
            data[i]['month'] =  data[i]['timestamp'].dt.month
            data[i]['day'] =  data[i]['timestamp'].dt.day 
            data[i]['hour'] =  data[i]['timestamp'].dt.hour
            data[i] = data[i].drop('date', 1)

            #reorder columns
            cols = list(data[i].columns)
            cols = cols[-4:] + cols[:len(cols)-4]
            data[i] = data[i][cols]

        return data


class InsertLags(BaseEstimator, TransformerMixin):
    """
    Automatically insert lags (compute new features in 'X', add to master 'data')
    """
    def __init__(self, vars, lags):
        self.lags = lags
        self.vars = vars

    def fit(self, X):
        return self

    def transform(self, X):

        print(X[0])
        print(X[1])
        #data = X

        # create column names
        cols = []
        for i in range(len(self.lags)):
            for j in range(len(self.vars)):
                cols.append(self.vars[j] + '_lag_' + str(self.lags[i]))

        # create data (lags)
        col_indices = [data.columns.get_loc(c) for c in self.vars if c in data]
        dummy = []
        for i in self.lags:
            dummy.append(pd.DataFrame(data.iloc[:,col_indices].shift(i)))
        X = pd.concat(dummy, axis=1)
        X.columns = cols

        # combine with master data frame
        data = pd.concat([data, X], axis=1)

        return data #pd.DataFrame(X, columns = cols + lag_col_names)


class Velocity(BaseEstimator, TransformerMixin):
    """
    Calculate differences (compute new features in 'X', add to master 'data')
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
        for i in range(len(self.diff)):
            for j in range(len(self.vars)):
                cols.append(self.vars[j] + '_velo_' + str(self.diff[i]))

        # create data
        col_indices = [data.columns.get_loc(c) for c in self.vars if c in data]
        dummy = []
        for i in self.diff:
            dummy.append(pd.DataFrame(data.iloc[:,col_indices].diff(periods = i)))
        X = pd.concat(dummy, axis=1)
        X.columns = cols

        # combine with master data frame
        data = pd.concat([data, X], axis=1)

        return data


class Acceleration(BaseEstimator, TransformerMixin):
    """
    Calculate difference of differences (using base values)
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
        for i in range(len(self.diff)):
            for j in range(len(self.vars)):
                cols.append(self.vars[j] + '_acc_' + str(self.diff[i]))

        # create data
        col_indices = [data.columns.get_loc(c) for c in self.vars if c in data]
        dummy = []
        for i in self.diff:
            dummy.append(pd.DataFrame(data.iloc[:,col_indices].diff(periods = i).diff(periods = i)))
        X = pd.concat(dummy, axis=1)
        X.columns = cols
        print(X)

        # combine with master data frame
        data = pd.concat([data, X], axis=1)

        return data







