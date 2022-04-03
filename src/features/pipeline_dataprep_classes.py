import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import copy 

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
        for k, v in data.items():
            print('Shape of', k, 'data:', data[k].shape)
            print(data[k])

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
        train, test = train_test_split(X, test_size=self.test_size, shuffle = self.shuffle)
        dict_data = {}
        dict_data['train'] = train
        dict_data['test'] = test

        return dict_data


class Times(BaseEstimator, TransformerMixin):

    def fit(self, X):
        return self

    def transform(self, dict_data):
        # convert to CET (UTC +1), then remove tz       
        for k, v in dict_data.items():
            dict_data[k]['timestamp'] = pd.to_datetime(dict_data[k]['date']).dt.tz_convert('Europe/Berlin').dt.tz_localize(None)
            dict_data[k]['month'] =  dict_data[k]['timestamp'].dt.month
            dict_data[k]['day'] =  dict_data[k]['timestamp'].dt.day 
            dict_data[k]['hour'] =  dict_data[k]['timestamp'].dt.hour
            dict_data[k] = dict_data[k].drop('date', 1)

            #reorder columns
            cols = list(dict_data[k].columns)
            cols = cols[-4:] + cols[:len(cols)-4]
            dict_data[k] = dict_data[k][cols]

        return dict_data


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

        data = copy.deepcopy(X)

        # create column names
        cols = []
        for i in range(len(self.lags)):
            for j in range(len(self.vars)):
                cols.append(self.vars[j] + '_lag_' + str(self.lags[i]))

        # create data (lags) for each data set k (train/test) in dict X
        for k, v in X.items():
            col_indices = [data[k].columns.get_loc(c) for c in self.vars if c in data[k]]
            dummy = []
            for i in self.lags:
                dummy.append(pd.DataFrame(data[k].iloc[:,col_indices].shift(i)))
            X[k] = pd.concat(dummy, axis=1)
            X[k].columns = cols
 
            # combine with master data frame
            data[k] = pd.concat([data[k], X[k]], axis=1)


        return data # a dict with training and test data


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
        data = copy.deepcopy(X)
 
        # create column names
        cols = []
        for i in range(len(self.diff)):
            for j in range(len(self.vars)):
                cols.append(self.vars[j] + '_velo_' + str(self.diff[i]))

        # create data
        for k, v in X.items():
            col_indices = [data[k].columns.get_loc(c) for c in self.vars if c in data[k]]
            dummy = []
            for i in self.diff:
                dummy.append(pd.DataFrame(data[k].iloc[:,col_indices].diff(periods = i)))
            X[k] = pd.concat(dummy, axis=1)
            X[k].columns = cols
 
            # combine with master data frame
            data[k] = pd.concat([data[k], X[k]], axis=1)

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
        data = copy.deepcopy(X)
 
        # create column names
        cols = []
        for i in range(len(self.diff)):
            for j in range(len(self.vars)):
                cols.append(self.vars[j] + '_acc_' + str(self.diff[i]))

        # create data
        for k, v in X.items():
            col_indices = [data[k].columns.get_loc(c) for c in self.vars if c in data[k]]
            dummy = []
            for i in self.diff:
                dummy.append(pd.DataFrame(data[k].iloc[:,col_indices].diff(periods = i).diff(periods = i)))
            X[k] = pd.concat(dummy, axis=1)
            X[k].columns = cols
 
            # combine with master data frame
            data[k] = pd.concat([data[k], X[k]], axis=1)
        
        return data

class Prepare(BaseEstimator, TransformerMixin):
    '''
    Prepare data for scikit-learn: drop NaN, convert to np.array -> and select vars for prediction
    '''
    def __init__(self, vars, target):
        self.vars = vars
        self.target = target

    def fit(self, X):
        return self

    def transform(self, dict_data):
        for k,v in dict_data.items():
                dict_data[k] = pd.concat([dict_data[k][self.target], dict_data[k][self.vars]], axis=1)
                dict_data[k] = dict_data[k].dropna()
                dict_data[k] = dict_data[k].to_numpy()
        
        return dict_data




