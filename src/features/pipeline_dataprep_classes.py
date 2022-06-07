import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, TimeSeriesSplit
import copy 
pd.options.mode.chained_assignment = None

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

        for k in data:
            #print('Shape of', k, 'data:', data[k].shape)
            print(data[k])

        return data


class Split(BaseEstimator, TransformerMixin):
    """
    Split data into train and test set -> sklearn.model_selection.TimeSeriesSplit
    """
    def __init__(self, test_size, shuffle):
        self.test_size = test_size
        self.shuffle = shuffle

    def fit(self, data):
        return self

    def transform(self, data):
        '''
        Build an (empty) dictionary which will be filled with data in later transformers
        '''

        # The dictionary which will contain the data
        dict_data = {}
        dict_data['train'] = {}
        dict_data['test'] = {}

        # Specifiy splitting for Time series cross validation
        tscv = TimeSeriesSplit(n_splits = 5)

        # get list of indices of original dataframe
        indices = list(data.index.values)

        # create indices and folds for time series data
        for fold, (train_index, test_index) in enumerate(tscv.split(indices)):
            print("Fold: {}".format(fold))
            print("TRAIN indices:", train_index, "\n", "TEST indices:", test_index)
            print("\n")
            train, test = data.iloc[train_index], data.iloc[test_index]
            dict_data['train']["train_fold_{}".format(fold)] = train
            dict_data['test']["test_fold_{}".format(fold)] = test
            
        return dict_data


class Times(BaseEstimator, TransformerMixin):

    def fit(self, X):
        return self

    def transform(self, dict_data):
        
        # convert to CET (UTC +1), then remove tz
        for i in dict_data:
            for k in dict_data[i]:
                dict_data[i][k]['timestamp'] = pd.to_datetime(dict_data[i][k]['date']).dt.tz_convert('Europe/Berlin').dt.tz_localize(None)
                dict_data[i][k]['month'] =  dict_data[i][k]['timestamp'].dt.month
                dict_data[i][k]['day'] =  dict_data[i][k]['timestamp'].dt.day 
                dict_data[i][k]['hour'] =  dict_data[i][k]['timestamp'].dt.hour
                dict_data[i][k] = dict_data[i][k].drop('date', axis = 1)

                #reorder columns
                cols = list(dict_data[i][k].columns)
                cols = cols[-4:] + cols[:len(cols)-4]
                dict_data[i][k] = dict_data[i][k][cols]

        return dict_data


class Velocity(BaseEstimator, TransformerMixin):
    """
    Calculate differences (compute new features in 'X' (= dict_data), add to master 'data')
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
        for i in self.diff:
            for j in self.vars:
                cols.append(j + '_velo_' + str(i))

        # create data (velocities) for each data set k (train/test) in dict X

        for i in X:
            for k in X[i]:
                col_indices = [data[i][k].columns.get_loc(c) for c in self.vars if c in data[i][k]]
                dummy = []
                for j in self.diff:
                    dummy.append(pd.DataFrame(data[i][k].iloc[:,col_indices].diff(periods = j)))
                X[i][k] = pd.concat(dummy, axis=1)
                X[i][k].columns = cols
    
                # combine with master data frame
                data[i][k] = pd.concat([data[i][k], X[i][k]], axis=1)

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
        for i in self.diff:
            for j in self.vars:
                cols.append(j + '_acc_' + str(i))

        # create data (accelerations) for each data set k (train/test) in dict X
        for i in X:
            for k in X[i]:
                col_indices = [data[i][k].columns.get_loc(c) for c in self.vars if c in data[i][k]]
                dummy = []
                for j in self.diff:
                    dummy.append(pd.DataFrame(data[i][k].iloc[:,col_indices].diff(periods = j).diff(periods = 1)))
                X[i][k] = pd.concat(dummy, axis=1)
                X[i][k].columns = cols
    
                # combine with master data frame
                data[i][k] = pd.concat([data[i][k], X[i][k]], axis=1)
        
        return data


class InsertLags(BaseEstimator, TransformerMixin):
    """
    Automatically insert lags (compute new features in 'X', add to master 'data')
    """
    def __init__(self, vars, diff):
        self.diff = diff
        self.vars = vars

    def fit(self, X):
        return self

    def transform(self, X):

        data = copy.deepcopy(X)

        # create column names
        cols = data['train']['train_fold_0'].columns.tolist()
        cols = cols[4:] # 4 to start from columns without time vars
        lags = []
        for i in self.diff:
            for j in cols:
                lags.append(j + '_lag_' + str(i))

        # create data (lags) for each data set k (train/test) in dict X
        for i in X:
            for k in X[i]:
                col_indices = [data[i][k].columns.get_loc(c) for c in cols if c in data[i][k]]
                dummy = []
                for j in self.diff:
                    dummy.append(pd.DataFrame(data[i][k].iloc[:,col_indices].shift(j)))
                X[i][k] = pd.concat(dummy, axis=1)
                X[i][k].columns = lags

                # combine with master data frame
                data[i][k] = pd.concat([data[i][k], X[i][k]], axis=1)

        print(data['train'])
        print(data['test'])
        return data # a dict with training and test data


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
        
        # complete dataframe for further use, e.g. evaluation
        dict_data['pd_df'] = pd.concat([dict_data['train'], dict_data['test']], axis=0)
                   
        # array data for sklearn
        for k,v in dict_data.items():
             if k != 'pd_df':
                if self.vars:
                    dict_data[k] = pd.concat([dict_data[k][self.target], dict_data[k][self.vars]], axis=1)
                    dict_data[k] = dict_data[k].dropna()
                    #dict_data[k] = dict_data[k].to_numpy()
                if not self.vars:
                # if no predictors are provided in config file, use all lagged variables for train and test set
                    all_vars = [ x for x in dict_data['pd_df'] if "lag" in x ]
                    time = ['month', 'day', 'hour']
                    dict_data[k] = pd.concat([dict_data[k][self.target], 
                                              dict_data[k][time],
                                              dict_data[k][all_vars]], axis=1)
                    dict_data[k] = dict_data[k].dropna()
                    #dict_data[k] = dict_data[k].to_numpy()

        return dict_data