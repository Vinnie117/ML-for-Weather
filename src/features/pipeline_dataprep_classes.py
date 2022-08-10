from matplotlib import scale
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
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
    Split data into train and test sets -> sklearn.model_selection.TimeSeriesSplit
    Splitting as first step in the pipeline to prevent any leakage
    """
    def __init__(self, n_splits):
        self.n_splits = n_splits


    def fit(self, data):
        return self

    def transform(self, data):
        '''
        Build an (empty) dictionary which will be filled with data in later transformers
        '''

        # The dictionary of dicts which will contain the train/test data
        dict_data = {}
        dict_data['train'] = {}
        dict_data['test'] = {}

        # Specifiy splitting for Time series cross validation
        tscv = TimeSeriesSplit(n_splits = self.n_splits)

        # get list of indices of original dataframe
        indices = list(data.index.values)

        # create indices and train/test folds for time series data
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
    def __init__(self, diff):
        self.diff = diff

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

        return data # a dict with training and test data


class Scaler(BaseEstimator, TransformerMixin):
    """
    Standardize predictors
    - NaNs are ignored (NaNs due to lags)
        - 4th point: https://scikit-learn.org/stable/whats_new/v0.20.html#id37
    https://datascience.stackexchange.com/questions/54908/data-normalization-before-or-after-train-test-split 
    """
    def __init__(self, std_target):
        self.std_target = std_target

    def fit(self, dict_data):
        return self

    def transform(self, dict_data):

        # define standard scaler and make seperate room in the data dictionary for std. data
        scaler = StandardScaler()
        scaled_data = {}
        scaled_data['train_std'] = dict.fromkeys(list(dict_data['train'].keys()))
        scaled_data['test_std'] = dict.fromkeys(list(dict_data['test'].keys()))

        # the last fold contains all training/test data relevant for parameters of std.
        # TimeSeriesSplits creates subsets
        last_train_key = list(dict_data['train'])[-1]

        # apply standardization parameters obtained from the training set as-is on test data.
        # Test data is unseen, recalculating parameters is inconsistent with model
        scaled = scaler.fit_transform(dict_data['train'][last_train_key].iloc[:, 1:])

        for i,j in zip(dict_data, scaled_data):
            for k,l in zip(dict_data[i], scaled_data[j]):
                scaled = scaler.transform(dict_data[i][k].iloc[:, 1:])
                scaled_cols = ['std_' + x for x in list(dict_data['train'][last_train_key].iloc[:, 1:])]
                scaled_df = pd.DataFrame(scaled, columns = scaled_cols)

                # target var is standardized but also extracted as normal value (labeled 'target_...')
                not_scaled = dict_data[i][k].iloc[:, [0,4]]
                scaled_df.index = not_scaled.index
                df_all = pd.concat([not_scaled, scaled_df], axis = 1,)

                # seperate room in the data dictionary for std. data
                scaled_data[j][l] = df_all
    
        # at the end build the complete data dict
        dict_data = dict_data | scaled_data
        
        return dict_data 


class Prepare(BaseEstimator, TransformerMixin):
    '''
    Prepare data for scikit-learn: drop NaN, convert to np.array -> and select vars for prediction
    '''
    def __init__(self, vars, target):
        self.vars = vars
        self.target = target

    def fit(self, dict_data):
        return self

    def transform(self, dict_data):
        
        # count number of folds in train/test key (-1 bc indexing starts at 0)
        folds = len(dict_data['train']) - 1

        # array data for sklearn
        time = ['month', 'day', 'hour']
        std_time = ['std_month', 'std_day', 'std_hour']


        for i in dict_data:
            for k in dict_data[i]:
                if self.vars:
                    if i == 'train' or i == 'test':
                        dict_data[i][k] = pd.concat([dict_data[i][k][self.target], dict_data[i][k][self.vars]], axis=1)
                        dict_data[i][k] = dict_data[i][k].dropna()
                    if  i =='train_std' or i =='test_std':
                        cols = ['std_' + x for x in self.vars if x not in time]
                        cols = time + cols
                        dict_data[i][k] = pd.concat([dict_data[i][k][self.target], dict_data[i][k][cols]], axis=1)
                        dict_data[i][k] = dict_data[i][k].dropna()
                if not self.vars:
                # if no predictors are provided in config file, use all lagged variables for train and test set
                    if i == 'train' or i == 'test':
                        all_vars = [x for x in dict_data[i][k] if "lag" in x]
                        dict_data[i][k] = pd.concat([dict_data[i][k][self.target], 
                                                    dict_data[i][k][time], 
                                                    dict_data[i][k][all_vars]], axis=1)
                        dict_data[i][k] = dict_data[i][k].dropna()
                    if i == 'train_std' or i == 'test_std':
                        all_vars = [x for x in dict_data[i][k] if "lag" in x]
                        dict_data[i][k] = pd.concat([dict_data[i][k][self.target], 
                                                    dict_data[i][k][std_time], 
                                                    dict_data[i][k][all_vars]], axis=1)
                        dict_data[i][k] = dict_data[i][k].dropna()

        # complete dataframe for further use, e.g. evaluation
        dict_data['pd_df'] = pd.concat([dict_data['train']["train_fold_{}".format(folds)],
                                        dict_data['test']["test_fold_{}".format(folds)]], 
                                        axis=0)

        return dict_data