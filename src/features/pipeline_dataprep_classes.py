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

        return data # a dict with training and test data


class Scaler(BaseEstimator, TransformerMixin):
    """
    Standardize predictors
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

        # apply normalization parameters  obtained from the training set as-is on test data.
        # Test data is unseen, recalculating parameters is inconsistent with model
        scaled = scaler.fit_transform(dict_data['train'][last_train_key].iloc[:, 4:]) 
        scaled_df = pd.DataFrame(scaled, columns = list(dict_data['train'][last_train_key].iloc[:, 4:]))

        not_scaled = dict_data['train'][last_train_key].iloc[:, 0:5]
        not_scaled = not_scaled.rename({'temperature': 'target_temperature'}, axis=1) 
        scaled_df.index = not_scaled.index

        df_all = pd.concat([not_scaled, scaled_df], axis = 1,)
        #print('train_fold_4', df_all.iloc[0:15,0:9])
        scaled_data['train_std']['train_fold_4'] = df_all

        for i,j in zip(dict_data, scaled_data):
            for k,l in zip(dict_data[i], scaled_data[j]):
                if k != 'train_fold_4':
                    scaled = scaler.transform(dict_data[i][k].iloc[:, 4:])
                    df = pd.DataFrame(scaled, columns = list(dict_data[i][k].iloc[:, 4:]))
                    # print(df)

                    # target var is standardized but also extracted as normal value (labeled 'target_...')
                    not_scaled = dict_data[i][k].iloc[:, 0:5]
                    not_scaled = not_scaled.rename({'temperature': 'target_temperature'}, axis=1) 
                    # print(not_scaled)
                    df.index = not_scaled.index

                    # seperate room in the data dictionary for std. data
                    df_all = pd.concat([not_scaled, df], axis = 1,)
                    # print(df_all.iloc[:,0:9])

                    scaled_data[j][l] = df_all
        
        # at the end build the complete data dict
        dict_data = dict_data | scaled_data # -> order keys in scaled_data! https://stackoverflow.com/questions/51086412/moving-elements-in-dictionary-python-to-another-index
        # for i in dict_data['train']:
        #     print(dict_data['train'][i].iloc[0:15,0:9])
        for i in dict_data['train_std']:
            print(dict_data['train_std'][i].iloc[0:15,0:9])
        # for i in dict_data['test']:
        #     print(dict_data['test'][i].iloc[0:15,0:9])
        for i in dict_data['test_std']:
            print(dict_data['test_std'][i].iloc[0:15,0:9])
        # # This approach only scales the last fold (i.e the biggest) of each train and test set
        # # -> Cannot loop through all folds of train bc. need fit_transform of last fold first
        # for i in dict_data:
        #     # the last fold contains all training/test data relevant for parameters of std.
        #     # TimeSeriesSplits creates subsets
        #     last_key = list(dict_data[i])[-1]

        #     # apply normalization parameters  obtained from the training set as-is on test data.
        #     # Test data is unseen, recalculating parameters is inconsistent with model
        #     if i == 'train':
        #         scaled = scaler.fit_transform(dict_data[i][last_key].iloc[:, 4:])
        #     if i == 'test':
        #         scaled = scaler.transform(dict_data[i][last_key].iloc[:, 4:]) 
        #     scaled_df = pd.DataFrame(scaled, columns = list(dict_data[i][last_key].iloc[:, 4:]))

        #     # target var is standardized but also extracted as normal value (labeled 'target_...')
        #     not_scaled = dict_data[i][last_key].iloc[:, 0:5]
        #     not_scaled = not_scaled.rename({'temperature': 'target_temperature'}, axis=1) 
        #     scaled_df.index = not_scaled.index

        #     df_all = pd.concat([not_scaled, scaled_df], axis = 1,)
        #     print(i, df_all.iloc[0:15,0:9])


#################################################################################################
        # # Standardize each fold in training and test data
        # for i in dict_data:
        #     for k in dict_data[i]:
        #         scaled = scaler.fit_transform(dict_data[i][k].iloc[:, 4:])
        #         df = pd.DataFrame(scaled, columns = list(dict_data[i][k].iloc[:, 4:]))
        #         # print(df)

        #         # target var is standardized but also extracted as normal value (labeled 'target_...')
        #         not_scaled = dict_data[i][k].iloc[:, 0:5]
        #         not_scaled = not_scaled.rename({'temperature': 'target_temperature'}, axis=1) 
        #         # print(not_scaled)
        #         df.index = not_scaled.index

        #         # seperate room in the data dictionary for std. data
        #         df_all = pd.concat([not_scaled, df], axis = 1,)
        #         # print(df_all.iloc[:,0:9])

        #         dict_data[i][k] = df_all
#################################################################################################

        return dict_data # a dict with training and test data


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

        # complete dataframe for further use, e.g. evaluation
        dict_data['pd_df'] = pd.concat([dict_data['train']["train_fold_{}".format(folds)],
                                        dict_data['test']["test_fold_{}".format(folds)]], 
                                        axis=0)
                   
        # array data for sklearn
        for i in dict_data:
            if i != 'pd_df':
                for k in dict_data[i]:
                    if self.vars:
                        dict_data[i][k] = pd.concat([dict_data[i][k]['target_{}'.format(self.target)], dict_data[i][k][self.vars]], 
                                                     axis=1)
                        dict_data[i][k] = dict_data[i][k].dropna()
                        #dict_data[i][k] = dict_data[i][k].to_numpy()
                    if not self.vars:
                    # if no predictors are provided in config file, use all lagged variables for train and test set
                        all_vars = [ x for x in dict_data['pd_df'] if "lag" in x ]
                        time = ['month', 'day', 'hour']
                        dict_data[i][k] = pd.concat([dict_data[i][k][self.target], 
                                                dict_data[i][k][time],
                                                dict_data[i][k][all_vars]], axis=1)
                        dict_data[i][k] = dict_data[i][k].dropna()
                        dict_data[i][k] = dict_data[i][k].to_numpy()

        return dict_data