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

class Times(BaseEstimator, TransformerMixin):

    def fit(self, X):
        return self

    def transform(self, data):
        
        # convert to CET (UTC +1), then remove tz
        data['timestamp'] = pd.to_datetime(data['date']).dt.tz_convert('Europe/Berlin').dt.tz_localize(None)
        data['year'] =  data['timestamp'].dt.year
        data['month'] =  data['timestamp'].dt.month
        data['day'] =  data['timestamp'].dt.day 
        data['hour'] =  data['timestamp'].dt.hour
        data = data.drop('date', axis = 1)

        #reorder columns
        cols = list(data.columns)
        cols = cols[-5:] + cols[:len(cols)-5]
        data = data[cols]
                
        return data


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

        # create data (velocities) 
        col_indices = [data.columns.get_loc(c) for c in self.vars if c in data]
        dummy = []
        for j in self.diff:
            dummy.append(pd.DataFrame(data.iloc[:,col_indices].diff(periods = j)))
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
        data = copy.deepcopy(X)
 
        # create column names
        cols = []
        for i in self.diff:
            for j in self.vars:
                cols.append(j + '_acc_' + str(i))

        # create data (accelerations) 
        col_indices = [data.columns.get_loc(c) for c in self.vars if c in data]
        dummy = []
        for j in self.diff:
            dummy.append(pd.DataFrame(data.iloc[:,col_indices].diff(periods = j).diff(periods = 1)))
        X = pd.concat(dummy, axis=1)
        X.columns = cols

        # combine with master data frame
        data = pd.concat([data, X], axis=1)
        
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
        cols = data.columns.tolist()
        cols = cols[5:] # 5 to start from columns without time vars
        lags = []
        for i in self.diff:
            for j in cols:
                lags.append(j + '_lag_' + str(i))

        # create data (lags) for each data set k (train/test) in dict X
        col_indices = [data.columns.get_loc(c) for c in cols if c in data]
        dummy = []
        for j in self.diff:
            dummy.append(pd.DataFrame(data.iloc[:,col_indices].shift(j)))
        X = pd.concat(dummy, axis=1)
        X.columns = lags

        # combine with master data frame
        data = pd.concat([data, X], axis=1)

        return data # a dict with training and test data


class Scaler(BaseEstimator, TransformerMixin):
    """
    Standardize predictors
    - NaNs are ignored (NaNs due to lags)
        - 4th point: https://scikit-learn.org/stable/whats_new/v0.20.html#id37
    https://datascience.stackexchange.com/questions/54908/data-normalization-before-or-after-train-test-split 
    """
    def __init__(self, target, std_target):
        self.std_target = std_target
        self.target = target

    def fit(self, dict_data):
        return self

    def transform(self, data):

        # define standard scaler and make seperate room in the data dictionary for std. data
        scaler = StandardScaler()

        # get column names of df
        cols = list(data)

        # apply standardization parameters except for 'timestamp'
        scaled = scaler.fit_transform(data[cols].iloc[:, 1:])

        scaled_cols = ['std_' + x for x in list(data[cols].iloc[:, 1:])]
        scaled_df = pd.DataFrame(scaled, columns = scaled_cols)

        # target var is standardized but also extracted as normal value (labeled 'target_...')
        target_idx = data.columns.get_loc(self.target)
        not_scaled = data.iloc[:, [0,target_idx]]
        scaled_df.index = not_scaled.index
        df_all = pd.concat([not_scaled, scaled_df], axis = 1,)
        
        return df_all 


class Prepare(BaseEstimator, TransformerMixin):
    '''
    Prepare data for scikit-learn: drop NaN, convert to np.array -> and select vars for prediction
    '''
    def __init__(self, predictors, target, vars):
        self.predictors = predictors
        self.target = target
        self.vars = vars

    def fit(self, dict_data):
        return self

    def transform(self, dict_data):
        
        # count number of folds in train/test key (-1 bc indexing starts at 0)
        folds = len(dict_data['train']) - 1

        # array data for sklearn
        time = ['year', 'month', 'day', 'hour']
        std_time = ['std_year', 'std_month', 'std_day', 'std_hour']

        for i in dict_data:
            for k in dict_data[i]:
                if self.predictors:
                    if i == 'train' or i == 'test':
                        dict_data[i][k] = pd.concat([dict_data[i][k][self.target], dict_data[i][k][self.predictors]], axis=1)
                        dict_data[i][k] = dict_data[i][k].dropna()
                    if  i =='train_std' or i =='test_std':
                        cols = ['std_' + x for x in self.predictors if x not in time]
                        cols = std_time + cols
                        dict_data[i][k] = pd.concat([dict_data[i][k][self.target], dict_data[i][k][cols]], axis=1)
                        dict_data[i][k] = dict_data[i][k].dropna()
                if not self.predictors:
                # if no predictors are provided in config file, use all lagged variables for train and test set
                    if i == 'train' or i == 'test':
                        all_predictors = [x for x in dict_data[i][k] if "lag" in x]
                        dict_data[i][k] = pd.concat([dict_data[i][k][self.target], 
                                                    dict_data[i][k][time], 
                                                    dict_data[i][k][all_predictors]], axis=1)
                        dict_data[i][k] = dict_data[i][k].dropna()
                    if i == 'train_std' or i == 'test_std':
                        all_predictors = [x for x in dict_data[i][k] if "lag" in x]
                        dict_data[i][k] = pd.concat([dict_data[i][k][self.target], 
                                                    dict_data[i][k][std_time], 
                                                    dict_data[i][k][all_predictors]], axis=1)
                        dict_data[i][k] = dict_data[i][k].dropna()

        # complete dataframe for further use, e.g. evaluation
        dict_data['pd_df'] = pd.concat([dict_data['train']["train_fold_{}".format(folds)],
                                        dict_data['test']["test_fold_{}".format(folds)]], 
                                        axis=0)

        # get the underlying time series of a variable back. Needed to append data for inference     
        vars = [x for x in self.vars if x not in self.target]
        for i in vars:
            shift_back = i + '_lag_1'
            value = dict_data['pd_df'][shift_back].shift(-1, axis = 0)
            dict_data['pd_df'].insert(loc=5, column=i, value=value) # loc=1 inserts new column at index 5

        dict_data['pd_df'] = dict_data['pd_df'].dropna() # using lags to retain underlying time series results in NaN of last row

        timestamp= pd.to_datetime(dict_data['pd_df'][['year', 'month', 'day', 'hour']])
        dict_data['pd_df'].insert(loc=1, column='timestamp', value=timestamp)


        return dict_data