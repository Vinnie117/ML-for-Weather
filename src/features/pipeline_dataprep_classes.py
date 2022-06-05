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
        '''
        Build an (empty) dictionary which will be filled with data in later transformers
        '''
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
        for i in self.diff:
            for j in self.vars:
                cols.append(j + '_acc_' + str(i))

        # create data (accelerations) for each data set k (train/test) in dict X
        for k, v in X.items():
            col_indices = [data[k].columns.get_loc(c) for c in self.vars if c in data[k]]
            dummy = []
            for i in self.diff:
                dummy.append(pd.DataFrame(data[k].iloc[:,col_indices].diff(periods = i).diff(periods = 1)))
            X[k] = pd.concat(dummy, axis=1)
            X[k].columns = cols
 
            # combine with master data frame
            data[k] = pd.concat([data[k], X[k]], axis=1)
        
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
        cols = data['train'].columns.tolist()
        cols = cols[4:] # 4 to start from columns without time vars
        lags = []
        for i in self.diff:
            for j in cols:
                lags.append(j + '_lag_' + str(i))

        # create data (lags) for each data set k (train/test) in dict X
        for k, v in X.items():
            col_indices = [data[k].columns.get_loc(c) for c in cols if c in data[k]]
            dummy = []
            for i in self.diff:
                dummy.append(pd.DataFrame(data[k].iloc[:,col_indices].shift(i)))
            X[k] = pd.concat(dummy, axis=1)
            X[k].columns = lags

            # combine with master data frame
            data[k] = pd.concat([data[k], X[k]], axis=1)


        # Tests are successful for train and test here!

        # revert_transform = data['test']['temperature_lag_1'].shift(-1)[:-1]
        # original = data['test']['temperature'][:-1]

        # print(revert_transform.equals(original))
        # print(revert_transform.compare(original))

        # print(original.iloc[7005:7010])
        # print(revert_transform.iloc[7005:7010]) 

        # print(original.iloc[7006])
        # print(revert_transform.iloc[7006]) 

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
        dict_data['pd_df'] = pd.concat([dict_data['train'], dict_data['test']], axis=0)  #.dropna()

        #print(dict_data['train'].isnull())
        #print(dict_data['train'].isnull().any(axis=1))

                             
        # array data for sklearn
        for k,v in dict_data.items():
             if k != 'pd_df':
                if self.vars:
                    dict_data[k] = pd.concat([dict_data[k][self.target], dict_data[k][self.vars]], axis=1)
                    dict_data[k] = dict_data[k].dropna()
                    #dict_data[k] = dict_data[k].to_numpy()
                # if not self.vars:
                # # if no predictors are provided in config file, use all lagged variables for train and test set
                #     all_vars = [ x for x in dict_data['pd_df'] if "lag" in x ]
                #     time = ['month', 'day', 'hour']
                #     dict_data[k] = pd.concat([dict_data[k][self.target], 
                #                               dict_data[k][time],
                #                               dict_data[k][all_vars]], axis=1)
                #     dict_data[k] = dict_data[k].dropna()
                #     #dict_data[k] = dict_data[k].to_numpy()

        #dict_data['pd_df'] = pd.concat([dict_data['train'], dict_data['test']], axis=0) #.dropna()


        # Hier schlägt der Test für test fehl, aber für train klappt es
        revert_transform = dict_data['test']['temperature_lag_1'].shift(-1)[:-1]
        original = dict_data['test']['temperature'][:-1]

        print(revert_transform.equals(original))
        print(revert_transform.compare(original))

        print(original.iloc[7005:7010])
        print(revert_transform.iloc[7005:7010]) 

        print(original.iloc[7006])
        print(revert_transform.iloc[7006]) 

        return dict_data