from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class IncrementTime(BaseEstimator, TransformerMixin):
    """
    Increment latest available timestamp by +1 (in last row)
    """
    def fit(self, data):
        return self

    def transform(self, df):
        df.loc[df.index[-1], "timestamp"] = df.loc[df.index[-2], "timestamp"] + pd.to_timedelta(1,unit='h') 
            
        return df


class SplitTimestamp(BaseEstimator, TransformerMixin):
    """
    Split timestamp to year, month, day, hour
    """
    def fit(self, data):
        return self

    def transform(self, df):

        df.loc[df.index[-1], "hour"]  =  df.loc[df.index[-1], "timestamp"].hour
        df.loc[df.index[-1], "day"]  =  df.loc[df.index[-1], "timestamp"].day
        df.loc[df.index[-1], "month"]  =  df.loc[df.index[-1], "timestamp"].month
        df.loc[df.index[-1], "year"]  =  df.loc[df.index[-1], "timestamp"].year
        
        df['hour'] = np.int64(df['hour'])
        df['day'] = np.int64(df['day'])
        df['month'] = np.int64(df['month'])
        df['year'] = np.int64(df['year'])
        
        return df


class IncrementLaggedUnderlyings(BaseEstimator, TransformerMixin):
    """
    Increment lags of underlying variables (base vars without transforms)
    by looking at column with base variable
    """

    def __init__(self, vars, lags):
        self.lags = lags
        self.vars = vars

    def fit(self, data):
        return self

    def transform(self, df):

        # collect all lags of base variables
        all_vars = [x for x in self.vars]
        all_lags = []
        for i in all_vars:
            for j in self.lags:
                all_lags.append(i + '_lag_' + str(j))
        
        # fill the last row of column with lagged base variables
        for i in all_lags:
            # i[-1] is the last character in a string ('2' in 'tempereture_lag_2')
            # .split('_lag_') gets everything before '_lag_'
            past = -int(i[-1])-1 # how far to look into the past, i.e. rows up in underlying var
            df.loc[df.index[-1], i]  =  df.loc[df.index[past], i.split('_lag_')[0]]
        
        return df


class IncrementLaggedVelocities(BaseEstimator, TransformerMixin):
    """
    Increment lags of velocities (differences) by looking at column with base variable
    """

    def fit(self, data):
        return self

    def transform(self, df):

        # collect all velos of base variables
        all_velos = [x for x in list(df) if '_velo_' in x]
        
        for i in all_velos:
            diff = i.split('_velo_')[1][0] # the diff, i.e. '1' in 'temperature_velo_1_lag_2'
            past = -int(i[-1])-1 # how far to look into the past, i.e. rows up in underlying var
            df.loc[df.index[-1], i]  = df['temperature'].diff(diff).iloc[past]

        # print(df[['year', 'month', 'day', 'hour', 'temperature', 'temperature_lag_1',
        #           'temperature_velo_1_lag_1', 'temperature_velo_1_lag_2',
        #           'temperature_velo_2_lag_1','temperature_velo_2_lag_3']].tail(10))

        return df

class IncrementLaggedAccelerations(BaseEstimator, TransformerMixin):
    """
    Increment lags of accelerations (diff of diffs) by looking at column with base variable
    """

    def fit(self, data):
        return self

    def transform(self, df):

        # collect all accelerations of base variables
        all_accs = [x for x in list(df) if '_acc_' in x]
        
        for i in all_accs:
            diff = i.split('_acc_')[1][0] # the diff, i.e. '2' in 'temperature_acc_2_lag_2'
            past = -int(i[-1])-1 # how far to look into the past, i.e. rows up in underlying var
            df.loc[df.index[-1], i]  = df['temperature'].diff(diff).diff(periods = 1).iloc[past]

        print(df[['day', 'hour', 'temperature', 'temperature_lag_1',
                  'temperature_velo_1_lag_1', 'temperature_velo_1_lag_2', 'temperature_velo_2_lag_2',
                  'temperature_velo_2_lag_1', 'temperature_acc_1_lag_1','temperature_acc_2_lag_1']].tail(10))

        return df