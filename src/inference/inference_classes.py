from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


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
            
        return df


class IncrementLaggedUnderlyings(BaseEstimator, TransformerMixin):
    """
    Increment lags of underlying variables (base vars without transforms)
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
            # .split('_lag_') gets everythin before '_lag_'
            df.loc[df.index[-1], i]  =  df.loc[df.index[-int(i[-1])-1], i.split('_lag_')[0]]
        
            
        return df