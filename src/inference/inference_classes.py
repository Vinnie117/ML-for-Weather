from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class IncrementTime(BaseEstimator, TransformerMixin):
    """
    Increment latest available timestamp by +1 (in last row)
    """
    def fit(self, data):
        return self

    def transform(self, df):

        df.loc[df.index[-1], "timestamp"] = df.loc[df.index[-2], "timestamp"]+ pd.to_timedelta(1,unit='h') 
            
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