import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from test_lags import InsertLags
from test_lags import df

class Debugger(BaseEstimator, TransformerMixin):
    """
    View pipeline data - another class
    """
    def fit(self, data, y=None, **fit_params):
        # No need to fit anything, because this is not an actual  transformation. 
        return self
    
    def transform(self, data):
        # Here you just print what you need + return the actual data. You're not transforming anything. 
        print("Shape of Pre-processed Data:", data.shape)
        print(pd.DataFrame(data).head())
        return data

####
pipe2 = Pipeline([
    ("lags", InsertLags([1,2,3])),
    ("debug", Debugger())
])

# Pipeline creates lags and prints the data
pipe2.fit_transform(df)
data = pipe2.fit_transform(df)


print("END")