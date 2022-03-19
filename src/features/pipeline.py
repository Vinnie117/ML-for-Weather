import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import numpy as np

# load data
df_raw = pd.read_csv(r'A:\Projects\ML-for-Weather\data\raw\test_simple.csv') 

# Cleaning
df = df_raw.drop(columns=["station_id", "dataset"])
df = df.pivot(index="date", columns="parameter", values="value").reset_index()

# Split
train, test = train_test_split(df, test_size=0.2, shuffle = False)

#################################################
# Preparation

################################################
class FeatureGenerator(BaseEstimator, TransformerMixin):

    #p stants for the number of step
    def __init__(self, p):
        self._p = p

    def fit(self, X, y):
        return self

    def transform(self, X): #This X is in reality going to be the "y"
        X1 = pd.concat([X.shift(+ i) for i in range(1,self._p+1)],axis=1)
        return X1 #.fillna(0)

feature_gen = FeatureGenerator(p = 4)
feature_gen.transform(train["wind_speed"])
pipeline= Pipeline(steps = [('feature_gen', feature_gen)])

a = pipeline.fit(train["wind_speed"], train["wind_speed"])


######################################################

# create a transformer
class InsertLags(BaseEstimator, TransformerMixin):
    """
    Automatically Insert Lags
    """
    def __init__(self, lags):
        self.lags = lags

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        original_cols=list(range(len(X[0,:])))
        for lag in self.lags:
            X_lagged=pd.DataFrame(X[:,original_cols]).shift(lag)
            X=np.concatenate((X,X_lagged), axis=1)
        return X

class Debug(BaseEstimator, TransformerMixin):
    """
    View pipeline data
    """
    def transform(self, X):
        print(X.shape)
        self.shape = shape
        # what other output you want
        return X

    def fit(self, X, y=None, **fit_params):
        return self

class Debugger(BaseEstimator, TransformerMixin):
    """
    View pipeline data - another class
    """
    def transform(self, data):

        # Here you just print what you need + return the actual data. You're not transforming anything. 

        print("Shape of Pre-processed Data:", data.shape)
        print(pd.DataFrame(data).head())
        return data

    def fit(self, data, y=None, **fit_params):

        # No need to fit anything, because this is not an actual  transformation. 

        return self

add_lags=InsertLags([1,2,3])
print(add_lags)

# the sample data
test=np.array([[1,3,5,7,9],
      [2,4,6,8,10]]).T
print(test)

# Works: add_lags transforms "test"-data
add_lags.fit_transform(test)
print(add_lags.fit_transform(test))

pipeline2 = Pipeline(steps = [('lags', add_lags),('debugger', Debugger())])

#b = pipeline2.fit(train["wind_speed"])
#print(b)
c = pipeline2.fit_transform(train["wind_speed"])
print(c)
d = pipeline2.named_steps["debug"].shape
print(d)

#############################################

# https://scikit-learn.org/stable/modules/compose.html#columntransformer-for-heterogeneous-data

# https://stackoverflow.com/questions/57149361/how-to-make-a-custom-sklearn-transformer-for-time-series
# https://stackoverflow.com/questions/37152723/how-to-auto-discover-a-lagging-of-time-series-data-in-scikit-learn-and-classify
# https://stackoverflow.com/questions/39840890/how-to-use-lagged-time-series-variables-in-a-python-pandas-regression-model
# https://stackoverflow.com/questions/54160370/how-to-use-sklearn-column-transformer
# https://machinelearningmastery.com/columntransformer-for-numerical-and-categorical-data/

################################################
test.to_csv(r'A:\Projects\ML-for-Weather\data\processed\test.csv', header=True, index=False)
train.to_csv(r'A:\Projects\ML-for-Weather\data\processed\train.csv', header=True, index=False)