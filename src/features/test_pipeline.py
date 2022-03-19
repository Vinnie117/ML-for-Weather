import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import numpy as np

# Can use dataframe from another python file like this
#from test_lags import result
#print(result)


#################################################
# Debug Pipeline

# class Debug(BaseEstimator, TransformerMixin):
#     """
#     View pipeline data
#     """
#     def transform(self, X):
#         print(X.shape)
#         self.shape = shape
#         # what other output you want
#         return X

#     def fit(self, X, y=None, **fit_params):
#         return self

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








#pipeline2 = Pipeline(steps = [('debugger', Debugger())])

#b = pipeline2.fit(train["wind_speed"])
#print(b)
#c = pipeline2.fit_transform(train["wind_speed"])
#print(c)
#d = pipeline2.named_steps["debug"].shape
#print(d)

#############################################

# https://nishalsach.github.io/posts/2021-08-17-debugging-sklearn-pipelines/
# https://stackoverflow.com/questions/48743032/get-intermediate-data-state-in-scikit-learn-pipeline

################################################
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator


# class Debug(BaseEstimator, TransformerMixin):

#     def transform(self, X):
#         print(X.shape)
#         self.shape = shape
#         # what other output you want
#         return X

#     def fit(self, X, y=None, **fit_params):
#         return self

pipe = Pipeline([
    ("tf_idf", TfidfVectorizer()),
    ("debug", Debugger()),
    ("nmf", NMF())
])

data = pd.DataFrame([["Salut comment tu vas", "Hey how are you today", "I am okay and you ?"]]).T
data.columns = ["test"]

pipe.fit_transform(data.test)



print("END")