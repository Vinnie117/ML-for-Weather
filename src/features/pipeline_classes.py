import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

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
        print("Shape of data", data.shape)
        print(pd.DataFrame(data).head())
        return data


# class Cleaner(BaseEstimator, TransformerMixin):
#     """
#     Basic cleaning of raw data from API
#     """

#     def fit(self, dummy):
#         return self

#     def transform(self, input_data):
#         input_data = input_data.drop(columns=["station_id", "dataset"])
#         input_data = input_data.pivot(index="date", columns="parameter", values="value").reset_index()
#         self.input_data = input_data.rename(columns={'temperature_air_mean_200': 'temperature', 
#                                                 'cloud_cover_total': 'cloud_cover',
#                                                 'wind_speed': 'wind_speed'})
#         return self.input_data


class InsertLags(BaseEstimator, TransformerMixin):
    """
    Automatically insert lags
    """
    def __init__(self, lags):
        self.lags = lags

    def fit(self, X):
        return self

    def transform(self, X):
        data = X
        X = X.to_numpy()

        # indices of 'np array columns', e.g. array with 3 columns -> [0,1,2]
        col_indices=list(range(len(X[0,:])))
        col_indices = col_indices[4:] # weather variables start after 4th column (timestamp, month, day, hour are before)

        # create lags
        for lag in self.lags:
            X_lagged=pd.DataFrame(X[:,col_indices]).shift(lag)
            X=np.concatenate((X,X_lagged), axis=1)

        # create column names (= normal columns + lagged columns)
        cols = data.columns.tolist()
        lag_col_names = []
        for x in range(len(self.lags)):
            for y in cols[4:]:
                lag_col_names.append(str(y) + '_lag_' + str(self.lags[x]))
        print(lag_col_names)
        return pd.DataFrame(X, columns = cols + lag_col_names)


class Times(BaseEstimator, TransformerMixin):

    def fit(self, X):
        return self

    def transform(self, data):
        # convert to CET (UTC +1), then remove tz
        data['timestamp'] = pd.to_datetime(data['date']).dt.tz_convert('Europe/Berlin').dt.tz_localize(None)
        data['month'] =  data['timestamp'].dt.month
        data['day'] =  data['timestamp'].dt.day 
        data['hour'] =  data['timestamp'].dt.hour
        data = data.drop('date', 1)

        #reorder columns
        cols = list(data.columns)
        cols = cols[-4:] + cols[:len(cols)-4]
        data = data[cols]

        return data