import pandas as pd
import numpy as np
from time import time
from sklearn import linear_model


# load data
train = pd.read_csv(r'A:\Projects\ML-for-Weather\data\processed\train.csv') 

X_train = train[['month', 'day', 'hour',
                 'temperature_lag_1', 'cloud_cover_lag_1', 'wind_speed_lag_1']]
y_train = train['temperature']

print(X_train)
print(y_train)

# Adjust data for scikit learn
y_train = y_train.iloc[1:]
X_train = X_train.dropna()
y_train = y_train.dropna()
X_train = X_train.to_numpy()
y_train = y_train.to_numpy()

print(X_train)
print(y_train)

# Train a model
t0 = time()
model_reg = linear_model.LinearRegression()
model_reg.fit(X_train, y_train)
print("Training duration %0.3fs" % (time() - t0))

print("END")



# To do
# - create (lagged) trend and seasonality variables
# - make data suitable for sklearn in pipeline




