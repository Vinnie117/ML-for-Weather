import pandas as pd
import numpy as np
from time import time
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


# load data
train = pd.read_csv(r'A:\Projects\ML-for-Weather\data\processed\train.csv') 
test = pd.read_csv(r'A:\Projects\ML-for-Weather\data\processed\test.csv') 

X_train = train[['month', 'day', 'hour',
                 'temperature_lag_1', 'cloud_cover_lag_1', 'wind_speed_lag_1']]
y_train = train['temperature']

X_test = test[['month', 'day', 'hour',
                 'temperature_lag_1', 'cloud_cover_lag_1', 'wind_speed_lag_1']]

y_test = test['temperature']                 

print(X_train)
print(y_train)

# Adjust data for scikit learn
y_train = y_train.iloc[1:]
X_train = X_train.dropna()
y_train = y_train.dropna()
X_train = X_train.to_numpy()
y_train = y_train.to_numpy()

y_test = y_test.iloc[3:]              # why 3?
X_test = X_test.dropna()
X_test = X_test.to_numpy()
y_test = y_test.dropna()
y_test = y_test.to_numpy()

print(X_train)
print(y_train)

# Train a model
t0 = time()
model_reg = linear_model.LinearRegression()
model_reg.fit(X_train, y_train)
print("Training duration %0.3fs" % (time() - t0))

# The model
print("Coefficients: \n", model_reg.coef_)

# Make predictions using the training / testing set
y_pred_train = model_reg.predict(X_train)
y_pred_test = model_reg.predict(X_test)


# Model evaluation on test data
print("Mean squared error: %.2f" % mean_squared_error(y_train, y_pred_train))
print("Coefficient of determination: %.2f" % r2_score(y_train, y_pred_train))

# Model evaluation on test data
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred_test))
print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred_test))


# Make a plot
# plt.scatter(X_test, y_test, color="black")
# plt.plot(X_test, y_pred, color="blue", linewidth=3)

# plt.xticks(())
# plt.yticks(())

# plt.show()


print("END")



# To do
# - create (lagged) trend and seasonality variables
# - make data suitable for sklearn in pipeline
# - scale variables to normalize coefficient between 0 and 1




