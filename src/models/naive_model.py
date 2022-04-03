import pandas as pd
import numpy as np
from time import time
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.inf)

# load data
train = pd.read_csv(r'A:\Projects\ML-for-Weather\data\processed\train.csv') 
test = pd.read_csv(r'A:\Projects\ML-for-Weather\data\processed\test.csv') 



###################################################################
X_train = train[['month', 'day', 'hour',
                 'temperature_lag_1', 'cloud_cover_lag_1', 'wind_speed_lag_1']]
y_train = train['temperature']

X_test = test[['month', 'day', 'hour',
                 'temperature_lag_1', 'cloud_cover_lag_1', 'wind_speed_lag_1']]

y_test = test['temperature']                 


# Adjust data for scikit learn

# training data
#print(y_train)
#y_train = y_train.iloc[1:]
X_train = X_train.dropna()

X_train = X_train.to_numpy()
y_train = y_train.to_numpy()

# test data
#y_test = y_test.iloc[1:]             

print(len(X_test))
print(X_test)                                 # there should be no NaN for lagged vars here!
X_test = X_test.dropna()
X_test = X_test.to_numpy()
y_test = y_test.to_numpy()

print(len(X_test))

######################################################################


# train = np.genfromtxt(r'A:\Projects\ML-for-Weather\data\processed\train_array.csv', delimiter=',')
# test = np.genfromtxt(r'A:\Projects\ML-for-Weather\data\processed\test_array.csv', delimiter=',')

# print(len(train))
# print(train)
# print(len(test))
# print(test)

# X_train = train[:, 1:]
# y_train = train[:, 0]
# X_test = test[:, 1:]
# y_test = test[:, 0]


# #print(train)
# print(X_train)
# print(y_train)
# print(X_test)
# print(len(X_test))
# print(y_test)
# print(len(y_test))


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
print(len(y_test))
print(len(y_pred_test))
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred_test))
print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred_test))


# Make a plot
# plt.scatter(X_test, y_test, color="black")
# plt.plot(X_test, y_pred, color="blue", linewidth=3)

# plt.xticks(())
# plt.yticks(())

# plt.show()


print("END")


# Question
# why is model evaluation different for both approaches?
# why is array y_test different for both approaches?
    # y_test is too long? (1752) -> but do not drop first 3 rows, only 1!
    # start of data is the same -> examine the end -> also the same
# for the first approach, len(y_test) = 1752 and len(y_pred_test) = 1750
    # the second approach has only len = 1750
# -> the raw csv data 'test' and 'test_array' differ in length!!!
    # but is it considered when loading the data?
# -> X_test is not correct (in first approach) -> first row should not be NaN
    # in second approach it seems correct
    # bug already starts in the pipeline
    # bug discovered and fixed




# To do
# - Select one approach -> bottom (preferred) or top
    # - adjust Prepare()
# - create (lagged) trend and seasonality variables
# - make data suitable for sklearn in pipeline
# - scale variables to normalize coefficient between 0 and 1




