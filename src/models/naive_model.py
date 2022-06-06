import pandas as pd
import numpy as np
from time import time
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import sys 
sys.path.append('A:\Projects\ML-for-Weather\src')  # import from parent directory
from features.pipeline_dataprep import pd_df
from models.functions import adjustedR2
#np.set_printoptions(threshold=np.inf)
from joblib import dump, load

train = np.genfromtxt(r'A:\Projects\ML-for-Weather\data\processed\train_array.csv', delimiter=',')
test = np.genfromtxt(r'A:\Projects\ML-for-Weather\data\processed\test_array.csv', delimiter=',')

# Select data
X_train = train[:, 1:]
y_train = train[:, 0]
X_test = test[:, 1:]
y_test = test[:, 0]


# Train a model
t0 = time()
model_reg = linear_model.LinearRegression()
model_reg.fit(X_train, y_train)
print("Training duration %0.3fs" % (time() - t0))

# The model
print("Coefficients: \n", model_reg.coef_)

####
# Make predictions using the training / testing set
y_pred_train = model_reg.predict(X_train)
y_pred_test = model_reg.predict(X_test)

# Model evaluation on training data
print("Mean squared error: %.2f" % mean_squared_error(y_train, y_pred_train))
print("Coefficient of determination: %.2f" % r2_score(y_train, y_pred_train))

# Model evaluation on test data
print(len(y_test))
print(len(y_pred_test))
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred_test))
print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred_test))


# Implementation of adjusted R2
r2 = r2_score(y_test, y_pred_test)
print(r2)
print(X_train.shape[0])
print(X_train.shape[1])
# for training data
adj_r2_training = adjustedR2(r2, X_train)
print("Adjusted R2 on training data: %.4f" % adj_r2_training)

#for test data
adj_r2_test = adjustedR2(r2, X_test)
print("Adjusted R2 on test data: %.4f" % adj_r2_test)

# save the model
dump(model_reg, r'A:\Projects\ML-for-Weather\src\models\saved\naive_reg.joblib') 

print("END")