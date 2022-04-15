import pandas as pd
import numpy as np
from time import time
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.inf)

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


#Make a plot
plt.scatter(X_test, y_test, color="black")
plt.plot(X_test, y_pred_test, color="blue", linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()


print("END")



# To do

# - implement config management
# - plot training / test error
# - implement adjusted R2 instead of R2
# - create (lagged) trend and seasonality variables
# - make data suitable for sklearn in pipeline
# - scale variables to normalize coefficient between 0 and 1




