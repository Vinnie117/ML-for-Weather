import pandas as pd
import numpy as np
from time import time
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import sys 
sys.path.append('A:\Projects\ML-for-Weather\src')  # import from parent directory
from models.functions import adjustedR2
import mlflow
from joblib import dump, load
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn import utils

#train = np.genfromtxt(r'A:\Projects\ML-for-Weather\data\processed\train_array.csv', delimiter=',')
#test = np.genfromtxt(r'A:\Projects\ML-for-Weather\data\processed\test_array.csv', delimiter=',')

train = pd.read_csv(r'A:\Projects\ML-for-Weather\data\processed\train_array.csv', delimiter=',')
test = pd.read_csv(r'A:\Projects\ML-for-Weather\data\processed\test_array.csv', delimiter=',')

# Select data
X_train = train.iloc[:, 1:]
y_train = train.iloc[:, 0]
X_test = test.iloc[:, 1:]
y_test = test.iloc[:, 0]

# Cannot pass floats to classifier -> convert to categories/classes
lab_enc = preprocessing.LabelEncoder()
y_train_encoded = lab_enc.fit_transform(y_train)
y_test_encoded = lab_enc.fit_transform(y_test)

#print(utils.multiclass.type_of_target(y_train_encoded))


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    adjusted_r2 = adjustedR2(r2, X_test)
    return rmse, mae, r2, adjusted_r2


#### Train a model

# Specifiy splitting for Time series cross validation
tscv = TimeSeriesSplit(n_splits = 5)

# defining parameter range
C =  [0.1, 1, 10, 100, 1000]
gamma = [1, 0.1, 0.01, 0.001, 0.0001]
kernel = ['rbf']

param_grid = {'C': C,
              'gamma':gamma,
              'kernel': kernel}

mlflow.set_experiment(experiment_name='Support Vector Machines') 

# max_tuning_runs: the maximum number of child Mlflow runs created for hyperparameter search estimators
mlflow.sklearn.autolog(max_tuning_runs=None) 

with mlflow.start_run():

    # Start training the model
    t0 = time()
    model = SVC(random_state=42)

    # scoring: Strategy to evaluate the performance of the cross-validated model on the test set; = None -> sklearn.metrics.r2_score 
    lr= GridSearchCV(model, param_grid, cv=tscv, scoring=None, verbose=2)
    lr.fit(X_train, y_train_encoded)
    duration = time() - t0

    # automatically the model with best params
    predicted_values = lr.predict(X_test)

    (rmse, mae, r2, adjusted_r2) = eval_metrics(y_test, predicted_values)

    # Logging model performance to mlflow -> is only done for the best model
    mlflow.log_param("C", C)
    mlflow.log_param("gamma", gamma)
    mlflow.log_param("kernel", kernel)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("adjusted_r2", adjusted_r2)
    mlflow.log_metric('duration', duration)






#if __name__ == "__main__":
print('END')