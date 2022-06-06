import pandas as pd
import numpy as np
from time import time
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import sys 
sys.path.append('A:\Projects\ML-for-Weather\src')  # import from parent directory
from features.pipeline_dataprep import pd_df
from models.functions import adjustedR2
import mlflow
from joblib import dump, load
from sklearn.model_selection import GridSearchCV

train = np.genfromtxt(r'A:\Projects\ML-for-Weather\data\processed\train_array.csv', delimiter=',')
test = np.genfromtxt(r'A:\Projects\ML-for-Weather\data\processed\test_array.csv', delimiter=',')

# Select data
X_train = train[:, 1:]
y_train = train[:, 0]
X_test = test[:, 1:]
y_test = test[:, 0]


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    adjusted_r2 = adjustedR2(r2, X_test)
    return rmse, mae, r2, adjusted_r2


# Train a model

# set model parameters
alpha = [0.3, 0.4, 0.5, 0.6]
l1_ratio = [0.3, 0.4, 0.5, 0.6]

parameters = {'alpha':alpha, 
              'l1_ratio':l1_ratio} 


mlflow.set_experiment(experiment_name='Elastic Nets') 

# max_tuning_runs: the maximum number of child Mlflow runs created for hyperparameter search estimators
mlflow.sklearn.autolog(max_tuning_runs=None) 

with mlflow.start_run():

    # Start training the model
    t0 = time()
    model = ElasticNet(random_state=42)
    lr= GridSearchCV(model, parameters)
    lr.fit(X_train, y_train)
    duration = time() - t0

    predicted_qualities = lr.predict(X_test)

    (rmse, mae, r2, adjusted_r2) = eval_metrics(y_test, predicted_qualities)

    # Logging model performance to mlflow
    mlflow.log_param("alpha", alpha)
    mlflow.log_param("l1_ratio", l1_ratio)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("adjusted_r2", adjusted_r2)
    mlflow.log_metric('duration', duration)




#if __name__ == "__main__":
print('END')


#### Combine gridsearch cv with mlflow ####

# Parameter combinations of all runs are stored in cv_results_
# - GridSearchCV.cv_results_['params']
# - https://stackoverflow.com/questions/34274598/does-gridsearchcv-store-all-the-scores-for-all-parameter-combinations

# need to log params in mlflow
# - loop through all parameters?
# - somehow build a function? What to put in? 
# - use autolog?
#   - https://www.mlflow.org/docs/latest/python_api/mlflow.sklearn.html 
#   - https://gist.github.com/liorshk/9dfcb4a8e744fc15650cbd4c2b0955e5


