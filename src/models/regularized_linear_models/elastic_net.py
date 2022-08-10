import pandas as pd
import numpy as np
from time import time
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sys 
sys.path.append('A:\Projects\ML-for-Weather\src')  # import from parent directory
from models.functions import adjustedR2
import mlflow
from joblib import dump, load
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit

# train = np.genfromtxt(r'A:\Projects\ML-for-Weather\data\processed\train_array.csv', delimiter=',')
# test = np.genfromtxt(r'A:\Projects\ML-for-Weather\data\processed\test_array.csv', delimiter=',')

# Select data -> if pandas dataframe
train = pd.read_csv(r'A:\Projects\ML-for-Weather\data\processed\train_array.csv', delimiter=',', header=0)
test = pd.read_csv(r'A:\Projects\ML-for-Weather\data\processed\test_array.csv', delimiter=',', header=0)
X_train = train.iloc[:, 1:]
y_train = train.iloc[:, 0]
X_test = test.iloc[:, 1:]
y_test = test.iloc[:, 0]

print(X_train)

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    adjusted_r2 = adjustedR2(r2, X_test)
    return rmse, mae, r2, adjusted_r2


#### Train a model

# set model parameters
alpha = [0.3, 0.4, 0.5, 0.6]
l1_ratio = [0.3, 0.4, 0.5, 0.6]

# Specifiy splitting for Time series cross validation
tscv = TimeSeriesSplit(n_splits = 5)

# Hyperparameter-tuning with grid search
parameters = {'alpha':alpha, 
              'l1_ratio':l1_ratio} 


mlflow.set_experiment(experiment_name='Elastic Nets') 

# max_tuning_runs: the maximum number of child Mlflow runs created for hyperparameter search estimators
mlflow.sklearn.autolog(max_tuning_runs=None) 

with mlflow.start_run():

    # Start training the model
    t0 = time()
    model = ElasticNet(random_state=42)

    # scoring: Strategy to evaluate the performance of the cross-validated model on the test set; = None -> sklearn.metrics.r2_score 
    lr= GridSearchCV(model, parameters, cv=tscv, scoring=None, verbose=2)
    lr.fit(X_train, y_train)
    duration = time() - t0

    # automatically the model with best params
    predicted_values = lr.predict(X_test)

    (rmse, mae, r2, adjusted_r2) = eval_metrics(y_test, predicted_values)

    # Logging model performance to mlflow -> is only done for the best model
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

#### To do
# -> compare mlflow results with GridSearchCV.cv_results_ ?? -> or check in isolation
# -> fully understand all metrics in GridSearchCV.cv_results_


#### Best model from GridsearchCV
# -> prediction on test data (and resulting metrics) is only done for the best model!
# -> GridseachCV yields the best model by default
#   - https://stackoverflow.com/questions/30102973/how-to-get-best-estimator-on-gridsearchcv-random-forest-classifier-scikit
#   - https://stackoverflow.com/questions/49455806/gridsearchcv-final-model
#   - https://stackoverflow.com/questions/55581950/how-to-use-optimal-parameters-identified-gridsearchcv

#### Scoring metrics (explanation)
# - can make an own scorer: 
#   - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html#sklearn.metrics.make_scorer
# - default scorer is r2 for regression -> relevant for mean_test_score, split0_test_score, split0_train_score etc.
#   - https://datascience.stackexchange.com/questions/94261/whats-the-default-scorer-in-sci-kit-learns-gridsearchcv
# https://stackoverflow.com/questions/44947574/what-is-the-meaning-of-mean-test-score-in-cv-result
# https://stackoverflow.com/questions/49899298/how-does-gridsearchcv-compute-training-scores 

#### Storage
# Parameter combinations of all runs are stored in cv_results_
# - GridSearchCV.cv_results_['params']
# - https://stackoverflow.com/questions/34274598/does-gridsearchcv-store-all-the-scores-for-all-parameter-combinations


