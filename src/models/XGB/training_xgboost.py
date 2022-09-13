import pandas as pd
from time import time
import xgboost as xgb
import sys
sys.path.append('A:\Projects\ML-for-Weather\src')  # import from parent directory
from models.functions import eval_metrics, track_features
import mlflow
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from config import data_config 
from hydra import compose, initialize
import yaml
import joblib

# get data
train = pd.read_csv(r'A:\Projects\ML-for-Weather\data\processed\train.csv', delimiter=',', header=0)
test = pd.read_csv(r'A:\Projects\ML-for-Weather\data\processed\test.csv', delimiter=',', header=0)
X_train = train.iloc[:, 1:]
y_train = train.iloc[:, 0]
X_test = test.iloc[:, 1:]
y_test = test.iloc[:, 0]

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# print(X_train)
# print(len(list(X_train)))
# print(list(X_train))

############################################################################

# training is somehow super slow -> why?
# shape of predicted values seems weird

import dvc.api
from io import StringIO
train_dvc = dvc.api.read(r'data_dvc\processed\train.csv', mode = 'r')
test_dvc = dvc.api.read(r'data_dvc\processed\test.csv', mode = 'r')
train_dvc = StringIO(train_dvc)
test_dvc = StringIO(test_dvc)

train_dvc = pd.read_csv(train_dvc, delimiter=',', header=0)
test_dvc = pd.read_csv(test_dvc, delimiter=',', header=0)

X_train2 = train_dvc.iloc[:, 1:]
y_train2 = train_dvc.iloc[:, 0]
X_test2 = test_dvc.iloc[:, 1:]
y_test2 = test_dvc.iloc[:, 0]

# print(X_train2.shape)
# print(y_train2.shape)
# print(X_test2.shape)
# print(y_test2.shape)

# print(y_test.shape)
# print(len(list(y_test)))
# print(list(y_test))


# # data is the same
# if (X_train == X_train2).all:
#     print("ES IST GLEICH!!!111!1!1!!11")




#### Train a model

initialize(config_path="..\..\conf", job_name="config")
cfg = compose(config_name="config")

# This would also work:
# cfg = omegaconf.OmegaConf.load(os.path.join(os.getcwd(), "src\conf\config.yaml")) 



def train_xgb(cfg: data_config, X_train = X_train2):

    mlflow.set_experiment(experiment_name='Weather') 

    # max_tuning_runs: the maximum number of child Mlflow runs created for hyperparameter search estimators
    mlflow.sklearn.autolog(max_tuning_runs=None) 

    with mlflow.start_run(run_name='XGB, dataset = [dvc id]'):

        # Start training the model
        t0 = time()
        model = xgb.XGBRegressor()
        
        # Hyperparameter-tuning with grid search
        parameters = {
        'n_estimators': [100, 400, 800],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.05, 0.1, 0.20],
        'min_child_weight': [1, 10, 100]

        # # FOR TESTING
        # 'n_estimators': [100],
        # 'max_depth': [3],
        # 'learning_rate': [0.05],
        # 'min_child_weight': [1]
        }
        # Specifiy splitting for Time series cross validation
        tscv = TimeSeriesSplit(n_splits = cfg.cv.n_splits)

        # scoring: Strategy to evaluate the performance of the cross-validated model on the test set; = None -> sklearn.metrics.r2_score 
        lr= GridSearchCV(model, parameters, cv=tscv, scoring=None, verbose=2)
        lr.fit(X_train, y_train)
        duration = time() - t0

        # automatically the model with best params
        predicted_values = lr.predict(X_test)

        (rmse, mae, r2, adjusted_r2) = eval_metrics(y_test, predicted_values, X_test)

        # Track the features used for model training
        dict_features = track_features(cfg = cfg, X_train = X_train)
        with open('artifacts/features/data_features.yaml', 'w') as outfile:
            yaml.dump(dict_features, outfile, default_flow_style=False)
        print(yaml.dump(dict_features, default_flow_style=False))

        amount_features = len(list(X_train))

        # Logging model performance to mlflow -> is only done for the best model
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("adjusted_r2", adjusted_r2)
        mlflow.log_metric('duration', duration)
        mlflow.log_metric('amount_features', amount_features)
        mlflow.log_param('date_start', cfg.date.start)
        mlflow.log_param('date_end', cfg.date.end)      
        mlflow.log_artifact("artifacts/features/data_features.yaml")

    # save model
    joblib.dump(lr, 'artifacts/models/xgb.joblib')


if __name__ == "__main__":
    train_xgb(cfg = cfg)


    print('END')









