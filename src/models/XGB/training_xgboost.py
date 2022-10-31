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
import os
import logging

# get data
import dvc.api
from io import StringIO


# function to load data
def model_data_loader(target):
    ''' Loads the data with the right target variable for the model

    @param target: the target variable of the model, i.e. what to predict
    @return: X_train, y_train, X_test, y_test
    
    '''
    logging.info('LOAD DATA FOR MODEL')
    
    dir_name = os.path.join('data_dvc', 'processed') 
    format = 'csv'

    # read data to pandas df
    file_train = 'train_' + target
    train_dvc = dvc.api.read(os.path.join(dir_name, file_train + '.' + format), mode = 'r')
    train_dvc = StringIO(train_dvc)
    train_dvc = pd.read_csv(train_dvc, delimiter=',', header=0)

    file_test = 'test_' + target
    test_dvc = dvc.api.read(os.path.join(dir_name, file_test + '.' + format), mode = 'r')
    test_dvc = StringIO(test_dvc)
    test_dvc = pd.read_csv(test_dvc, delimiter=',', header=0)

    X_train = train_dvc.iloc[:, 1:]
    y_train = train_dvc.iloc[:, 0]
    X_test = test_dvc.iloc[:, 1:]
    y_test = test_dvc.iloc[:, 0]

    return X_train, y_train, X_test, y_test



#### Train a model


def train_xgb(cfg: data_config, target, X_train, y_train, X_test, y_test):
    ''' Model training with XGBoost

    @param target: the target variable to predict
    @param X_train: the training data to be used
    '''

    logging.info('START TRAINING')

    mlflow.set_experiment(experiment_name='Weather') 

    # max_tuning_runs: the maximum number of child Mlflow runs created for hyperparameter search estimators
    mlflow.sklearn.autolog(max_tuning_runs=None) 

    run_name = 'XGB, target: ' + target

    with mlflow.start_run(run_name= run_name):

        # Start training the model
        t0 = time()
        model = xgb.XGBRegressor()
        
        # Hyperparameter-tuning with grid search
        parameter_grid = {
        'n_estimators': cfg.xgb.n_estimators,
        'max_depth': cfg.xgb.max_depth,
        'learning_rate': cfg.xgb.learning_rate,
        'min_child_weight': cfg.xgb.min_child_weight
        }
        # Specifiy splitting for Time series cross validation
        tscv = TimeSeriesSplit(n_splits = cfg.cv.n_splits)

        # scoring: Strategy to evaluate the performance of the cross-validated model on the test set; = None -> sklearn.metrics.r2_score 
        lr= GridSearchCV(model, parameter_grid, cv=tscv, scoring=None, verbose=2)
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

    initialize(config_path="..\..\conf", job_name="config")
    cfg = compose(config_name="config")

    # This would also work:
    # cfg = omegaconf.OmegaConf.load(os.path.join(os.getcwd(), "src\conf\config.yaml")) 

    #### Train a model
    X_train, y_train, X_test, y_test = model_data_loader(target = cfg.model.target)
    print(y_train)

    train_xgb(cfg = cfg, target = cfg.model.target, X_train = X_train)


    print('END')









