import pandas as pd
from time import time
from sklearn.svm import SVR
import sys 
sys.path.append('A:\Projects\ML-for-Weather\src')  # import from parent directory
from models.functions import eval_metrics
import mlflow
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from config import data_config 
from hydra import compose, initialize


# Select data 
train = pd.read_csv(r'A:\Projects\ML-for-Weather\data\processed\train_std_array.csv', delimiter=',', header=0)
test = pd.read_csv(r'A:\Projects\ML-for-Weather\data\processed\test_std_array.csv', delimiter=',', header=0)
X_train = train.iloc[:, 1:]
y_train = train.iloc[:, 0]
X_test = test.iloc[:, 1:]
y_test = test.iloc[:, 0]


#### Train a model

initialize(config_path="..\..\conf", job_name="config")
cfg = compose(config_name="config")

# # This would also work:
# import os, omegaconf
# cfg = omegaconf.OmegaConf.load(os.path.join(os.getcwd(), "src\conf\config.yaml")) 


def train_svm(cfg: data_config):

    mlflow.set_experiment(experiment_name='Weather') 

    # max_tuning_runs: the maximum number of child Mlflow runs created for hyperparameter search estimators
    mlflow.sklearn.autolog(max_tuning_runs=None) 

    with mlflow.start_run(run_name='svm'):

        # Start training the model
        t0 = time()
        model = SVR()

        # Hyperparameter-tuning with grid search
        param_grid = {'C': cfg.svm.C, 'gamma':cfg.svm.gamma, 'kernel': cfg.svm.kernel}
        # Specify splitting for Time series cross validation
        tscv = TimeSeriesSplit(n_splits = cfg.cv.n_splits)

        # scoring: Strategy to evaluate the performance of the cross-validated model on the test set; = None -> sklearn.metrics.r2_score 
        lr= GridSearchCV(model, param_grid, cv=tscv, scoring=None, verbose=2)
        lr.fit(X_train, y_train)
        duration = time() - t0

        # automatically the model with best params
        predicted_values = lr.predict(X_test)

        (rmse, mae, r2, adjusted_r2) = eval_metrics(y_test, predicted_values, X_test)

        # Logging model performance to mlflow -> is only done for the best model
        mlflow.log_param("C", cfg.svm.C)
        mlflow.log_param("gamma", cfg.svm.gamma)
        mlflow.log_param("kernel", cfg.svm.kernel)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("adjusted_r2", adjusted_r2)
        mlflow.log_metric('duration', duration)


if __name__ == "__main__":
    train_svm(cfg = cfg)
    print('END')