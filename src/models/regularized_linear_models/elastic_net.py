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
    return rmse, mae, r2


# Train a model
if __name__ == "__main__":

    # set model parameters
    # alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    # l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    alpha = 0.5
    l1_ratio = 0.5

    mlflow.create_experiment('Elastic Nets') 

    with mlflow.start_run('Elastic Nets'):

        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(X_train, y_train)

        predicted_qualities = lr.predict(X_test)

        (rmse, mae, r2) = eval_metrics(y_test, predicted_qualities)

        # Model performance
        print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        # logging to mlflow
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)


print('END')