############ File for custom functions ############

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def adjustedR2(r2, data):
    '''Custom function to calculate adjsuted R2'''

    adjustedR2 =  1 - (1 - r2) * ((data.shape[0] - 1) / (data.shape[0] - data.shape[1] - 1))
    return adjustedR2

def eval_metrics(actual, pred, X_test):
    '''Custom function to calculate different performance measures'''

    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    adjusted_r2 = adjustedR2(r2, X_test)
    return rmse, mae, r2, adjusted_r2
















