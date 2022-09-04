############ File for custom functions ############

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def adjustedR2(r2, test_data):
    '''Custom function to calculate adjsuted R2
    
    @param r2: the R2 metric needed to calculate adjusted R2
    @param data: calculation of adjusted R2 requires number of ros and columns of test data
    @return adjustedR2: the adjusted R2 metric  
    '''
    adjustedR2 =  1 - (1 - r2) * ((test_data.shape[0] - 1) / (test_data.shape[0] - test_data.shape[1] - 1))
    return adjustedR2

def eval_metrics(actual, pred, X_test):
    '''Custom function to calculate different performance measures
    
    @param actual: actual y-values from test data
    @param pred: predicted y-values from model
    @param X_test: Test data (predictors)
    @return rmse, mae, r2, adjusted_r2: Root mean squared error, mean absolute error, R2 and adjusted R2
    
    '''

    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    adjusted_r2 = adjustedR2(r2, X_test)
    return rmse, mae, r2, adjusted_r2
















