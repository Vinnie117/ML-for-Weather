############ File for custom functions ############
import logging
import numpy as np
import re
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from config import data_config 
import os
import dvc.api
from io import StringIO
import pandas as pd


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


#@hydra.main(config_path="..\conf", config_name="config")
def track_features(cfg: data_config, X_train):
    '''This function tracks the features that have been used in order to train the model
    
    @param cfg.transform.vars: a list of variables that have transformations
    @param X_train: training data with predictors
    @return d: a nested dictionary with tranformed variabes and their lags

    '''
    features = list(X_train)
    d = {} 
    d['time'] = ['year' ,'month', 'day', 'hour']
    for i in cfg.transform.vars:
        d[i] = {}

        list_transforms = []    # a list to collect transforms of each variable for all features
        list_lags = []          # a list to collect lags of each feature
        for j in features:
            transform = re.search(rf"(?<={i}_).*?(?=_lag)", j)

            if transform:
                transform = transform.group(0)
                lag = re.search(rf"(?=lag)(.*)", j)
                lag = lag.group(0)
                
                if transform not in list_transforms:    # reset lags for a new transform
                    list_lags = []

                list_transforms.append(transform)       # keep track for which transforms lags are collected
                list_transforms = sorted(list(set(list_transforms)))
                
                list_lags.append(lag)
                d[i][transform] = sorted(list(set(list_lags)))

    return d
