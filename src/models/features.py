######## Extract features to track them with mlflow ########

#
#
#
# -> THIS FILE IS A PLAYGROUND TO TEST FEATURE TRACKING FUNCTION
#
#
#


import sys
sys.path.append('A:\Projects\ML-for-Weather\src') 
import pandas as pd
import omegaconf
import os
import re
from hydra import compose, initialize
from config import data_config 
from collections import defaultdict

# get data
train = pd.read_csv(r'A:\Projects\ML-for-Weather\data\processed\train.csv', delimiter=',', header=0)
test = pd.read_csv(r'A:\Projects\ML-for-Weather\data\processed\test.csv', delimiter=',', header=0)
X_train = train.iloc[:, 1:]
y_train = train.iloc[:, 0]
X_test = test.iloc[:, 1:]
y_test = test.iloc[:, 0]

features = list(X_train)
print('features are: ', features)

initialize(config_path="..\conf", job_name="config")
cfg = compose(config_name="config")


def track_features(cfg: data_config):
    '''This function tracks the features that have been used in order to train the model
    
    @param cfg.transform.vars: a list of variables that have transformations
    @return: a nested dictionary with tranformed variabes and their lags

    '''
    d = {} 
    d['time'] = ['month', 'day', 'hour']
    for i in cfg.transform.vars:
        d[i] = {}

        list_transforms = []    # a list to collect transforms of each variable for all features
        list_lags = []          # a list to collect lags of each feature
        for j in features:
            transform = re.search(rf"(?<={i}_).*?(?=_lag)", j)

            if transform:
                transform = transform.group(0)
                print('transform: ', transform)

                lag = re.search(rf"(?=lag)(.*)", j)
                lag = lag.group(0)
                print(lag)

                if transform not in list_transforms:    # reset lags for a new transform
                    list_lags = []

                list_transforms.append(transform)       # keep track for which transforms lags are collected
                list_transforms = sorted(list(set(list_transforms)))
                
                list_lags.append(lag)
                d[i][transform] = sorted(list(set(list_lags)))

    return d

d = track_features(cfg = cfg)

print(d)

print("END")

