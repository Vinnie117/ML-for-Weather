######## Extract features to track them with mlflow ########
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
    '''This function tracks the features that have been used in order to train the model'''

    # when no preds are given, all are used
    if cfg.model.predictors == []:                                  
        d = {} 
        d['time'] = ['month', 'day', 'hour']
        for i in cfg.transform.vars:
            list_transforms = []
            d[i] = {}
            for j in features:
                transform = re.search(rf"(?<={i}_).*?(?=_lag)", j)  
                if transform:

                    # extract variable transformation
                    transform = transform.group(0)
                    list_transforms.append(transform)
            list_transforms = sorted(list(set(list_transforms)))

            # Insert lags into nested dict
            for k in list_transforms:
                d[i][k] = []
                for l in cfg.diff.lags:
                    d[i][k].append('lag_' + str(l))

    # the case for when predictors are specified
    else:
        d = {} 
        d['time'] = ['month', 'day', 'hour']
        for i in cfg.transform.vars:
            list_transforms = []
            d[i] = {}
            #list_lags = []
            for j in features:
                #list_lags = []
                transform = re.search(rf"(?<={i}_).*?(?=_lag)", j)
                if transform:
                    
                    # extract variable transformation
                    transform = transform.group(0)
                    print('transform: ', transform)
                    #list_transforms.append(transform)
                    #list_transforms = sorted(list(set(list_transforms)))
                    #print('list_transform: ', list_transforms)
                    
                    # create inner dict
                    d[i][transform] = []    # -> This is problematic: new list is initialized after each step.


                    # lag = re.search(rf"(?=lag)(.*)", j)
                    # lag = lag.group(0)
                    # print(lag)
                    # d[i][transform].append(lag)




                    # list_lags = []
                    # # extract lag
                    # lag = re.search(rf"(?=lag)(.*)", j)
                    # lag = lag.group(0)
                    # print('lag: ', lag)
                    # list_lags.append(lag)
                    # print('list_lags: ', list_lags)

                    # list_lags = sorted(list(set(list_lags)))
                    # d[i][transform] = list_lags


    return d

d = track_features(cfg = cfg)

print(d)

####
# Problem velo_1 gets overwritten by velo_2, because list_transform is longer

# extract 'lag_1' with (?<=temperature_velo_1_).*
# extract  temperature_velo_1_ with .*?(?=lag)
# (?=lag)(.*)
# lag(.*)$

# TO DO: Refactor the function
# TO DO: Include raw variable



print("END")


'''
ML FLow Tracking should say:

The following lags are used in this experiment

- Option 1
base: temperature, wind_speed, cloud_cover
velo_1:
    lag_1: temperature, wind_speed, cloud_cover
    lag_2: temperature, wind_speed, cloud_cover
    lag_3: temperature, wind_speed, cloud_cover
velo_2:
    lag_1: temperature, wind_speed, cloud_cover
    lag_2: temperature, wind_speed, cloud_cover
    lag_3:
acc_1:
    lag_1:
    lag_2:
    lag_3:
acc_2:
    lag_1:
    lag_2:
    lag_3:

- Option 2
base: month, day, hour, temperature, wind_speed, cloud_cover
temperature:
    velo_1: lag_1, lag_2, lag_3
    velo_2: lag_1, lag_2, lag_3
    acc_1:
    acc_2:
wind_speed:
    velo:_1
    ...

- Many more options

-> 3 Ebenen, welche Info auf welche Ebene?
-> so ,dass die wichtigere Info auf der höheren Ebene steht
-> Option 2!!!

-> Testn, mit verschiedenen Listen durchspielen!

'''













