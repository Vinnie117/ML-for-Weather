######## Extract features to track them with mlflow ########
import sys
sys.path.append('A:\Projects\ML-for-Weather\src') 
import pandas as pd
import omegaconf
import os
import re
from hydra import compose, initialize
from config import data_config 

# get data
train = pd.read_csv(r'A:\Projects\ML-for-Weather\data\processed\train.csv', delimiter=',', header=0)
test = pd.read_csv(r'A:\Projects\ML-for-Weather\data\processed\test.csv', delimiter=',', header=0)
X_train = train.iloc[:, 1:]
y_train = train.iloc[:, 0]
X_test = test.iloc[:, 1:]
y_test = test.iloc[:, 0]

features = list(X_train)
#print(features)

initialize(config_path="..\conf", job_name="config")
cfg = compose(config_name="config")
#cfg = omegaconf.OmegaConf.load(os.path.join(os.getcwd(), "src\conf\config.yaml")) 
#print(cfg)
#variables = cfg['transform']['vars']
#print(variables)

def track_features(cfg: data_config):

    d = {} 
    for i in cfg.transform.vars:
        list_transforms = []
        for j in features:
            d[i] = {}
            transform = re.search(rf"(?<={i}_).*?(?=_lag)", j)  # extract the feature engineering of each feature
            if transform:
                transform = transform.group(0)
                list_transforms.append(transform)
        list_transforms = sorted(list(set(list_transforms)))
        for k in list_transforms:
            d[i][k] = []
            for l in cfg.diff.lags:
                d[i][k].append('lag_' + str(l))
    
    return d

d = track_features(cfg = cfg)

print(d)







#####################################
# For the lags use config.yaml! 
# depending if [] or preds are given -> cfg.model.predictors

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













