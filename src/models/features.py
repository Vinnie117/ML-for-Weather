######## Extract features to track them with mlflow ########

import pandas as pd
import omegaconf
import os
import re

# get data
train = pd.read_csv(r'A:\Projects\ML-for-Weather\data\processed\train.csv', delimiter=',', header=0)
test = pd.read_csv(r'A:\Projects\ML-for-Weather\data\processed\test.csv', delimiter=',', header=0)
X_train = train.iloc[:, 1:]
y_train = train.iloc[:, 0]
X_test = test.iloc[:, 1:]
y_test = test.iloc[:, 0]

features = list(X_train)

cfg = omegaconf.OmegaConf.load(os.path.join(os.getcwd(), "src\conf\config.yaml")) 
print(features)


print(cfg)
variables = cfg['transform']['vars']
print(variables)

dict = dict.fromkeys(variables)
print(dict)

list_temperature_transforms = []
list_cloud_cover_transforms = []
list_wind_speed_transforms = []
dummy = []
for i in features:
    if 'temperature' in i:
        if 'velo' in i:
            dummy.append(i)

            # need to include delimiter
            # test = i.split('lag')[1]
            test = re.split('(\W)',i)
            print(test)

            transform = re.search(r"(?<=temperature_).*?(?=_lag)", i).group(0)
            list_temperature_transforms.append(transform)
            list_unique_temperature_transforms = list(set(list_temperature_transforms))
print('dummy is: ', dummy)
print(list_temperature_transforms)
print(list(set(list_temperature_transforms)))


# a) extract substring between two markers "temperature_" and "_lag"
# or b) extract substring "velo" and the following 2 characters -> a) is better


# -> in config.yaml: für acc und velo transform.names erstellen? 
# Und dann auch in pipeline_dataprep.classes anpassen
# Dann im for loop auf velo und acc in der config.yaml verweisen -> "... if ... in ..."


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













