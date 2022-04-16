import sys
sys.path.append('A:\Projects\ML-for-Weather\src')  # import from parent directory
from omegaconf import OmegaConf
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
from config import data_config
from pipeline_dataprep_classes import Prepare
from pipeline_dataprep_classes import Acceleration
from pipeline_dataprep_classes import Velocity
from pipeline_dataprep_classes import InsertLags
from pipeline_dataprep_classes import Debugger
from pipeline_dataprep_classes import Times
from pipeline_dataprep_classes import Split
#from sklearn.model_selection import train_test_split
#from functions import clean
#import hydra
from hydra.core.config_store import ConfigStore
from hydra import compose, initialize
#import os



# load data
#df_raw = pd.read_csv(r'A:\Projects\ML-for-Weather\data\raw\test_simple.csv') 

###########################################################
# Mark data_handler with hydra.main()
# Use instance of config dataclass
# cs = ConfigStore.instance()
# cs.store(name = 'data_config', node = data_config)

# @hydra.main(config_path='..\conf', config_name='config')
# def data_handler(cfg: data_config):
    
#     # load data
#     df_raw = pd.read_csv(cfg.data.path)
    
#     # clean up and prepare
#     data = df_raw.drop(columns=["station_id", "dataset"])
#     data = data.pivot(index="date", columns="parameter", values="value").reset_index()
    
#     # renaming
#     for i in cfg.vars_old:
#         data = data.rename(columns={cfg.vars_old[i]: cfg.vars_new[i]})

#     # ordering
#     data.insert(1, cfg.vars_new.temp, data.pop(cfg.vars_new.temp))
#     print(data)

#     return data

# This one works but does not return values (intended by hydra!)
# https://www.google.de/search?q=python+hydra+config+in+function+return+value&ei=qJxZYrbvBOOVxc8PjOu9-AE&oq=python+hydra+config+in+function+return&gs_lcp=Cgdnd3Mtd2l6EAMYADIFCCEQoAEyBQghEKABOgQIABBHOgcIIRAKEKABSgQIQRgASgUIQBIBMUoECEYYAFC-B1ilDGCeF2gAcAJ4AIABsAGIAfEFkgEDMC42mAEAoAEByAEIwAEB&sclient=gws-wiz

############################################################
# Does not use hydra
# basic configs with yaml
# conf = OmegaConf.load('src\conf\config.yaml')

# def data_handler():
    
#     # load data
#     df_raw = pd.read_csv(conf.data.path)
    
#     # clean up and prepare
#     data = df_raw.drop(columns=["station_id", "dataset"])
#     data = data.pivot(index="date", columns="parameter", values="value").reset_index()
    
#     # renaming
#     for i in conf.vars_old:
#         data = data.rename(columns={conf.vars_old[i]: conf.vars_new[i]})

#     # ordering
#     data.insert(1, conf.vars_new.temp, data.pop(conf.vars_new.temp))
#     #print(data)

#     return data
################################################################

# Use Compose API of hydra 
initialize(config_path="..\conf", job_name="config")
cfg = compose(config_name="config")
print(OmegaConf.to_yaml(cfg))

# Use instance of config dataclass
cs = ConfigStore.instance()
cs.store(name = 'data_config', node = data_config)

def data_loader(cfg: data_config):
    
    # load data
    df_raw = pd.read_csv(cfg.data.path)
    
    # clean up and prepare
    data = df_raw.drop(columns=['station_id', 'dataset'])
    data = data.pivot(index='date', columns='parameter', values='value').reset_index()
    
    # renaming
    for i in cfg.vars_old:
        data = data.rename(columns={cfg.vars_old[i]: cfg.vars_new[i]})
    
    # ordering
    data.insert(1, cfg.vars_new.temp, data.pop(cfg.vars_new.temp))

    return data
##################



df2 = data_loader(cfg=cfg)
print(df2)

####################################################

# part of hydra-main()
# if __name__ == "__main__":
#     data_handler()


#####################################################
# Implement later:

# Feature engineering

def feature_engineering(cfg: data_config):

    pipe = Pipeline([
        ("split", Split(test_size=0.2, shuffle = False)), # -> sklearn.model_selection.TimeSeriesSplit
        ("times", Times()),
        ("lags", InsertLags(cfg.insert_lags.vars, lags=[1,2,24])),
        ('debug', Debugger()),
        ('velocity', Velocity(['temperature', 'cloud_cover', 'wind_speed'], diff=[1,2])),   
        ('lagged_velocity', InsertLags(['temperature_velo_1', 'cloud_cover_velo_1', 'wind_speed_velo_1'], [1,2])),     # lagged difference = differenced lag
        ('acceleration', Acceleration(['temperature', 'cloud_cover', 'wind_speed'], diff=[1])),                        # diff of 1 day between 2 velos
        ('lagged_acceleration', InsertLags(['temperature_acc_1', 'cloud_cover_acc_1', 'wind_speed_acc_1'], [1,2])),   
        ('cleanup', Prepare(target = ['temperature'],
                            vars=['month', 'day', 'hour', 'temperature_lag_1', 'cloud_cover_lag_1', 'wind_speed_lag_1']))
        ])

    return pipe

pipeline = feature_engineering(cfg = cfg)

data = pipeline.fit_transform(df2) 

train = data['train']
test = data['test']

print(test)

# np.savetxt(r'A:\Projects\ML-for-Weather\data\processed\train_array.csv', train, delimiter=",", fmt='%s')
# np.savetxt(r'A:\Projects\ML-for-Weather\data\processed\test_array.csv', test, delimiter=",", fmt='%s')

