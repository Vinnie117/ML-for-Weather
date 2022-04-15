import sys
sys.path.append('A:\Projects\ML-for-Weather\src')  # import from parent directory
import sys
from omegaconf import DictConfig
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
from sklearn.model_selection import train_test_split
from functions import clean
import hydra
from hydra.core.config_store import ConfigStore
import os



# load data
#df_raw = pd.read_csv(r'A:\Projects\ML-for-Weather\data\raw\test_simple.csv') 

# Use instance of config dataclass
cs = ConfigStore.instance()
cs.store(name = 'data_config', node = data_config)

@hydra.main(config_path='..\conf', config_name='config')
def data_handler(cfg: data_config) -> None:
    
    #print(cfg)
    df_raw = pd.read_csv(cfg.data.path)
    
    data = df_raw.drop(columns=["station_id", "dataset"])
    data = data.pivot(index="date", columns="parameter", values="value").reset_index()
    
    for i in cfg.vars_old:
        data = data.rename(columns={cfg.vars_old[i]: cfg.vars_new[i]})

    print(data)
    # data.insert(1, 'temperature', data.pop('temperature'))


    # indicate var names to be changed
    # df = clean(df_raw, old = [cfg.vars_old.temp, 
    #                           cfg.vars_old.cloud],
    #                 new = [cfg.vars_new.temp,
    #                        cfg.vars_new.cloud])
    return data


df = data_handler()
print(df)

####################################################

# if __name__ == "__main__":
#     data_handler()


# # Feature engineering
# pipe = Pipeline([
#     ("split", Split(test_size=0.2, shuffle = False)), # -> sklearn.model_selection.TimeSeriesSplit
#     ("times", Times()),
#     ("lags", InsertLags(['temperature', 'cloud_cover', 'wind_speed'], lags=[1,2,24])),
#     ('velocity', Velocity(['temperature', 'cloud_cover', 'wind_speed'], diff=[1,2])),   
#     ('lagged_velocity', InsertLags(['temperature_velo_1', 'cloud_cover_velo_1', 'wind_speed_velo_1'], [1,2])),     # lagged difference = differenced lag
#     ('acceleration', Acceleration(['temperature', 'cloud_cover', 'wind_speed'], diff=[1])),                        # diff of 1 day between 2 velos
#     ('lagged_acceleration', InsertLags(['temperature_acc_1', 'cloud_cover_acc_1', 'wind_speed_acc_1'], [1,2])),   
#     ('cleanup', Prepare(target = ['temperature'],
#                         vars=['month', 'day', 'hour', 'temperature_lag_1', 'cloud_cover_lag_1', 'wind_speed_lag_1']))
#     ])

# data = pipe.fit_transform(df) 

# train = data['train']
# test = data['test']

# #print(test)

# np.savetxt(r'A:\Projects\ML-for-Weather\data\processed\train_array.csv', train, delimiter=",", fmt='%s')
# np.savetxt(r'A:\Projects\ML-for-Weather\data\processed\test_array.csv', test, delimiter=",", fmt='%s')

