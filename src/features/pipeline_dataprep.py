from omegaconf import DictConfig
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
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
import os 

# dir_path = os.path.dirname(os.path.realpath(__file__))
# print(dir_path)

# cwd = os.getcwd()
# print(cwd)


# load data
df_raw = pd.read_csv(r'A:\Projects\ML-for-Weather\data\raw\test_simple.csv') 

@hydra.main(config_path='..\conf', config_name='config')
def data_handler(cfg: DictConfig) -> None:

    print(cfg)
    return

    # # indicate var names to be changed
    # df = clean(df_raw, old = ['temperature_air_mean_200', 
    #                         'cloud_cover_total'],
    #                 new = ['temperature',
    #                         'cloud_cover'])
    # return df


####################################################

if __name__ == "__main__":
    data_handler()


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

