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

# load data
df_raw = pd.read_csv(r'A:\Projects\ML-for-Weather\data\raw\test_simple.csv') 
# indicate var names to be changed
df = clean(df_raw, old = ['temperature_air_mean_200', 
                          'cloud_cover_total'],
                   new = ['temperature',
                          'cloud_cover'])


# Feature engineering
pipe = Pipeline([
    ("split", Split(test_size=0.2, shuffle = False)), # -> sklearn.model_selection.TimeSeriesSplit
    ("times", Times()),
    ("lags", InsertLags(['temperature', 'cloud_cover', 'wind_speed'], lags=[1,2,24])),
    ('velocity', Velocity(['temperature', 'cloud_cover', 'wind_speed'], diff=[1,2])),   
    ('lagged_velocity', InsertLags(['temperature_velo_1', 'cloud_cover_velo_1', 'wind_speed_velo_1'], [1,2])),     # lagged difference = differenced lag
    ('acceleration', Acceleration(['temperature', 'cloud_cover', 'wind_speed'], diff=[1])),                        # diff of 1 day between 2 velos
    ('lagged_acceleration', InsertLags(['temperature_acc_1', 'cloud_cover_acc_1', 'wind_speed_acc_1'], [1,2])),   
#    ('cleanup', Prepare(target = ['temperature'],
#                        vars=['month', 'day', 'hour', 'temperature_lag_1', 'cloud_cover_lag_1', 'wind_speed_lag_1'])),
    ("debug8", Debugger())
])

data = pipe.fit_transform(df) 

train = data['train']
test = data['test']

print(train)



train.to_csv(r'A:\Projects\ML-for-Weather\data\processed\train.csv', header=True, index=False)
test.to_csv(r'A:\Projects\ML-for-Weather\data\processed\test.csv', header=True, index=False)

# np.savetxt(r'A:\Projects\ML-for-Weather\data\processed\train_array.csv', train, delimiter=",", fmt='%s')
# np.savetxt(r'A:\Projects\ML-for-Weather\data\processed\test_array.csv', test, delimiter=",", fmt='%s')

