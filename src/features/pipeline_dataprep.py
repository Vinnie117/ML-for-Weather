import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
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


# Splitting here to prevent information leakage
train, test = train_test_split(df, test_size=0.2, shuffle = False)

# Feature engineering
pipe = Pipeline([
    ("split", Split(test_size=0.2, shuffle = False)),
    ("times", Times()),
    ("lags", InsertLags(['temperature', 'cloud_cover', 'wind_speed'], lags=[1,2,24])),
    ('velocity', Velocity(['temperature', 'cloud_cover', 'wind_speed'], diff=[1,2])),   
    ('lagged_velocity', InsertLags(['temperature_velo_1', 'cloud_cover_velo_1', 'wind_speed_velo_1'], [1,2])),    # lagged difference = differenced lag!
    ('acceleration', Acceleration(['temperature', 'cloud_cover', 'wind_speed'], diff=[1])),
    ('lagged_acceleration', InsertLags(['temperature_acc_1', 'cloud_cover_acc_1', 'wind_speed_acc_1'], [1,2])),   
    ("debug8", Debugger())
])

data = pipe.fit_transform(df) 

# To do:
# Split data into train and test in pipeline
#    ls = pipe.fit_transform(df)
#    train = ls[0]
#    test = ls[1]


data.to_csv(r'A:\Projects\ML-for-Weather\data\processed\df.csv', header=True, index=False)
print("END")