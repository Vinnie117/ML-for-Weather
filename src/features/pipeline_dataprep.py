import pandas as pd
from sklearn.pipeline import Pipeline
from pipeline_dataprep_classes import Acceleration
from pipeline_dataprep_classes import Velocity
from pipeline_dataprep_classes import InsertLags
from pipeline_dataprep_classes import Debugger
from pipeline_dataprep_classes import Times
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
# - add acceleration -> difference of velocity or difference normal values twice (x) 
#   - https://stackoverflow.com/questions/54505175/calculating-second-order-derivative-from-timeseries-using-pandas-diff
# - add lagged difference (x)
# - add lagged acceleration (x)
# - def clean() -> Column names nicht hart verdrahten sondern als Argumente der Funktion (2 Listen für alte und neue Namen) (x)
# - scenario for more or less than the three variables (x)



data.to_csv(r'A:\Projects\ML-for-Weather\data\processed\df.csv', header=True, index=False)
print("END")