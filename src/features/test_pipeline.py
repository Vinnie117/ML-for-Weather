import pandas as pd
from sklearn.pipeline import Pipeline
from pipeline_classes import Velocity
from pipeline_classes import InsertLags
from pipeline_classes import Debugger
from pipeline_classes import Times
from sklearn.model_selection import train_test_split
from functions import clean

# load data
df_raw = pd.read_csv(r'A:\Projects\ML-for-Weather\data\raw\test_simple.csv') 
df = clean(df_raw)

# Splitting here to prevent information leakage
train, test = train_test_split(df, test_size=0.2, shuffle = False)

# Feature engineering
pipe = Pipeline([
    ("times", Times()),
    ("debug3", Debugger()),
    ("lags", InsertLags(['temperature', 'cloud_cover', 'wind_speed'], lags=[1,2,24])),
    ("debug4", Debugger()),
    ('velocity', Velocity(['temperature', 'cloud_cover', 'wind_speed'], diff=[1,2])),   # lagged differences? differenced lags? -> the same
    ("debug5", Debugger()),
    ('lagged_velocity', InsertLags(['wind_speed_velo_1'], [1,2])),   
    ("debug6", Debugger())       
])

data = pipe.fit_transform(df) 

# To do:
# - add acceleration ()
# - add lagged difference (x)
# - add lagged acceleration ()
# - def clean() -> Column names nicht hart verdrahten sondern als Argumente der Funktion (2 Listen für alte und neue Namen) ()
# - scenario for more or less than the three variables ()


data.to_csv(r'A:\Projects\ML-for-Weather\data\processed\df.csv', header=True, index=False)
print("END")