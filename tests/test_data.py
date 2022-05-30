######################## Test validity of data with Pytest ########################

import sys

from pandas import NA 
sys.path.append('A:\Projects\ML-for-Weather\src') 
from features.pipeline_dataprep import pd_df
import numpy as np
import pandas as pd


# data = {'temperature_lag_1':[pd_df['temperature_lag_1'].shift(-1)]}
# df = pd.DataFrame(data)
# pd.concat([data['train'], X], axis=1)


revert_transform = pd_df['temperature_lag_1'].shift(-1)[:-1]
revert_transform.columns = ['temperature']
print(revert_transform)
print(pd_df['temperature'][:-1])
print(type(revert_transform))
print(type(pd_df['temperature'][:-1]))

# Unterschiedliche Zeilenlängen!


##########################################
# Minimal test example

def inc(x):
    return x + 1

def test_answer_right():
    assert inc(4) == 5

def test_answer_wrong():
    assert inc(3) == 5
###########################################

def test_temperature_lag_1():
    
    revert_transform = pd_df['temperature_lag_1'].shift(-1)[:-1]
    #revert_transform.columns = ['temperature']

    original = pd_df['temperature'][:-1]
    #original.columns = ['temperature']
    
    assert (original == revert_transform).all()

print('end')