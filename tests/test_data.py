######################## Test validity of data with Pytest ########################

import sys
from matplotlib import transforms

from pandas import NA 
sys.path.append('A:\Projects\ML-for-Weather\src') 
from features.pipeline_dataprep import pd_df
import numpy as np
import pandas as pd

revert_transform = pd_df['temperature_lag_1'].shift(-1)[:-1]
#revert_transform.columns = ['temperature']
#print(revert_transform)
#print(pd_df['temperature'][:-1])
#print(type(revert_transform))
#print(type(pd_df['temperature'][:-1]))

revert_transform = pd_df['temperature_velo_1'].shift(-1)[:-1]
original = pd_df['temperature']
original_1 = pd_df['temperature'].diff(1)[1:]

transform = pd_df['temperature_velo_1'][1:]

print(original)
print(original_1)
print(transform)


###########################################
# https://stackoverflow.com/questions/53830081/python-pandas-the-truth-value-of-a-series-is-ambiguous

# -> testen, indem falsche Werte provoziert werden!

def test_temperature_lag_1():
    
    # revert the lag and delete resulting NaN
    revert_transform = pd_df['temperature_lag_1'].shift(-1)[:-1]
    original = pd_df['temperature'][:-1]
    assert (original == revert_transform).all()


def test_temperature_lag_2():
    
    # revert the lag and delete resulting NaN
    revert_transform = pd_df['temperature_lag_2'].shift(-2)[:-2]
    original = pd_df['temperature'][:-2]
    assert (original == revert_transform).all()


def test_temperature_lag_24():
    
    # revert the lag and delete resulting NaN
    revert_transform = pd_df['temperature_lag_24'].shift(-24)[:-24]
    original = pd_df['temperature'][:-24]
    assert (original == revert_transform).all()


def test_temperature_velo_1():

    transformed = pd_df['temperature_velo_1'][1:]
    original_transformed = pd_df['temperature'].diff(1)[1:]
    assert (original_transformed == transformed).all()