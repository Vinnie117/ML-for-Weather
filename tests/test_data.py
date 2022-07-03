######################## Test validity of data with Pytest ########################
import sys
sys.path.append('A:\Projects\ML-for-Weather\src') 
from features.pipeline_dataprep import pd_df
import numpy as np
import pandas as pd

###################################################################################
# Manual testing

#revert_transform = pd_df['temperature_lag_1'].shift(-1)[:-1]  #[:-1] drops the last row
#original = pd_df['temperature'][:-1]
transformed = pd_df['temperature_velo_1'][1:]
original_transformed = pd_df['temperature'].diff(1)[1:]
transformed.fillna(original_transformed, inplace=True)

# Test
print(transformed.equals(original_transformed))
print(transformed.compare(original_transformed))

# pd_df is a merged df (train + test) -> has NAs at train/test split point
# - NAs are from the start of test data, where lags are inserted
# - NAs are to be filled for the pytest
# In general: dropping rows with NAs is ok for train/test -> lots of data!

###################################################################################
# https://stackoverflow.com/questions/53830081/python-pandas-the-truth-value-of-a-series-is-ambiguous

def test_temperature_lag_1():
    
    # revert the lag and delete resulting NaN from last row
    revert_transform = pd_df['temperature_lag_1'].shift(-1)[:-1]
    original = pd_df['temperature'][:-1]
    revert_transform.fillna(original, inplace=True)
    assert (original == revert_transform).all()


def test_temperature_lag_2():
    
    revert_transform = pd_df['temperature_lag_2'].shift(-2)[:-2]
    original = pd_df['temperature'][:-2]
    revert_transform.fillna(original, inplace=True)
    assert (original == revert_transform).all()


def test_temperature_lag_24():
    
    revert_transform = pd_df['temperature_lag_24'].shift(-24)[:-24]
    original = pd_df['temperature'][:-24]
    revert_transform.fillna(original, inplace=True)
    assert (original == revert_transform).all()


def test_temperature_velo_1():

    transformed = pd_df['temperature_velo_1'][1:]
    original_transformed = pd_df['temperature'].diff(1)[1:]
    transformed.fillna(original_transformed, inplace=True)
    assert (original_transformed == transformed).all()


def test_temperature_velo_1_lag_1():

    transformed = pd_df['temperature_velo_1_lag_1'][2:]
    original_transformed = pd_df['temperature'].diff(1).shift(1)[2:]
    transformed.fillna(original_transformed, inplace=True)
    assert (original_transformed == transformed).all()