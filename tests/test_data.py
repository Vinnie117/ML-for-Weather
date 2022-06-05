######################## Test validity of data with Pytest ########################
import sys
sys.path.append('A:\Projects\ML-for-Weather\src') 
from features.pipeline_dataprep import pd_df, train, test
import numpy as np
import pandas as pd




#revert_transform = pd_df['temperature_lag_1'].shift(-1)[:-1]  #[:-1] drops the last row
#original = pd_df['temperature'][:-1]
transformed = pd_df['temperature_velo_1'][1:]
original_transformed = pd_df['temperature'].diff(1)[1:]

transformed.fillna(original_transformed, inplace=True)

#revert_transform.fillna(original, inplace=True)
# print(revert_transform)
# print(original)

# Test schlägt fehl
print(transformed.equals(original_transformed))
print(transformed.compare(original_transformed))




# weil pd_df ein gemergetes df ist, tauchen in der Mitte des df NAs auf
# -> von den Test-Daten der Anfang, wo gelaggt wird
# -> für das Training ist droppen ok, weil viele Daten
# -> Die NAs müssen für das pytest gefüllt werden


###########################################
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