######################## Test validity of data with Pytest ########################
import sys
sys.path.append('A:\Projects\ML-for-Weather\src') 
from features.pipeline_dataprep import pd_df, train, test
import numpy as np
import pandas as pd



revert_transform = train['temperature_lag_1'].shift(-1)[:-1]
original = train['temperature'][:-1]

print(revert_transform)
print(original)

# Test schlägt fehl
print(revert_transform.equals(original))
print(revert_transform.compare(original))

#show specific rows: 7007, 7253, 7277
print(original.iloc[7007])
print(revert_transform.iloc[7007])    # -> should be the same!


# fehlgeschlagene Tests kommen durch die Lags zustande
# -> wenn train und test data gemerged werden
#   -> wenn die NAs gedroppt werden
# Lösung: dict_data['pd_df']  ganz am Ende erzeugen? -> Nein
# Frage: Wo ist Reihe 7008? -> gedroppt wegen NA 
#   -> einzelne Reihen droppen ok, weil so viele Daten"

# für train klappen die Tests!
# Aber bei test schlägt der Test auch fehl!

# -> separate tests für test und train, aber in einer Testfunktion für die gleiche Variable!



###########################################
# https://stackoverflow.com/questions/53830081/python-pandas-the-truth-value-of-a-series-is-ambiguous

def test_temperature_lag_1():
    
    # revert the lag and delete resulting NaN
    revert_transform = train['temperature_lag_1'].shift(-1)[:-1]
    original = train['temperature'][:-1]
    assert (original == revert_transform).all()


def test_temperature_lag_2():
    
    revert_transform = pd_df['temperature_lag_2'].shift(-2)[:-2]
    original = pd_df['temperature'][:-2]
    assert (original == revert_transform).all()


def test_temperature_lag_24():
    
    revert_transform = pd_df['temperature_lag_24'].shift(-24)[:-24]
    original = pd_df['temperature'][:-24]
    assert (original == revert_transform).all()


def test_temperature_velo_1():

    transformed = pd_df['temperature_velo_1'][1:]
    original_transformed = pd_df['temperature'].diff(1)[1:]
    assert (original_transformed == transformed).all()


def test_temperature_velo_1_lag_1():

    transformed = pd_df['temperature_velo_1_lag_1'][2:]
    original_transformed = pd_df['temperature'].diff(1).shift(1)[2:]
    assert (original_transformed == transformed).all()