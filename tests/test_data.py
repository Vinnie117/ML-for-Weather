import sys 
sys.path.append('A:\Projects\ML-for-Weather\src') 
from features.pipeline_dataprep import pd_df
import numpy as np
import pytest

print(pd_df)
print(type(pd_df))

print('end')

# Test variables 
np.testing.assert_array_equal([1.0,2.33333,np.nan],
                              [np.exp(0),2.33333, np.nan],
                              err_msg='Fehler')


def inc(x):
    return x + 1


def test_answer():
    assert inc(3) == 5

print('end')