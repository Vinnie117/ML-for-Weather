######## Test  data for statistical properties ########
from statsmodels.tsa.stattools import adfuller
import pandas as pd
import numpy as np
from pipeline_dataprep import pd_df
from hydra.core.config_store import ConfigStore
from hydra import compose, initialize
from omegaconf import OmegaConf
from config import data_config

#### Test for stationarity with Augmented Dickey Fuller Test
# need stationary time series: we want to reject H0 of ADF test
adf_cloud_cover_total = adfuller(pd_df["temperature"])
print('temperature ADF Statistic: %f' % adf_cloud_cover_total[0])
print('p-value: %f' % adf_cloud_cover_total[1])

adf_cloud_cover_total = adfuller(pd_df["cloud_cover"])
print('cloud_cover ADF Statistic: %f' % adf_cloud_cover_total[0])
print('p-value: %f' % adf_cloud_cover_total[1])


adf_cloud_cover_total = adfuller(pd_df["wind_speed"])
print('wind_speed ADF Statistic: %f' % adf_cloud_cover_total[0])
print('p-value: %f' % adf_cloud_cover_total[1])

#### Do all stationary tests at once
cfg = compose(config_name="config")


dict_stationarity_tests = {}
statistic = ['ADF Statistic', 'p-value']

def stationarity_tests(cfg: data_config):
    '''
    Build a nested dictionary with stationary test results for each variable
    '''

    for i in cfg.transform.vars:
        dict_stationarity_tests[i] = {}
        for j in statistic:
            adf_test = adfuller(pd_df[i])
            #print(adf_test[statistic.index(j)])
            dict_stationarity_tests[i][j] = adf_test[statistic.index(j)]

    return dict_stationarity_tests

dc = stationarity_tests(cfg = cfg)
print(dc)

print('END')


