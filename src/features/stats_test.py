######## Test  data for statistical properties ########
from statsmodels.tsa.stattools import adfuller
import pandas as pd
import numpy as np
from pipeline_dataprep import pd_df


#### Test for stationarity with Augmented Dickey Fuller Test
# need stationary time series: we want to reject H0 of ADF test
adf_cloud_cover_total = adfuller(pd_df["temperature"])
print('ADF Statistic: %f' % adf_cloud_cover_total[0])
print('p-value: %f' % adf_cloud_cover_total[1])

adf_cloud_cover_total = adfuller(pd_df["cloud_cover"])
print('ADF Statistic: %f' % adf_cloud_cover_total[0])
print('p-value: %f' % adf_cloud_cover_total[1])


adf_cloud_cover_total = adfuller(pd_df["wind_speed"])
print('ADF Statistic: %f' % adf_cloud_cover_total[0])
print('p-value: %f' % adf_cloud_cover_total[1])


