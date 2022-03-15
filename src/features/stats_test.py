######## Test  data for statistical properties ########
from statsmodels.tsa.stattools import adfuller
import pandas as pd

df = pd.read_csv(r'A:\Projects\ML-for-Weather\data\interim\df.csv') 

#### Test for stationarity with Augmented Dickey Fuller Test
# need stationary time series: we want to reject H0 of ADF test
adf_cloud_cover_total = adfuller(df["cloud_cover_total"])
print('ADF Statistic: %f' % adf_cloud_cover_total[0])
print('p-value: %f' % adf_cloud_cover_total[1])


adf_cloud_cover_total_AIC = adfuller(df["cloud_cover_total"], autolag='AIC')
print('ADF Statistic: %f' % adf_cloud_cover_total_AIC[0])
print('p-value: %f' % adf_cloud_cover_total_AIC[1])







