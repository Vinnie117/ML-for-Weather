######## Exploratory Data Analysis ########
import sys 
sys.path.append('A:\Projects\ML-for-Weather\src')  # import from parent directory
import matplotlib.pyplot as plt
from features.pipeline_dataprep import pd_df, train, test
#from models.naive_model import y_test


################################################################
temperature_lags = {}
temperature_lags['temperature_lag_1'] = train['temperature_lag_1']
temperature_lags['temperature_lag_2'] = train['temperature_lag_2']
temperature_lags['temperature_lag_3'] = train['temperature_lag_3']
temperature_lags['temperature_lag_4'] = train['temperature_lag_4']
temperature_lags['temperature_lag_5'] = train['temperature_lag_5']
temperature_lags['temperature_lag_6'] = train['temperature_lag_6']
temperature_lags['temperature_lag_9'] = train['temperature_lag_9']
temperature_lags['temperature_lag_12'] = train['temperature_lag_12']
temperature_lags['temperature_lag_24'] = train['temperature_lag_24']


y = train['temperature']

rows = 3
cols = 3

fig, axs = plt.subplots(rows, cols, sharex=True, sharey=True, figsize=(9,9))
fig.suptitle('Lag Analysis')

# axs is 2d array (rows and columns), flatten it to loop over
axs = axs.flatten()

for i, j in enumerate(temperature_lags):
    axs[i].scatter(temperature_lags[j], y, s=1)
    axs[i].set(xlabel=list(temperature_lags)[i], ylabel='temperature')







fig.show()

print('END')