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


fig, axs = plt.subplots(3, 3, figsize=(9,9))
fig.suptitle('Lag Analysis')
axs[0, 0].scatter(temperature_lags['temperature_lag_1'], y, s=1)
axs[0, 1].scatter(temperature_lags['temperature_lag_2'], y, s=1)
axs[0, 2].scatter(temperature_lags['temperature_lag_3'], y, s=1)
axs[1, 0].scatter(temperature_lags['temperature_lag_4'], y, s=1)
axs[1, 1].scatter(temperature_lags['temperature_lag_5'], y, s=1)
axs[1, 2].scatter(temperature_lags['temperature_lag_6'], y, s=1)
axs[2, 0].scatter(temperature_lags['temperature_lag_9'], y, s=1)
axs[2, 1].scatter(temperature_lags['temperature_lag_12'], y, s=1)
axs[2, 2].scatter(temperature_lags['temperature_lag_24'], y, s=1)

# Loop through this!

# for i, ax in enumerate(axs.flat):
#     print(i)
#     ax.set(xlabel='A', ylabel='temperature')

for i, ax in zip(range(len(temperature_lags)), axs.flat):
    ax.set(xlabel=list(temperature_lags)[i], ylabel='temperature')

#plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

fig.show()

print('END')