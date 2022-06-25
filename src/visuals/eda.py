######## Exploratory Data Analysis ########
import sys 
sys.path.append('A:\Projects\ML-for-Weather\src')  # import from parent directory
import matplotlib.pyplot as plt
from features.pipeline_dataprep import pd_df, train, test
from config import data_config
from hydra import compose


#print(data_config)
cfg = compose(config_name="config")
print(cfg['model']['predictors'])

################################################################

def lag_analysis(var, target):

    print(var)
    dict_name = '{}_lags'.format(var)
    print(dict_name)

    lags = {}
    for i in cfg['model']['predictors']:
        if str(var) + '_lag_' in i:
            lags[i] = train[i]

    y = train[target]
    rows = 3
    cols = 3
    fig, axs = plt.subplots(rows, cols, sharex=True, sharey=True, figsize=(9,9))

    fig.suptitle('Temperature - Lag Analysis')

    # axs is 2d array (rows and columns), flatten it to loop over
    axs = axs.flatten()

    for i, j in enumerate(lags):
        axs[i].scatter(lags[j], y, s=1)
        axs[i].set(xlabel=list(lags)[i], ylabel='temperature')
    
    return fig

lag_analysis(var = 'cloud_cover', target = 'temperature')


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

cloud_cover_lags = {}
cloud_cover_lags['cloud_cover_lag_1'] = train['cloud_cover_lag_1']
cloud_cover_lags['cloud_cover_lag_2'] = train['cloud_cover_lag_2']
cloud_cover_lags['cloud_cover_lag_3'] = train['cloud_cover_lag_3']
cloud_cover_lags['cloud_cover_lag_4'] = train['cloud_cover_lag_4']
cloud_cover_lags['cloud_cover_lag_5'] = train['cloud_cover_lag_5']
cloud_cover_lags['cloud_cover_lag_6'] = train['cloud_cover_lag_6']
cloud_cover_lags['cloud_cover_lag_9'] = train['cloud_cover_lag_9']
cloud_cover_lags['cloud_cover_lag_12'] = train['cloud_cover_lag_12']
cloud_cover_lags['cloud_cover_lag_24'] = train['cloud_cover_lag_24']


y = train['temperature']

rows = 3
cols = 3

# fig, axs = plt.subplots(rows, cols, sharex=True, sharey=True, figsize=(9,9))
# fig.suptitle('Temperature Lag Analysis')

# # axs is 2d array (rows and columns), flatten it to loop over
# axs = axs.flatten()

# for i, j in enumerate(temperature_lags):
#     axs[i].scatter(temperature_lags[j], y, s=1)
#     axs[i].set(xlabel=list(temperature_lags)[i], ylabel='temperature')


fig, axs = plt.subplots(rows, cols, sharex=True, sharey=True, figsize=(9,9))
fig.suptitle('Cloud Cover - Lag Analysis')

# axs is 2d array (rows and columns), flatten it to loop over
axs = axs.flatten()

for i, j in enumerate(cloud_cover_lags):
    axs[i].scatter(cloud_cover_lags[j], y, s=1)
    axs[i].set(xlabel=list(cloud_cover_lags)[i], ylabel='temperature')




fig.show()

# Lag_24 seems good -> try 25 and 26 -> might be even better than 6 or 9 though it is further away

print('END')