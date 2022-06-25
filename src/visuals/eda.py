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

def lag_analysis(var, target, rows, cols):

    # create dict with data for lag analysis
    lags = {}
    for i in cfg['model']['predictors']:
        if str(var) + '_lag_' in i:
            lags[i] = train[i]

    y = train[target]
    fig, axs = plt.subplots(rows, cols, sharex=True, sharey=True, figsize=(9,9))
    fig.suptitle('Temperature - Lag Analysis')

    # axs is 2d array (rows and columns), flatten it to loop over
    axs = axs.flatten()

    # build subplots
    for i, j in enumerate(lags):
        axs[i].scatter(lags[j], y, s=1)
        axs[i].set(xlabel=list(lags)[i], ylabel=target)
    
    return fig

lag_analysis(var = 'temperature', target = 'temperature', rows = 3, cols = 3)







fig.show()

# Lag_24 seems good -> try 25 and 26 -> might be even better than 6 or 9 though it is further away

print('END')