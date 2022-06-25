######## Exploratory Data Analysis ########
import sys 
sys.path.append('A:\Projects\ML-for-Weather\src')  # import from parent directory
import matplotlib.pyplot as plt
from features.pipeline_dataprep import pd_df, train, test
import pandas as pd
from config import data_config
from hydra import compose


cfg = compose(config_name="config")


################################################################

def lag_analysis(var, target, rows, cols):

    # create dict with data for lag analysis
    lags = {}
    for i in cfg['model']['predictors']:
        if str(var) + '_lag_' in i:
            lags[i] = train[i]

    y = train[target]


    # Build plots for lags
    fig, axs = plt.subplots(rows, cols, sharex=True, sharey=True, figsize=(9,9))
    fig.suptitle('Temperature - Lag Analysis')

    axs = axs.flatten()    # axs is 2d array (rows and columns), flatten it to loop over
    corrs = []
    for i, j in enumerate(lags):

        # calculate correlation
        corrs.append(round(y.corr(lags[j]), 4))

        axs[i].scatter(lags[j], y, s=1)
        axs[i].set(xlabel=list(lags)[i], ylabel=target)
        axs[i].annotate('r = {}'.format(corrs[i]), xy = (0,1), xytext = (0.05,0.9), xycoords = 'axes fraction')
    

    # barplot of correlations
    df = pd.DataFrame(list(zip(list(lags), corrs)), columns =['lag', 'corr'])
    df = df.sort_values('corr')
    plt.figure(figsize=(12, 5))
    bar = plt.barh(df['lag'], df['corr'])
    plt.title('Correlations with temperature')
    plt.bar_label(bar)
    
    return fig, bar

lag_analysis(var = 'temperature', target = 'temperature', rows = 3, cols = 3)


# Lag_24 seems good -> try 25 and 26 -> might be even better than 6 or 9 though it is further away

print('END')