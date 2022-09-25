######## Exploratory Data Analysis ########
import sys 
import os
sys.path.append('A:\Projects\ML-for-Weather\src')  # import from parent directory
import matplotlib.pyplot as plt
from features.pipeline_dataprep import pd_df, train, test, cfg
import pandas as pd

# from hydra import compose
# cfg = compose(config_name="config")

# import omegaconf
# cfg = omegaconf.OmegaConf.load(os.path.join(os.getcwd(), "src\conf\config.yaml")) 

################################################################

def lag_analysis(var, target, rows, cols):
    ''' EDA function to explore relationship of an underlying with its lags

    - config.yaml -> cfg.model.predictors set? Must not be empty!
    - config.yaml -> cfg.model.target set? Must not be empty and in line with func args
    - config.yaml -> cfg.model.predictors lags must also be available in .csv data!

    @param var: the lags of the underlying to compare with
    @param target: the underlying to be compared with its lags
    @param rows: number of rows with subplots
    @param cols: number of cols with subplots 
    @return fig, bar: scatterplot and barplot with correlations
    '''

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
        axs[i].set(xlabel=str(list(lags)[i].split('_lag_')[0]) + ' at t-' + str(list(lags)[i].split('_lag_')[-1]), 
                   ylabel=str(target) + ' at time t')
        axs[i].annotate('r = {}'.format(corrs[i]), xy = (0,1), xytext = (0.05,0.9), xycoords = 'axes fraction')
    

    # barplot of correlations
    df = pd.DataFrame(list(zip(list(lags), corrs)), columns =['lag', 'corr'])
    df = df.sort_values('corr')
    plt.figure(figsize=(12, 5))
    bar = plt.barh(df['lag'], df['corr'])
    plt.title('Correlations with temperature')
    plt.bar_label(bar)
    
    return fig, bar

if __name__ == "__main__":
    lag_analysis(var = 'temperature', target = 'temperature', rows = 3, cols = 3)


    # Lag_24 seems good -> try 25 and 26 -> might be even better than 6 or 9 though it is further away
    # also 48, 47, 49

    print('END')