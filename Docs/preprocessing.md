# Data Preprocessing

This document contains relevant information on the data structures involved and describes data preprocessing steps. 




Preproc can be controlled via the config.yaml.




As of now, the target variable to predict is the temperature

The following (custom) transformers are applied to the data for feature engineering:

| Class | Description | Arguments |
|---|---|---|
| Debugger | This class does nothing but provide a look into the processed data frames. The position of this class can be adjusted  arbitrarily within the pipeline to show the data frames after prior transformations. |  |
| Split | This transformer splits data into train and test sets. It is to be called at the beginning of the pipeline in order to  prevent information leakage. The transformer also splits train/test data into chronological folds. Due to the nature of time series, each subsequent split contains data from the prior fold(s), i.e. shuffling single observed rows for time  series data is not appropriate. | - n_splits: the number of folds to create |
| Time | The transformer takes a hourly timestamp and splits it into 3 variables: month, day and hour. |  |
| Velocity |  |  |
| Acceleration |  |  |
| BollingerBand | - not implemented yet - |  |
| MovingAverage | - not implemented yet - |  |
| InsertLags | This transformer creates lags of all previous features (except time variables and the target itself). The lagged time  series must be used when predicting the target because a model predicting a variable at time t only observes the predictors at time t-1. | - diff: length of the lag |
| Scaler | This transformer standardizes (z-score) all previous features. It appends separate keys (['train_std'] and [test_std']) to  the data dictionary for storing the standardized values. The parameters of standardization have to be calculated solely on  the training data and when applying the standardization itself on test data, those same parameters have to be utilized.  The calculation of the standardization parameters relies on the last fold of training data, since the last folds contains data from all prior folds (chronological order of time series). | - std_target: if target variable should be standardized or not |
| Prepare |  |  |
|  |  |  |

When the pipeline has finished, the resulting data frame looks like this: 
![grafik](https://user-images.githubusercontent.com/52510339/175814437-8152c8bd-b0b3-4e79-9f51-ec09337d31fb.png)


The resulting object that holds all information is a dictionary called data and structured as follows:
- data
  - train
    - train_fold_0
    - ...
    - train_fold_K
  - test
    - test_fold_0
    - ...
    - test_fold_K
  - train_std
    - train_fold_0
    - ...
    - train_fold_K
  - test_std
    - test_fold_0
    - ...
    - test_fold_K
  - pd_df













