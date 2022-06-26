# Data Preprocessing

This document contains relevant information on the data structures involved and describes data preprocessing steps. 




Preproc can be controlled via the config.yaml.




As of now, the target variable to predict is the temperature

The following (custom) transformers are applied to the data for feature engineering:

| Class | Description |
|---|---|
| Debugger | This class does nothing but provide a look into the processed data frames. The position of this class can be adjusted arbitrarily within the pipeline to show the data frames after prior transformations. |
| Split |  |
| Time |  |
| Velocity |  |
| Acceleration |  |
| BollingerBand | - not implemented yet - |
| MovingAverage | - not implemented yet - |
| InsertLags |  |
| Scaler |  |
| Prepare |  |
|  |  |

When the pipeline has finished, the resulting data frame looks like this: 
![grafik](https://user-images.githubusercontent.com/52510339/175811052-a2762c6c-81f2-4973-8913-20201878b91b.png)

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













