Description of available functions and an overview of the function landscape:

- main()
  - location: app.py
  - arguments:
    - cfg: data_config
    - api_download: boolean 
    - training: boolean
    - inference: boolean
  - invokes: download(), main_training(), main_inference()
  - description: This is the main function that executes the whole program. It wraps the download of necessary data from the DWD api, the data preprocessing, the model training and inference. Depending on which arguments are set, only certain functionalities are executed. Since main_training() works for a single target variable, we need to loop over the list in cfg.transform.vars in order to train models for all base variables. This allows for a a wholistic inference approach.

<br/>

- download()
  - location: src/utils/functions.py
  - arguments:
    - cfg: data_config -> used to control parameters for downloading data: start, end, resolution etc.
  - invokes: DwdObservationRequest()
  - description: This function downloads the raw data from the DWD weather api and writes the data to the directory data_dvc/raw/training.

<br/>

- main_training()
  - location: app.py
  - arguments:
    - target: str -> the target variable of the model training, which is also an element of the list cfg.transform.vars
  - invokes: data_loader(), pipeline_training_preproc(), dict_to_df(), save(), model_data_loader(), train_xgb()
  - description: This function wraps all functions in the process from data sourcing, data preprocessing and feature engineering to model training. However, it can only do so for a single target variable.

<br/>

- main_inference()
  - location: app.py
  - arguments:
    - cfg: data_config
  - invokes: data_loader(), pipeline_inference_preproc(), walking_inference()
  - description: This function is responsible for the inference procedure. We follow a 'walking inference' approach which means that we subsequently predict new rows of data. Each row, i.e. data for weather at time t requires a complete row of data (with all engineered features) at time t-1.

<br/>

- data_loader()
  - location: src/utils/functions.py
  - arguments:
    - data: str -> indicate purpose of data, e.g. 'training' or 'inference' (see cfg.data)
    - cfg: data_config
  - invokes: no user-defined functions
  - description: This function reads the raw data that has been downloaded by download() from the DWD api and prepares the data for further usage by the preprocessing pipeline. Here, the dataframe is pivoted from long to wide format, unnecessary variables are discarded and variables are renamed.

<br/>

- pipeline_training_preproc()
  - location: src/preprocessing/training_prepoc.py
  - arguments:
    - cfg: data_config
    - target: str -> the target variable of the model training, which is also an element of the list cfg.transform.vars
  - invokes: various custom classes from src/preprocessing/classes_training_preproc.py
  - description: This is a sklearn pipeline which handles all data preparation and feature engineering prior to model training. For this purpose, various custom classes are used that create new features such as velocity or acceleration of base variables. Moreover, time variables and lags are created here. The output of this pipeline is an artefact called 'dict_data' containing multiple folds of training and test data.

<br/>

- dict_to_df()
  - location: src/preprocessing/functions.py
  - arguments:
    - dict_data: dict -> a nested dictionary with training/test data
  - invokes: no user-defined functions
  - description: This function reads a nested dictionary object (the one from pipeline_training_preproc()) and returns the last fold of (standardized) training and test data

<br/>

- save()
  - location: src/preprocessing/functions.py
  - arguments:
    - var: str -> the target variable of the model training, which is also an element of the list cfg.transform.vars
    - train: pandas dataframe
    - test: pandas dataframe
    - train_std: pandas dataframe
    - test_std: pandas dataframe
  - invokes: no user-defined functions
  - descriptions: This function saves dataframes with 'var' being the target variable to the directory data_dvc/processed

<br/>

- model_data_loader()
  - location: src/training/functions.py
  - arguments: 
    - target: str -> the target variable of the model training, which is also an element of the list cfg.transform.vars
  - invokes: no user-defined functions
  - descriptions: Loads and returns training and test data for model training from the directory data_dvc/processed with the respective target variable.

<br/>

- train_xgb()
  - location: src/training/XGB/training_xgboost.py
  - arguments:
    - cfg: data_config -> used for hyperparameter-tuning of model training
    - target: str -> the target variable of the model training, which is also an element of the list cfg.transform.vars
    - X_train: pandas dataframe
    - y_train: pandas dataframe
    - X_test: pandas dataframe
    - y_test: pandas dataframe
  - invokes: eval_metrics(), track_features(); from sklearn.model_selection: TimeSeriesSplit(), GridSearchCV()
  - description: This function executes the model training of an XGBoost model and logs the results in an MLFLow experiment. The incoming training data is split using sklearn's TimeSeriesSplit() and a grid search approach with GridSearchCV() is applied when training multiple models in order to find the best hyperparameters. We use eval_metrics() to calculate different performance metrics on the test data and track_features() to track the features that have been used in order to train the model. Tracked features are stored in 'artifacts/features/data_features.yaml'. Additionally, various other info is measured such as the duration of model training. Results of the experiment can be checked in the MLFlow UI. The model is saved in 'artifacts/models/xgb.joblib' but it is better to retrieve them from 'mlruns' directory.

<br/>

- eval_metrics()
  - location: src/training/functions.py
  - arguments:
    - actual: pandas series -> actual y-values from test data
    - pred: pandas series -> predicted y-values from model
    - X_test: pandas dataframe -> test data not seen during model training
  - invokes: adjusted_R2() -> a custom function to calculate adjusted R2
  - description: This function calculates and returns various performance metrics of the trained model: root mean-squared error, mean absolute error, R2 and adjusted R2.

<br/>

- track_features()
  - lcoation: src/training/functions.py
  - arguments:
    - cfg: data_config
    - X_train: pandas dataframe
  - invokes: no user-defined functions
  - description: This function tracks all features used during model training. It creates and returns a nested dictionary with base variables and their transformations that are stored in a lower level of the nested dict. Since the training of a time-series model can only use past data, the lags of each transformation used for training are stored in the 3rd level of the nested dict. The nested dictionary of features used for training is logged in MLFlow and can also be inspected with MLFlow UI.

<br/>

- pipeline_inference_preproc()
  - location: src/inference/inference.py
  - arguments:
    - cfg: data_config
  - invokes: various custom classes from src/inference/classes_inference_preproc.py
  - description: This is a sklearn pipeline which handles all data preparation and feature engineering prior to model inference. For this purpose, various custom classes are used that create new features such as velocity or acceleration of base variables. Moreover, time variables and lags are created here. The output of this pipeline is a dataframe ready for inference

<br/>

- walking_inference()
  - location: src/inerence/inference.py
  - arguments:
    - cfg: data_config
    - walking_df: pandas dataframe -> the inference-ready dataframe to apply row-wise inference on
    - end_date: pandas TimeStamp -> predictions are made until this point in time
  - invokes: pipeline_inference_complete(), model_loader()
  - description:

<br/>

- pipeline_inference_complete()
  - location: src/inerence/inference.py
  - arguments:
    - cfg: data_config
  - invokes: various custom classes from src/inference/classes_inference_complete.py
  - description: 

<br/>

- model_loader()
  - location: src/inerence/functions.py
  - arguments: None
  - invokes: no user-defined functions
  - description: This function retrieves the best model (by lowest adjusted R2) for each target base variable from the 'mlruns' directory. It scans the 'mlruns" directory by searching for all runs with the tag 'XGB, target' which is set during model training. Then, in a dataframe those models are sorted by their adjusted R2 and grouped by their respective target base variable. The first model in each group of target base variables (lowest adjusted R2 of the group) is fetched together with its 'run_id' and put into a dictionary. That dict is returned.
