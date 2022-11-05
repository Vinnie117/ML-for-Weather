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










