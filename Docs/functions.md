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

- download()
  - location: src/utils/functions.py
  - arguments:
    - cfg: data_config -> used to control parameters for downloading data: start, end, resolution etc.
  - invokes: DwdObservationRequest()
  - description: This function downloads the raw data from the DWD weather api.

- main_training()
  - location: app.py
  - arguments:
    - target: str -> the target variable of the model training, which is also an element of the list cfg.transform.vars
  - invokes: data_loader(), pipeline_training_preproc(), dict_to_df(), save(), model_data_loader(), train_xgb()
  - description: This function wraps all functions in the process from data sourcing to model training.

- main_inference()
  - description: This function is responbile for the inference procedure. We follwo a 'walking inference' approach which means that we predict ...
