Description of available functions and an overview of the function landscape:

- main()
  - location: app.py
  - arguments:
    - cfg, api_download, training, inference
  - invokes: download(), main_training(), main_inference()
  - description: This is the main function that executes the whole program. It wraps the download of necessary data from the DWD api, the data preprocessing, the model training and inference. Depending on which arguments are set, only certain functionalities are executed.
