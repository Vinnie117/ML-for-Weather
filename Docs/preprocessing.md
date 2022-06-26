# Data Preprocessing

This document contains relevant information on the data structures involved and describes data preprocessing steps. 

download.py -> data from API comes out like this / in this form (this repo works for any (hourly) ts data)
-> after preproc it looks like this ...


Preproc can be controlled via the config.yaml.

image.png


As of now, the target variable to predict is the temperature
# To do
- Structure of dict_data / data -> the dictionary which contains all the data -> Bulletpoints
- table of custom data transformers

When the pipeline has finished, the resulting data frame looks like this: 
![grafik](https://user-images.githubusercontent.com/52510339/175811052-a2762c6c-81f2-4973-8913-20201878b91b.png)

The resulting object that holds all information is a dictionary called data and structured as follows:
- data
  - train
  - test
  - train_std
  - test_std













